/*
 * profzone.c - X280 profiler drainer: 2-reader + 1-collect + 1-relay hart split (Inc-1, D8).
 *
 * Inc-1 of the per-(core,risc) LIM-mirror redesign (see PROFILER_PACKET_PIPELINE.md D8). Goal:
 * make (core,risc) identity STRUCTURAL (a mirror's address encodes it) and move all reshape off
 * the reader read-release path onto a dedicated collect hart. THIS increment is a pure-drain
 * baseline: the collect hart does NO reshape and NO identity stamping -- it just merges the
 * per-(core,risc) mirrors into one SPSC so we can measure how busy that hart is just collecting.
 *
 * Pipeline (4 harts, all of them):
 *   - readers (0..nread-1): each owns a DISJOINT core subset. Per (core,risc) L1 ring, LOSSLESS
 *     direct-read of [head,tail) and bulk-copy the raw 2-word markers into that ring's OWN LIM
 *     mirror SPSC (MIRROR[core_idx*5 + risc]). Advances the L1 head only after the mirror write
 *     (producer unblocks). Blocks if the mirror is full (waits on the collect hart). No reshape.
 *   - collect (hart nread): round-robins ALL mirrors; drains each into the single SPSC, packed
 *     8 raw markers / 64B page (partial pages zero-padded so host skips them). No reshape/stamp.
 *   - relay (hart nread+1): pure page-copy of the single SPSC to the ONE D2H socket FIFO (NOC1).
 * End-to-end lossless: producers block on full L1 rings; readers block on full mirrors; collect
 * blocks on a full single SPSC; relay blocks on a full D2H FIFO. Nothing dropped (lap-guard clamps).
 *
 * SPSC L1 contract (profiler_msg_t @ prof_l1): control_vector[32] then buffer[r] @ +128 + r*2048;
 * head=ctrl[r], tail=ctrl[5+r] (monotonic WORD counts), storage idx = count%512.
 * Marker = 2 words: w0 = 0x80000000 | (timer_id<<12) | time_H, w1 = time_L. type = (timer_id>>16)&7.
 *
 * LIM map:
 *   PARAMS  @ 0x08011000 : +0x00 config_addr +0x08 pcie_x +0x10 pcie_y +0x18 prof_l1
 *                          +0x20 num_cores +0x28 stop +0x30 nread +0x38 nharts
 *   RESULTS @ 0x08011040 : relay +0x00 total_pages +0x08 loops +0x18 done +0x20 stalls
 *                          +0xB0 wall +0xB8 reserve +0xC0 copy; hart0 +0x30 heartbeat(0xB007);
 *                          readers +0x50+h*8 dropped +0x60 bulk_words +0x70 segs +0x80 wall
 *                          +0x90 passes +0xA0 polls;  collect +0xC8 markers +0xD0 loops
 *                          +0xD8 wall +0xE0 empty-spin +0xE8 copy.
 *   COORDS  @ 0x08011200 : num_cores x { u32 core_x, u32 core_y } (as relayed; host translates).
 *   MIRRORCTL @ 0x08018000 : per mirror i (16 B): +0 head, +8 tail  (u32 monotonic WORD counts).
 *   SINGLECTL @ 0x0801C000 : +0x00 prod, +0x08 cons (u64 page counts); +0x10 collect_done;
 *                            +0x20+h*8 reader_done[h].
 *   MIRROR  @ 0x08040000 : mirror i storage @ +i*MIRROR_STRIDE, MIRROR_DEPTH words (2 KiB).
 *   SINGLE  @ MIRROR_BASE + (num_cores*5)*MIRROR_STRIDE : SINGLE_NREC x 64 B pages (dynamic base).
 * Host pre-zeros MIRRORCTL + SINGLECTL before boot (deterministic, no init race).
 */
#include <stdint.h>

#include "noc.h"
#include "x280_boot.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL
#define P_CONFIG_ADDR (MBOX_PARAMS + 0x00)
#define P_PCIE_X (MBOX_PARAMS + 0x08)
#define P_PCIE_Y (MBOX_PARAMS + 0x10)
#define P_PROF_L1 (MBOX_PARAMS + 0x18)
#define P_NUM_CORES (MBOX_PARAMS + 0x20)
#define P_STOP (MBOX_PARAMS + 0x28)
#define P_NREAD (MBOX_PARAMS + 0x30)
#define P_NHARTS (MBOX_PARAMS + 0x38)
#define RES(o) (MBOX_RESULTS + (o))
#define RES_DONE 0x18
#define DONE_MAGIC 0x20E50FFEE1ULL
/* Per-hart boot heartbeat @ RES(0x100 + hartid*8): 1=entered main, 2=passed guards, 3=reached the
 * work loop. The host pre-zeros these and waits for EVERY hart to reach 3 before trusting the drainer
 * -- the X280 multi-hart release_reset is intermittently flaky (a hart can fail to start), and a
 * missing reader/collect/relay silently cripples the pipeline (undrained cores -> workers block ->
 * trace wedges). Host retries the reset if any hart is missing. */
#define HART_STAGE(h) RES(0x100 + (uint64_t)(h) * 8)

#define NRISC 5
#define RING_CAP 512u /* producer L1 ring depth (words) */
/* Idle poll backoff: after a whole pass drains nothing, spin ~this many cycles (X280 ~1 GHz => ~ns)
 * before re-polling. A PRODUCTIVE pass skips it, so real bursts still poll at full rate.
 * RETUNED 500000 (~500 us) -> 5000 (~5 us): the reader-wall breakdown (FINDINGS §23) showed the reader
 * spent 86% of its wall asleep here, and its response latency (~500 us) -- not throughput -- was what
 * back-pressured the producers: a marker burst landing during the 500 us nap filled a core's L1 ring
 * and blocked it. Each poll is only ~0.3 us and a full 55-core sweep ~16 us, so a 5 us backoff keeps
 * the reader responsive (drains rings before they fill) while still yielding NoC between idle sweeps. */
#define POLL_BACKOFF_CYC 5000u
#define WRITE_WIN 200u
#define PAGE 64u
#define CTRL_HEAD(r) (r)
#define CTRL_TAIL(r) (5u + (r))

/* socket sender_socket_md word offsets @ config_addr */
#define C_BYTES_SENT 0
#define C_WRITE_PTR 2
#define C_DN_BSENT_LO 3
#define C_DN_FIFO_LO 4
#define C_FIFO_TOTAL 5
#define C_BACKED 8
#define C_BSENT_HI 12
#define C_FIFO_HI 13

/* Per-(core,risc) LIM mirror SPSC. Depth = producer ring (512 words = 2 KiB) so a reader never
 * blocks mid-segment. Index i = core_idx*NRISC + risc; storage is dense over the drained core set. */
#define MIRROR_DEPTH 512u /* words per mirror (== RING_CAP) */
#define MIRROR_BASE 0x08040000UL
#define MIRROR_STRIDE (MIRROR_DEPTH * 4u) /* 2048 B */
#define MSTORE(i) (MIRROR_BASE + (uint64_t)(i) * MIRROR_STRIDE)
#define MIRRORCTL 0x08018000UL
#define MHEAD(i) (MIRRORCTL + (uint64_t)(i) * 16 + 0) /* collect advances (consumer) */
#define MTAIL(i) (MIRRORCTL + (uint64_t)(i) * 16 + 8) /* reader advances (producer) */
#define MAX_CORES 140u                                /* LIM cap: mirrors + single must fit < LIM end */
#define LIM_END 0x081E0000UL                          /* 0x08000000 + 1.875 MiB */

/* Single SPSC (collect -> relay), raw markers packed 8/64 B page so the relay stays a page-copy.
 * Base is placed right after the USED mirrors (dynamic in num_cores); ctrl is fixed. */
#define SINGLE_NREC 4096u
#define SINGLECTL 0x0801C000UL
#define S_PROD (SINGLECTL + 0x00)
#define S_CONS (SINGLECTL + 0x08)
#define COLLECT_DONE (SINGLECTL + 0x10)
#define READER_DONE(h) (SINGLECTL + 0x20 + (uint64_t)(h) * 8)
/* Collect-hart stats handed to the relay (coherent X280-internal LIM). The collect hart wfi's right
 * after writing, so its own RES cache line may never write back to the SRAM the host NoC-reads (and
 * the relay shares that line) -> the collect telemetry read back as 0. Fix: collect writes here, the
 * relay (whose RES writes ARE host-visible) copies these into RES 0xC8..0xE8 before DONE_MAGIC. */
#define COLLECT_STATS (SINGLECTL + 0x40) /* +0 moved +8 loops +16 wall +24 empty +32 copy (u64) */

/* per-reader ILP bulk-read scratch in LIM (for the 16-word ctrl poll). 4 KiB stride. */
#define SCRATCH_BASE 0x08012000UL
#define SCRATCH_STRIDE 0x1000UL
#define SCRATCH(h) (SCRATCH_BASE + (uint64_t)(h) * SCRATCH_STRIDE)

/* Live queue-fill gauges: each stage writes its input-queue CURRENT fill so the host can sample
 * them over time and reconstruct the fill-vs-time trajectory (which queue climbs to capacity, and
 * when). Fixed LIM slots in the free gap between the boot mailboxes (0x08016000) and MIRRORCTL. */
#define GAUGE_BASE 0x08016400UL
#define G_L1(h) (GAUGE_BASE + (uint64_t)(h) * 8) /* max L1 ring fill (words, /RING_CAP=512) this pass */
#define G_MIRROR (GAUGE_BASE + 0x20)             /* max LIM mirror fill (words, /MIRROR_DEPTH) this loop */
#define G_SINGLE (GAUGE_BASE + 0x28)             /* single-SPSC fill (pages, /SINGLE_NREC) */
#define G_D2H (GAUGE_BASE + 0x30)                /* D2H FIFO fill (bytes, /fifo_total) */

/* Fill-trajectory ring: the relay (idle-ish) appends a time-stamped snapshot of all four gauges
 * every TRAJ_PERIOD_CYC, so the host can reconstruct fill-vs-time (which queue climbs to capacity,
 * and when -- e.g. does a queue ramp monotonically until ~op 90?). Fixed region between SINGLECTL
 * and the mirrors (0x08040000), so placement is independent of num_cores. 32 B/sample. */
#define TRAJ_BASE 0x0801D000UL
#define TRAJ_CAP 4096u /* 4096 * 32 B = 128 KiB, ends 0x0803D000 < MIRROR_BASE (0x08040000) */
#define TRAJ_STRIDE 32u
#define TRAJ_COUNT (SINGLECTL + 0x100) /* total samples written (host reads count, then the ring) */
#define TRAJ_PERIOD_CYC 200000u        /* ~200 us @ 1 GHz -> 4096 samples covers ~0.8 s (ring wraps) */

static inline uint32_t r32(uint64_t a) { return *(volatile uint32_t*)a; }
static inline void w32(uint64_t a, uint32_t v) { *(volatile uint32_t*)a = v; }
static inline uint64_t r64(uint64_t a) { return *(volatile uint64_t*)a; }
static inline void w64(uint64_t a, uint64_t v) { *(volatile uint64_t*)a = v; }
static inline void fence_(void) { __asm__ volatile("fence iorw, iorw"); }
static inline uint64_t rdcycle_(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}
/* copy one 64 B page as a single wide vector load+store (LIM->FIFO or LIM->LIM) */
static inline void page_copy(uint64_t src, uint64_t dst) {
    __asm__ volatile("vsetivli zero, 8, e64, m1, ta, ma\n vle64.v v0, (%0)\n vse64.v v0, (%1)\n"
                     :
                     : "r"(src), "r"(dst)
                     : "memory", "v0");
}
/* bulk copy `nwords` (even) src->dst with wide LMUL=8 vectors (many beats in flight). Works for an
 * uncached NoC System-Port src (L1 ring) or LIM src (mirror). */
static inline void bulk_copy_words(uint64_t dst, uint64_t src, uint32_t nwords) {
    uint32_t nel = nwords >> 1; /* e64 doublewords */
    while (nel > 0) {
        uint64_t vl;
        __asm__ volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl) : "r"((uint64_t)nel));
        __asm__ volatile("vle64.v v0, (%0)\n vse64.v v0, (%1)\n"
                         :
                         : "r"(src), "r"(dst)
                         : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
        src += vl * 8u;
        dst += vl * 8u;
        nel -= (uint32_t)vl;
    }
}

/* Marker packet-type bits: type = (w0 >> 28) & 7. From hostdevcommon PacketTypes. */
#define PT_ZONE_START 0u
#define PT_ZONE_END 1u
#define PT_STICKY_META 6u

/* Inc-2a: the collect hart RESHAPES each raw 2-word marker into a 28B WorkerZoneWire and packs 2 per
 * 64B single-SPSC page (relay stays a pure page-copy). WorkerZoneWire (realtime_profiler_packets.hpp):
 *   +0 header{u16 type, u16 reserved}  +4 core_x  +8 core_y  +12 risc  +16 timer_id  +20 time_hi  +24 time_lo
 * header.type = WorkerZone = 0; we repurpose header.reserved as the WIRE VALIDITY sentinel (0xA5A5 =
 * valid, 0 = pad) since WorkerZone==0 can't itself flag validity. Identity (core_x/y, risc) is
 * STRUCTURAL -- taken from the mirror index -- so no STICKY_META / host forward-fill (kills orphans). */
#define WZ_VALID 0xA5A5u
#define WZW_BYTES 28u
static inline void w_wzw(
    uint64_t dst, uint32_t cx, uint32_t cy, uint32_t risc, uint32_t timer_id, uint32_t time_hi, uint32_t time_lo) {
    w32(dst + 0, (uint32_t)(WZ_VALID << 16)); /* header: type=0(WorkerZone) low16 | reserved=0xA5A5 high16 */
    w32(dst + 4, cx);
    w32(dst + 8, cy);
    w32(dst + 12, risc);
    w32(dst + 16, timer_id);
    w32(dst + 20, time_hi);
    w32(dst + 24, time_lo);
}

/* ---- boot-handoff: return to the resident idle FW instead of parking in wfi ----
 * profzone is now an ACTIVE FW: the idle FW (x280_boot.h) JUMPed all 4 harts into
 * _start; on shutdown they must jump BACK to the idle FW so the next run can be
 * handed off without ever touching reset. */
static inline void cpu_pause(void) {
    __asm__ volatile(".word 0x0100000F" ::: "memory"); /* Zihintpause: quiescent spin */
}
static inline __attribute__((noreturn)) void x280_jump(uint64_t entry) {
    /* fence.i: the L2CPU is never reset across handoffs, so its I-cache still holds
     * the idle FW's bytes; harmless here (idle code is unchanged) but kept for symmetry. */
    __asm__ volatile("fence ow, ow\n fence.i\n jr %0\n" : : "r"(entry) : "memory");
    __builtin_unreachable();
}
/* Helper harts (reader-1, collect, relay, or any hart beyond nharts) re-enter the
 * idle FW without touching the boot phase. */
static inline __attribute__((noreturn)) void helper_to_idle_fw(void) { x280_jump(X280_IDLE_FW_LOAD_ADDR); }
/* Hart 0 (reader-0) is the return coordinator: the idle FW's hart-0 _start re-arms
 * the heartbeat/mailboxes for the next handoff, so hart 0 MUST be the one to
 * re-enter idle -- but only AFTER the relay has flushed the whole pipeline
 * (DONE_MAGIC), so a fresh JUMP can't be accepted mid-drain. */
static inline __attribute__((noreturn)) void return_to_idle_fw(void) {
    *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_RETURNED_TO_IDLE;
    __asm__ volatile("fence ow, ow");
    x280_jump(X280_IDLE_FW_LOAD_ADDR);
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        w64(RES(0x30), 0xB007ULL); /* heartbeat: main() entered (hart 0) -- kept for the prime check */
        /* tell the host + idle FW the JUMP landed and the active FW is running */
        *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_RUNNING_ACTIVE_FW;
    }
    w64(HART_STAGE(hartid), 1); /* per-hart boot heartbeat: entered main */
    fence_();
    uint64_t nread = r64(P_NREAD);
    uint64_t nharts = r64(P_NHARTS);
    uint64_t num_cores = r64(P_NUM_CORES);
    uint64_t prof_l1 = r64(P_PROF_L1);
    uint64_t ctrl_off = prof_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;

    /* Single SPSC storage lives right after the USED mirrors (dense over the drained core set). */
    uint64_t nmirror = num_cores * NRISC;
    uint64_t single_store = MIRROR_BASE + nmirror * MIRROR_STRIDE;

    if (hartid >= nharts) {
        helper_to_idle_fw(); /* not used this run (hartid>=1 here) -- return to idle */
    }
    /* LIM overflow guard: too many cores would run the mirrors + single ring past LIM end. */
    if (num_cores > MAX_CORES || single_store + (uint64_t)SINGLE_NREC * PAGE > LIM_END) {
        if (hartid == 0) {
            w64(RES(0x40), 0xBAD00000ULL | num_cores); /* signal misconfig; host sees no DONE_MAGIC */
            return_to_idle_fw();                       /* re-arm idle so the host isn't stuck (no pipeline ran) */
        }
        helper_to_idle_fw();
    }

    w64(HART_STAGE(hartid), 2); /* passed the nharts + LIM-overflow guards */
    fence_();
    if (hartid < nread) {
        /* ============ READER: L1 [head,tail) -> per-(core,risc) LIM mirror (raw, no reshape) ====== */
        uint64_t q = (num_cores + nread - 1) / nread;
        uint64_t lo = hartid * q, hi = lo + q;
        if (hi > num_cores) {
            hi = num_cores;
        }
        for (uint64_t c = lo; c < hi; c++) {
            (void)noc_configure_tlb_2m((uint32_t)c, coords[c * 2 + 0], coords[c * 2 + 1], prof_l1, 0, 0);
        }
        fence_();
        if (hartid == 0) {
            w64(RES(0x38), 0x5E70ULL); /* setup complete */
        }

        w64(HART_STAGE(hartid), 3); /* reader: setup done, entering drain loop */
        fence_();
        uint64_t dropped = 0;
        uint64_t scratch = SCRATCH(hartid);
        uint64_t bulk_words = 0, segs = 0;
        uint64_t passes = 0, polls = 0;
        /* Reader-wall breakdown. rdcycle is cheap on X280 (~ns, not a trap) -- last run's wall was
         * identical with/without these timers, so the earlier "unaccounted" was MISSING COVERAGE, not
         * contamination. Full coverage: wall ~= poll + bulk + mwait + backoff + "other"(parse/scan).
         * mwait_spins is a dilution-immune COUNT: high => reader blocked on a FULL mirror => collect
         * (the reshape hart) is the burst wall, not the reader itself. */
        uint64_t cyc_poll = 0, cyc_bulk = 0, cyc_mwait = 0, cyc_backoff = 0, mwait_spins = 0;
        uint64_t t_start = rdcycle_();
        for (;;) {
            uint64_t progressed = 0;
            passes++;
            uint32_t pass_l1max = 0; /* fullest L1 ring (words) seen this pass -> G_L1 gauge */
            for (uint64_t c = lo; c < hi; c++) {
                uint64_t cbase = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off;
                uint64_t rbufs = cbase + 128;
                /* Poll all 5 RISCs' head/tail in ONE ILP burst: ctrl[0..4]=head, ctrl[5..9]=tail. */
                uint64_t tp = rdcycle_();
                bulk_copy_words(scratch, cbase, 16);
                fence_();
                cyc_poll += rdcycle_() - tp;
                polls++;
                uint32_t hd[NRISC], tl[NRISC];
                {
                    volatile uint32_t* cv = (volatile uint32_t*)scratch;
                    for (uint32_t r = 0; r < NRISC; r++) {
                        hd[r] = cv[r];
                        tl[r] = cv[5u + r];
                    }
                }
                for (uint32_t r = 0; r < NRISC; r++) {
                    uint32_t tail = tl[r];
                    uint32_t head = hd[r];
                    uint32_t l1fill = tail - head; /* gauge: how full is this producer's L1 ring */
                    if (l1fill > pass_l1max) {
                        pass_l1max = l1fill;
                    }
                    if (head == tail) {
                        continue;
                    }
                    progressed = 1;
                    if ((uint32_t)(tail - head) > RING_CAP) { /* lap guard (blocking keeps <= RING_CAP) */
                        dropped += (uint32_t)(tail - head) - RING_CAP;
                        head = tail - RING_CAP;
                    }
                    uint64_t ring_base = rbufs + (uint64_t)r * 2048;
                    uint32_t gi = (uint32_t)(c * NRISC + r); /* mirror index: identity by location */
                    uint32_t mtail = r32(MTAIL(gi));         /* this reader is the sole producer of gi */
                    uint32_t h = head;
                    while (h != tail) {
                        uint32_t hidx = h % RING_CAP;
                        uint32_t seg = RING_CAP - hidx; /* contiguous L1 words until L1 wraps */
                        uint32_t rem = (uint32_t)(tail - h);
                        if (seg > rem) {
                            seg = rem;
                        }
                        /* reserve `seg` words in the mirror; block on the collect hart (consumer). */
                        uint64_t tmw = rdcycle_();
                        while ((uint32_t)(mtail + seg - r32(MHEAD(gi))) > MIRROR_DEPTH) {
                            mwait_spins++; /* reader blocked on a FULL mirror => collect is the wall */
                            if (r64(P_STOP)) {
                                cyc_mwait += rdcycle_() - tmw;
                                goto reader_done;
                            }
                        }
                        cyc_mwait += rdcycle_() - tmw;
                        uint32_t midx = mtail % MIRROR_DEPTH;
                        uint32_t mfit = MIRROR_DEPTH - midx; /* words until mirror wraps */
                        uint64_t l1src = ring_base + (uint64_t)hidx * 4;
                        uint64_t tb = rdcycle_();
                        if (seg <= mfit) {
                            bulk_copy_words(MSTORE(gi) + (uint64_t)midx * 4, l1src, seg);
                        } else {
                            bulk_copy_words(MSTORE(gi) + (uint64_t)midx * 4, l1src, mfit);
                            bulk_copy_words(MSTORE(gi), l1src + (uint64_t)mfit * 4, seg - mfit);
                        }
                        cyc_bulk += rdcycle_() - tb;
                        mtail += seg;
                        fence_();              /* mirror bytes visible before the tail bump */
                        w32(MTAIL(gi), mtail); /* publish to collect */
                        bulk_words += seg;
                        segs++;
                        h += seg;
                        w32(cbase + CTRL_HEAD(r) * 4, h); /* advance L1 head -> producer unblocks */
                    }
                }
            }
            w64(G_L1(hartid), pass_l1max); /* publish fullest L1 ring this pass (host samples over time) */
            if (!progressed) {
                if (r64(P_STOP)) {
                    break;
                }
                uint64_t tbk = rdcycle_();
                while ((rdcycle_() - tbk) < POLL_BACKOFF_CYC) {
                    if (r64(P_STOP)) {
                        break;
                    }
                }
                cyc_backoff += rdcycle_() - tbk;
            }
        }
    reader_done:
        fence_();
        w64(RES(0x50 + hartid * 8), dropped);
        w64(RES(0x60 + hartid * 8), bulk_words);
        w64(RES(0x70 + hartid * 8), segs);
        w64(RES(0x80 + hartid * 8), rdcycle_() - t_start);
        w64(RES(0x90 + hartid * 8), passes);
        w64(RES(0xA0 + hartid * 8), polls);
        /* reader-wall breakdown -> RES 0x140/0x150/0x160/0x170/0x120 + h*8 (host reads at
         * params+0x180/0x190/0x1A0/0x1B0/0x160). */
        w64(RES(0x140 + hartid * 8), cyc_poll);
        w64(RES(0x150 + hartid * 8), cyc_bulk);
        w64(RES(0x160 + hartid * 8), cyc_mwait);
        w64(RES(0x170 + hartid * 8), cyc_backoff);
        w64(RES(0x120 + hartid * 8), mwait_spins);
        w64(READER_DONE(hartid), 1); /* tell the collect hart this reader is finished */
        if (hartid == 0) {
            /* coordinator: wait for the relay to flush the whole pipeline (DONE_MAGIC written
             * last) before re-arming idle, so a fresh JUMP can't be accepted mid-drain. */
            while (r64(RES(0x18)) != DONE_MAGIC) {
                cpu_pause();
            }
            return_to_idle_fw();
        }
        helper_to_idle_fw();
    }

    if (hartid == nread) {
        /* ==== COLLECT (Inc-2a): round-robin mirrors, RESHAPE each raw marker -> WorkerZoneWire with
         * STRUCTURAL (core,risc) from the mirror index, pack 2/64B page into the single SPSC. ==== */
        w64(HART_STAGE(hartid), 3); /* collect: entering drain loop */
        fence_();
        /* per-mirror consumer head (collect owns MHEAD). Local cache avoids re-reading LIM. */
        static uint32_t chead[MAX_CORES * NRISC];
        for (uint64_t i = 0; i < nmirror; i++) {
            chead[i] = 0;
        }
        uint32_t single_prod = 0;
        uint32_t pending = 0;          /* 0 or 1 WorkerZoneWire already written to the in-progress page */
        uint64_t moved = 0, loops = 0; /* moved = WorkerZoneWire records emitted (zone start/end only) */
        uint64_t cyc_copy = 0;         /* time in the reshape+pack path (INCLUDES cyc_swait) */
        /* collect breakdown: cyc_swait/swait_spins = blocked on a FULL single-SPSC (relay back-pressure;
         * expect ~0 since relay is idle -> confirms collect is limited by its OWN work, not downstream).
         * scanned/productive = round-robin volume: how much of the ~550-mirror sweep is empty vs found data. */
        uint64_t cyc_swait = 0, swait_spins = 0, scanned = 0, productive = 0;
        uint64_t t_start = rdcycle_();
        for (;;) {
            uint64_t progressed = 0;
            loops++;
            uint32_t loop_mirror_max = 0; /* fullest mirror (words) this loop -> G_MIRROR gauge */
            for (uint64_t i = 0; i < nmirror; i++) {
                uint32_t mtail = r32(MTAIL(i));
                uint32_t mh = chead[i];
                scanned++; /* round-robin visit (most are empty during idle) */
                uint32_t mfill = mtail - mh;
                if (mfill > loop_mirror_max) {
                    loop_mirror_max = mfill;
                }
                if (mh == mtail) {
                    continue;
                }
                productive++; /* this mirror had data */
                /* STRUCTURAL identity: mirror i IS (core_idx, risc). No STICKY_META / forward-fill. */
                uint32_t core_idx = (uint32_t)(i / NRISC);
                uint32_t risc = (uint32_t)(i % NRISC);
                uint32_t cx = coords[core_idx * 2 + 0];
                uint32_t cy = coords[core_idx * 2 + 1];
                while (mh != mtail) {
                    uint32_t midx = mh % MIRROR_DEPTH;
                    uint32_t seg = MIRROR_DEPTH - midx; /* contiguous mirror words until wrap */
                    uint32_t rem = mtail - mh;
                    if (seg > rem) {
                        seg = rem;
                    }
                    volatile uint32_t* mk = (volatile uint32_t*)(MSTORE(i) + (uint64_t)midx * 4);
                    uint64_t tcp = rdcycle_();
                    for (uint32_t k = 0; k + 1u < seg; k += 2u) { /* each marker = 2 words */
                        uint32_t w0 = mk[k];
                        uint32_t type = (w0 >> 28) & 0x7u;
                        if (type != PT_ZONE_START && type != PT_ZONE_END) {
                            continue; /* drop STICKY_META (identity is structural) + non-zone types */
                        }
                        uint32_t timer_id = (w0 >> 12) & 0x7FFFFu; /* (type<<16)|hash */
                        uint32_t thi = w0 & 0xFFFu;
                        uint32_t w1 = mk[k + 1];
                        if (pending == 0) {
                            /* start a new single-SPSC page (2 records/page): reserve 1 page (block on relay) */
                            uint64_t tsw = rdcycle_();
                            while ((uint32_t)(single_prod + 1u - r32(S_CONS)) > SINGLE_NREC) {
                                swait_spins++; /* blocked on FULL single-SPSC => relay is the wall */
                                if (r64(P_STOP)) {
                                    cyc_swait += rdcycle_() - tsw;
                                    cyc_copy += rdcycle_() - tcp;
                                    goto collect_done;
                                }
                            }
                            cyc_swait += rdcycle_() - tsw;
                            w_wzw(
                                single_store + (uint64_t)(single_prod % SINGLE_NREC) * PAGE,
                                cx,
                                cy,
                                risc,
                                timer_id,
                                thi,
                                w1);
                            pending = 1;
                        } else {
                            uint64_t pg = single_store + (uint64_t)(single_prod % SINGLE_NREC) * PAGE;
                            w_wzw(pg + WZW_BYTES, cx, cy, risc, timer_id, thi, w1);
                            fence_();
                            single_prod++; /* page complete (2 records) */
                            w32(S_PROD, single_prod);
                            pending = 0;
                        }
                        moved++;
                    }
                    cyc_copy += rdcycle_() - tcp;
                    mh += seg;
                    w32(MHEAD(i), mh); /* free mirror -> reader unblocks */
                    chead[i] = mh;
                    progressed = 1;
                }
            }
            w64(G_MIRROR, loop_mirror_max); /* publish fullest mirror this loop (host samples over time) */
            w64(COLLECT_STATS + 0, moved);
            w64(COLLECT_STATS + 8, loops);
            if (!progressed) {
                /* done when stopping AND every reader finished AND all mirrors drained. */
                if (r64(P_STOP)) {
                    uint64_t all = 1;
                    for (uint64_t h = 0; h < nread; h++) {
                        if (!r64(READER_DONE(h))) {
                            all = 0;
                        }
                    }
                    if (all) {
                        uint64_t empty = 1;
                        for (uint64_t i = 0; i < nmirror; i++) {
                            if (chead[i] != r32(MTAIL(i))) {
                                empty = 0;
                                break;
                            }
                        }
                        if (empty) {
                            break;
                        }
                    }
                }
            }
        }
    collect_done:;
        /* Flush a half-filled page: pad slot 1 (WorkerZoneWire header.reserved=0 -> host skips it) and
         * publish, so the last odd record isn't stranded. (A full 2-record page publishes inline above.) */
        if (pending) {
            uint64_t pg = single_store + (uint64_t)(single_prod % SINGLE_NREC) * PAGE;
            for (uint32_t w = 0; w < WZW_BYTES / 4u; w++) {
                w32(pg + WZW_BYTES + (uint64_t)w * 4u, 0); /* zero slot 1 -> reserved=0 -> skipped */
            }
            fence_();
            single_prod++;
            w32(S_PROD, single_prod);
            pending = 0;
        }
        const uint64_t cwall = rdcycle_() - t_start;
        w64(COLLECT_STATS + 0, moved);
        w64(COLLECT_STATS + 8, loops);
        w64(COLLECT_STATS + 16, cwall);            /* collect wall */
        w64(COLLECT_STATS + 24, cwall - cyc_copy); /* empty-spin (round-robin over idle mirrors) */
        w64(COLLECT_STATS + 32, cyc_copy);         /* copy time (reshape + swait) */
        w64(COLLECT_STATS + 40, cyc_swait);        /* subset of copy: blocked on relay (single-SPSC full) */
        w64(COLLECT_STATS + 48, swait_spins);
        w64(COLLECT_STATS + 56, scanned);    /* total mirror-visits (round-robin volume) */
        w64(COLLECT_STATS + 64, productive); /* mirror-visits that found data */
        fence_();                            /* stats visible to the relay before the done flag */
        w64(COLLECT_DONE, 1);
        helper_to_idle_fw(); /* collect done -- return to the resident idle FW */
    }

    /* ================= RELAY: single SPSC -> ONE D2H socket FIFO (NOC1) ========================= */
    w64(HART_STAGE(hartid), 3); /* relay: entering drain loop */
    fence_();
    uint64_t cfg = r64(P_CONFIG_ADDR);
    uint64_t pcie_x = r64(P_PCIE_X), pcie_y = r64(P_PCIE_Y);
    volatile uint32_t* c = (volatile uint32_t*)cfg;
    uint32_t write_ptr = c[C_WRITE_PTR];
    uint32_t fifo_total = c[C_FIFO_TOTAL];
    uint64_t fifo_addr = ((uint64_t)c[C_FIFO_HI] << 32) | c[C_DN_FIFO_LO];
    uint64_t bsent_addr = ((uint64_t)c[C_BSENT_HI] << 32) | c[C_DN_BSENT_LO];
    uint64_t backed_addr = cfg + (uint64_t)C_BACKED * 4;
    uint32_t bytes_sent = c[C_BYTES_SENT];

    noc_tlb_2m_t wt;
    wt.data[0] = 0;
    wt.data[1] = 0;
    wt.data[2] = 0;
    wt.data[3] = 0;
    wt.addr = fifo_addr >> 21;
    wt.x_end = (uint32_t)pcie_x;
    wt.y_end = (uint32_t)pcie_y;
    wt.x_start = (uint32_t)pcie_x;
    wt.y_start = (uint32_t)pcie_y;
    wt.posted = 1;
    wt.noc_selector = 1; /* NOC1: relay writes split off the readers' NOC0 reads */
    (void)noc_configure_tlb_2m_ext(WRITE_WIN, &wt, 0);
    uint64_t wbase = NOC_2M_WINDOW_BASE + (uint64_t)WRITE_WIN * NOC_2M_WINDOW_STRIDE;
    uint64_t fifo_off = fifo_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t bsent_off = bsent_addr & (NOC_2M_WINDOW_STRIDE - 1ULL);
    wt.addr = bsent_addr >> 21;
    wt.posted = 0; /* NON-POSTED bytes_sent so it lands promptly (host frees FIFO room) */
    (void)noc_configure_tlb_2m_ext(WRITE_WIN + 1u, &wt, 0);
    uint64_t wbase_bsent = NOC_2M_WINDOW_BASE + (uint64_t)(WRITE_WIN + 1u) * NOC_2M_WINDOW_STRIDE;
    fence_();

    uint32_t sig_ctr = 0;
    uint32_t cn = 0; /* single-SPSC consumer page index */
    uint64_t total = 0, loops = 0, stalls = 0;
    uint64_t cyc_reserve = 0, cyc_copy = 0;
    uint64_t t_start = rdcycle_();
    uint64_t traj_next = t_start; /* fill-trajectory sampler cadence */
    uint64_t traj_n = 0;
    w64(TRAJ_COUNT, 0);
    for (;;) {
        uint64_t progressed = 0;
        uint32_t pr = r32(S_PROD);
        while (cn != pr) {
            int stopped = 0;
            uint64_t rs = 0;
            uint64_t trs = rdcycle_();
            for (;;) { /* reserve one page of FIFO space (bytes in flight = bytes_sent - acked) */
                fence_();
                uint32_t acked = r32(backed_addr);
                if (fifo_total - (bytes_sent - acked) >= PAGE) {
                    break;
                }
                stalls++;
                if ((stalls & 0xFFFFF) == 0) {
                    w64(RES(0x10), bytes_sent);
                    w64(RES(0x20), stalls);
                }
                if (r64(P_STOP) && ++rs > 50000000ull) {
                    stopped = 1;
                    break;
                }
            }
            cyc_reserve += rdcycle_() - trs;
            if (stopped) {
                goto relay_done;
            }
            uint64_t tcp = rdcycle_();
            page_copy(single_store + (uint64_t)(cn % SINGLE_NREC) * PAGE, wbase + fifo_off + write_ptr);
            write_ptr += PAGE;
            if (write_ptr >= fifo_total) {
                write_ptr -= fifo_total;
            }
            bytes_sent += PAGE;
            if (++sig_ctr >= 256u) {
                w32(wbase_bsent + bsent_off, bytes_sent);
                sig_ctr = 0;
            }
            cn++;
            total++;
            progressed = 1;
            cyc_copy += rdcycle_() - tcp;
        }
        if (sig_ctr) {
            w32(wbase_bsent + bsent_off, bytes_sent); /* flush the tail (non-posted) */
            sig_ctr = 0;
        }
        w32(S_CONS, cn);
        loops++;
        w64(RES(0x00), total);
        w64(RES(0x08), loops);
        w64(G_SINGLE, (uint64_t)(uint32_t)(r32(S_PROD) - cn));           /* single-SPSC pages in flight */
        w64(G_D2H, (uint64_t)(uint32_t)(bytes_sent - r32(backed_addr))); /* D2H FIFO bytes in flight */
        {
            uint64_t now = rdcycle_();
            if (now >= traj_next) { /* time-gated snapshot of all 4 gauges -> trajectory ring */
                uint64_t slot = TRAJ_BASE + (traj_n % TRAJ_CAP) * TRAJ_STRIDE;
                w64(slot + 0, now);
                w32(slot + 8, (uint32_t)r64(G_L1(0)));
                w32(slot + 12, (uint32_t)r64(G_L1(1)));
                w32(slot + 16, (uint32_t)r64(G_MIRROR));
                w32(slot + 20, (uint32_t)(r32(S_PROD) - cn));
                w32(slot + 24, (uint32_t)(bytes_sent - r32(backed_addr)));
                w32(slot + 28, 0);
                traj_n++;
                w64(TRAJ_COUNT, traj_n);
                traj_next = now + TRAJ_PERIOD_CYC;
            }
        }
        if (!progressed) {
            if (r64(COLLECT_DONE) && cn == r32(S_PROD)) {
                break;
            }
        }
    }
relay_done:
    fence_();
    w32(S_CONS, cn);
    w64(RES(0x00), total);
    w64(RES(0x20), stalls);
    w64(RES(0xB0), rdcycle_() - t_start);
    w64(RES(0xB8), cyc_reserve);
    w64(RES(0xC0), cyc_copy);
    /* Copy the collect hart's stats into RES (the relay is the sole writer of this cache line, and its
     * RES writes ARE host-visible; the collect hart's own RES writes were not -- see COLLECT_STATS). */
    w64(RES(0xC8), r64(COLLECT_STATS + 0));   /* collect moved (markers) */
    w64(RES(0xD0), r64(COLLECT_STATS + 8));   /* collect loops */
    w64(RES(0xD8), r64(COLLECT_STATS + 16));  /* collect wall */
    w64(RES(0xE0), r64(COLLECT_STATS + 24));  /* collect empty-spin */
    w64(RES(0xE8), r64(COLLECT_STATS + 32));  /* collect copy */
    /* collect breakdown (relay-copied so it's host-visible): swait/swait_spins/scanned/productive
     * -> RES 0x28/0x48/0xF0/0xF8 (host reads at params+0x68/0x88/0x130/0x138). */
    w64(RES(0x28), r64(COLLECT_STATS + 40));  /* collect swait (blocked on relay) */
    w64(RES(0x48), r64(COLLECT_STATS + 48));  /* collect swait_spins */
    w64(RES(0xF0), r64(COLLECT_STATS + 56));  /* collect scanned (mirror-visits) */
    w64(RES(0xF8), r64(COLLECT_STATS + 64));  /* collect productive (visits with data) */
    w64(RES(0x18), DONE_MAGIC);               /* written LAST: host waits on it before reading results */
    helper_to_idle_fw();                      /* relay done -- return to the resident idle FW */
    return 0;                                 /* unreachable (helper_to_idle_fw is noreturn) */
}
