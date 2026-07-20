/*
 * profstream.c - X280 LINEARIZED profiler pipeline: 2 readers + 1 relay + single host SPSC.
 *
 * The never-linearize per-lane design hit a multi-lane host-read/ack coherence wall (see
 * x280_lossless_ring_d2h_race memory). This reverts to the PROVEN single-stream shape and solves
 * identity with a STICKY-SRC header the reader injects at each source switch:
 *
 *   worker L1 SPSCs  --(reader harts, ILP reads)-->  per-reader LIM SPSC  --(relay hart, round-robin)-->
 *     single host ring (ONE sent/acked pair -> coherent, no per-lane fan-out)  --> host demuxes by sticky
 *
 * reader hart h (0..nread-1): owns a contiguous slice of cores. For each (core,risc) that has new data it
 * FIRST copies the precomputed 8 B STICKY-SRC packet for that (core,risc) (SRCLUT lookup), THEN the ring
 * words [head,tail); advances the worker head so the producer unblocks. All output goes into the reader's
 * own LIM SPSC. relay hart (last): round-robins the reader SPSCs and copies their words verbatim to the
 * single host ring, flow-controlled by the host's ack. The host reads one stream: a STICKY-SRC sets the
 * current (core,risc); every marker/meta after it binds to that lane until the next STICKY-SRC.
 *
 * Boot: idle-FW + JUMP handoff, baked in (see x280_boot.h / profll.c) -- untouched.
 *
 * LIM:
 *   PARAMS  @ 0x08011000 : +0x00 pcie_enc +0x08 host_base +0x10 prof_l1 +0x18 num_cores
 *                          +0x20 hring_words +0x28 stop +0x30 read_noc +0x38 nread
 *   RESULTS @ 0x08011040 : per-hart slot h @ +h*0x40: +0x00 total_words +0x08 cycles +0x18 done
 *   HARTHB  @ 0x08011140 : per-hart boot heartbeat (h*8)
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated)
 *   HEADS   @ 0x08013000 : num_cores*5 x u32 worker-ring head (LIM mirror)
 *   SRCLUT  @ 0x08014000 : num_cores*5 x 8 B precomputed STICKY-SRC packets (host fills before boot)
 *   STAGECTL@ 0x08018000 : per reader h @ +h*64: +0 prod(u64) +8 cons(u64)   (LIM SPSC pointers, words)
 *   HSENT   @ 0x08017000 : u32 words the relay has pushed to the host ring (host reads)
 *   HACKED  @ 0x08017040 : u32 words the host has drained (host writes) -- SINGLE pair, coherent
 *   STAGE   @ 0x08020000 : reader h ring @ +h*STAGE_STRIDE, STAGE_WORDS words
 */
#include <stdint.h>

#include "noc.h"
#include "x280_boot.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL
#define HEADS_BASE 0x08013000UL
#define SRCLUT_BASE 0x08014000UL
#define HSENT_BASE 0x08017000UL  /* per-hart u32; X280 writes, host reads (stride 0x40) */
#define HACKED_BASE 0x08017200UL /* per-hart u32; host writes, X280 reads (stride 0x40, own line) */
#define HSENT(h) (HSENT_BASE + (uint64_t)(h) * 0x40)
#define HACKED(h) (HACKED_BASE + (uint64_t)(h) * 0x40)
/* split (reader/relay) uses the single hart-0 pair */
#define HSENT_ADDR HSENT(0)
#define HACKED_ADDR HACKED(0)
#define STAGECTL 0x08018000UL
#define STAGE_BASE 0x08020000UL
#define STAGE_STRIDE 0x10000UL /* 64 KiB per reader */
#define STAGE_WORDS_NORMAL 4096u /* 16 KiB LIM SPSC (per-risc mode; >= 8 worker rings) */
#define STAGE_WORDS_BULK 16384u  /* 64 KiB LIM SPSC (bulkcore: holds several 2560-word cores; fills stride) */

#define P_PCIE_ENC (MBOX_PARAMS + 0x00)
#define P_HOST_BASE (MBOX_PARAMS + 0x08)
#define P_PROF_L1 (MBOX_PARAMS + 0x10)
#define P_NUM_CORES (MBOX_PARAMS + 0x18)
#define P_HRING_WORDS (MBOX_PARAMS + 0x20)
#define P_STOP (MBOX_PARAMS + 0x28)
#define P_NONCE (MBOX_PARAMS + 0x30)
#define P_NREAD (MBOX_PARAMS + 0x38)

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_TOTAL 0x00
#define RES_CYCLES 0x08
#define RES_DONE 0x18
#define DONE_MAGIC 0xC0570FFEE1ULL
#define HARTHB(h) (MBOX_RESULTS + 0x100 + (uint64_t)(h) * 8)

#define PROD(h) (STAGECTL + (uint64_t)(h) * 64 + 0)
#define CONS(h) (STAGECTL + (uint64_t)(h) * 64 + 8)

#define NRISC 5
#define RING_CAP 512u                /* worker L1 ring depth, words */
#define ADAPT_THRESH (4u * RING_CAP) /* adaptive: bulk-read a core once >= 4 rings' worth of data pending */
#define PP_BULK_CORE 5u              /* raw-bulk core frame type (MUST match prof_packet.h) */
#define WRITE_WIN_BASE 200u

static inline uint64_t rdcycle(void) {
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}
static inline uint32_t r32(uint64_t a) { return *(volatile uint32_t*)a; }
static inline void w32(uint64_t a, uint32_t v) { *(volatile uint32_t*)a = v; }
static inline uint64_t r64(uint64_t a) { return *(volatile uint64_t*)a; }
static inline void w64(uint64_t a, uint64_t v) { *(volatile uint64_t*)a = v; }
static inline void fence_(void) { __asm__ volatile("fence iorw, iorw"); }
static inline void cpu_pause(void) { __asm__ volatile("nop"); }

/* copy `n` 32-bit words src->dst via wide RVV loads. e32/m8 (8 vector regs) => one vle32.v moves up to
 * VLEN*8/32 words and streams many NoC read requests pipelined, amortizing the NoC round-trip latency
 * (the profile showed the scalar reader was NoC-latency-bound at ~249 cyc/word). vsetvli returns the real
 * VL, so this is correct for any VLEN and needs only 4 B alignment. */
static inline void copy_words(uint64_t dst, uint64_t src, uint32_t n) {
    while (n) {
        uint32_t vl;
        __asm__ volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));
        __asm__ volatile("vle32.v v0, (%0)" ::"r"(src) : "memory", "v0");
        __asm__ volatile("vse32.v v0, (%0)" ::"r"(dst) : "memory");
        src += (uint64_t)vl * 4;
        dst += (uint64_t)vl * 4;
        n -= vl;
    }
}

/* Read one core's 5 RISC tails (20 contiguous bytes, ctrl words 5..9) in ONE NoC transaction via a vector
 * load, vs 5 separate `r32` NoC reads. Doesn't speed the reader's wall (the poll overlaps the copy/visit
 * latency + the LSU already pipelines the scalar reads) but cuts poll NoC traffic 5x -- cheaper on the NoC,
 * which matters as more harts/traffic contend. vl=5 (NRISC); X280 VLEN=512 so m2 VLMAX=32 >= 5 -> exact. */
static inline void read_tails(uint64_t src, uint32_t* dst) {
    __asm__ volatile("vsetvli x0, %0, e32, m2, ta, ma" ::"r"((uint64_t)NRISC));
    __asm__ volatile("vle32.v v0, (%0)" ::"r"(src) : "memory", "v0");
    __asm__ volatile("vse32.v v0, (%0)" ::"r"(dst) : "memory");
}

/* ---- boot-handoff helpers (idle FW baked in; we just re-enter it) ---- */
static inline __attribute__((noreturn)) void x280_jump(uint64_t entry) {
    __asm__ volatile("fence ow, ow\n fence.i\n jr %0\n" : : "r"(entry) : "memory");
    __builtin_unreachable();
}
static inline __attribute__((noreturn)) void helper_to_idle_fw(void) { x280_jump(X280_IDLE_FW_LOAD_ADDR); }
static inline __attribute__((noreturn)) void return_to_idle_fw(void) {
    *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_RETURNED_TO_IDLE;
    __asm__ volatile("fence ow, ow");
    x280_jump(X280_IDLE_FW_LOAD_ADDR);
}

/* ============================== READER ============================== */
/* Drain this hart's slice of cores into its LIM SPSC, injecting a STICKY-SRC before each (core,risc)'s
 * data. Blocks on a full SPSC (the relay drains cons). Runs until P_STOP and no worker data remains. */
static void reader_run(
    uint64_t hartid,
    uint64_t num_cores,
    uint64_t prof_l1,
    uint64_t nread,
    uint64_t read_noc,
    uint64_t fullread,
    uint64_t bulkcore,
    uint64_t adaptive) {
    uint64_t q = (num_cores + nread - 1) / nread;
    uint64_t lo = hartid * q, hi = lo + q;
    if (hi > num_cores) {
        hi = num_cores;
    }
    if (lo > num_cores) {
        lo = num_cores;
    }
    uint64_t ctrl_off = prof_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);

    /* one read window per core in this slice (index = core index) */
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    for (uint64_t c = lo; c < hi; c++) {
        noc_tlb_2m_t rt;
        rt.data[0] = 0;
        rt.data[1] = 0;
        rt.data[2] = 0;
        rt.data[3] = 0;
        rt.addr = prof_l1 >> 21;
        rt.x_end = coords[c * 2 + 0];
        rt.y_end = coords[c * 2 + 1];
        rt.x_start = coords[c * 2 + 0];
        rt.y_start = coords[c * 2 + 1];
        rt.noc_selector = (uint32_t)read_noc;
        (void)noc_configure_tlb_2m_ext((uint32_t)c, &rt, 0);
    }
    fence_();

    volatile uint32_t* heads = (volatile uint32_t*)HEADS_BASE;
    for (uint64_t c = lo; c < hi; c++) {
        for (uint32_t r = 0; r < NRISC; r++) {
            heads[c * NRISC + r] = 0; /* LIM mirror -- MUST start at 0 or tail-head underflows to a huge run */
        }
    }
    uint64_t sbase = STAGE_BASE + hartid * STAGE_STRIDE; /* this reader's LIM SPSC ring */
    uint32_t stage_words = (bulkcore || adaptive) ? STAGE_WORDS_BULK : STAGE_WORDS_NORMAL; /* big SPSC if bulk */
    uint32_t swm = stage_words - 1u;                                                       /* power-of-2 mask */
    uint64_t nbulk = 0; /* PROFILE: # cores drained via bulk (vs per-risc) -- shows the adaptive switch */
    uint32_t prod = 0;  /* LOCAL word count; only WRITTEN to LIM for the relay, never re-read */
    uint64_t total = 0;
    w32(PROD(hartid), 0);
    fence_();

    uint64_t t_copy = 0, t_wait = 0; /* PROFILE: cycles in the copy (NoC read+LIM write) vs SPSC-full wait */
    uint64_t visits = 0, polls = 0;  /* PROFILE: drains (tail!=head) vs total tail reads -> avg run = words/visits */
    uint64_t t0 = rdcycle();
    for (;;) {
        uint64_t pending = 0;
        for (uint64_t c = lo; c < hi; c++) {
            uint64_t cbase = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off;
            uint64_t rbufs = cbase + 128;
            uint32_t tails[NRISC];
            read_tails(cbase + 5u * 4, tails); /* all 5 RISC tails in one NoC read */
            /* ADAPTIVE SWITCH: the tails are already in hand, so decide per-core whether to do one bulk read
             * (amortized, best when the core is mostly full) or per-risc drains (efficient at light load,
             * skips empty riscs). A single dynamic switch, not separate modes. Threshold = ADAPT_THRESH words
             * of pending data across the 5 riscs. `--bulkcore` forces bulk always; plain mode forces per-risc. */
            uint32_t do_bulk = bulkcore;
            if (adaptive) {
                uint32_t full = 0;
                for (uint32_t r = 0; r < NRISC; r++) {
                    full += (uint32_t)(tails[r] - heads[c * NRISC + r]);
                }
                do_bulk = full >= (uint32_t)ADAPT_THRESH;
            }
            if (do_bulk) {
                nbulk++;
                /* RAW-BULK: ONE streaming NoC read of the whole core (5 contiguous rings = 2560 words) --
                 * reclaims the single-read amortization the per-risc framing lost. Emit a BULK_CORE frame:
                 * [header 2w][NRISC meta words: head_mod|run][pad to even][2560 RAW words]. The host splits it
                 * per-risc using the meta (takes only `run` valid words per ring, ignores the over-read past
                 * tail). Lossless (host takes exactly the valid range). One fence + one PROD publish/core. */
                uint32_t rawn = NRISC * RING_CAP; /* 2560 -- whole core, raw */
                uint32_t total_run = 0;
                for (uint32_t r = 0; r < NRISC; r++) {
                    total_run += (uint32_t)(tails[r] - heads[c * NRISC + r]);
                }
                if (total_run == 0) {
                    polls += NRISC;
                    continue; /* nothing to drain (bulkcore forced on an empty core) */
                }
                uint32_t prefix = 2u + NRISC;
                if (prefix & 1u) {
                    prefix++; /* pad to even so the stream stays 2-word aligned */
                }
                uint32_t need = prefix + rawn;
                uint64_t tw = rdcycle();
                while ((uint32_t)(stage_words - (prod - r32(CONS(hartid)))) < need) {
                    cpu_pause();
                }
                t_wait += rdcycle() - tw;
                uint64_t tc = rdcycle();
                /* header */
                w32(sbase + (uint64_t)(prod & swm) * 4, (PP_BULK_CORE << 27) | (uint32_t)(c & 0x7FFFFFFu));
                w32(sbase + (uint64_t)((prod + 1) & swm) * 4, rawn);
                prod += 2;
                /* per-risc meta: head_mod (hi16) | run (lo16) */
                for (uint32_t r = 0; r < NRISC; r++) {
                    uint32_t head = heads[c * NRISC + r], tail = tails[r];
                    uint32_t run = tail - head;
                    w32(sbase + (uint64_t)(prod & swm) * 4, ((head % RING_CAP) << 16) | (run & 0xFFFFu));
                    prod += 1;
                }
                if ((2u + NRISC) & 1u) {
                    w32(sbase + (uint64_t)(prod & swm) * 4, 0); /* pad word */
                    prod += 1;
                }
                /* ONE streaming bulk read: 5 contiguous rings (2560 words) NoC -> SPSC (split at SPSC wrap) */
                uint64_t src = rbufs;
                uint32_t di = prod, leftw = rawn;
                while (leftw) {
                    uint32_t sslot = di & swm;
                    uint32_t chunk = leftw;
                    if (chunk > stage_words - sslot) {
                        chunk = stage_words - sslot;
                    }
                    copy_words(sbase + (uint64_t)sslot * 4, src, chunk);
                    src += (uint64_t)chunk * 4;
                    di += chunk;
                    leftw -= chunk;
                }
                prod += rawn;
                total += rawn;
                for (uint32_t r = 0; r < NRISC; r++) {
                    heads[c * NRISC + r] = tails[r];
                    w32(cbase + r * 4, tails[r]); /* advance heads -> producers unblock */
                }
                t_copy += rdcycle() - tc;
                fence_();                /* frame + raw visible before PROD advances */
                w32(PROD(hartid), prod); /* ONE publish for the whole core */
                pending = 1;
                visits++;
                polls += NRISC;
                continue;
            }
            for (uint32_t r = 0; r < NRISC; r++) {
                uint64_t L = c * NRISC + r;
                uint32_t tail = tails[r];
                uint32_t head = heads[L];
                polls++;
                uint32_t run;
                if (fullread) {
                    /* "all buffers full" bench: poll the tail (cost kept) but IGNORE it -- always drain a
                     * FULL buffer. Copy RING_CAP-2 words from head (over-reads stale past tail, harmless);
                     * head still advances to the REAL tail so producers stay consistent. Deterministic
                     * max-drain workload regardless of actual occupancy. */
                    run = RING_CAP - 2u;
                } else {
                    if (tail == head) {
                        continue;
                    }
                    run = tail - head; /* <= RING_CAP (worker flow control) */
                }
                pending = 1;
                visits++;
                uint32_t need = 2u + run;   /* sticky + data */
                /* wait for room in our LIM SPSC. We read the RELAY's cons fresh from LIM each spin; our own
                 * prod stays local (only written to LIM for the relay) -- the proven profcons_split pattern. */
                uint64_t tw = rdcycle();
                while ((uint32_t)(stage_words - (prod - r32(CONS(hartid)))) < need) {
                    cpu_pause();
                }
                t_wait += rdcycle() - tw;
                uint64_t tc = rdcycle();
                /* inject the precomputed STICKY-SRC (2 words) for this lane */
                uint64_t lut = SRCLUT_BASE + L * 8;
                w32(sbase + (uint64_t)(prod & swm) * 4, r32(lut));
                w32(sbase + (uint64_t)((prod + 1) & swm) * 4, r32(lut + 4));
                prod += 2;
                /* copy the worker ring words [head,tail) into the SPSC in contiguous chunks (split at BOTH
                 * the worker-ring and SPSC wraps), each chunk copied with ILP flits. */
                uint64_t wl1 = rbufs + (uint64_t)r * 2048;
                uint32_t si = head, di = prod, left = run;
                while (left) {
                    uint32_t wslot = si % RING_CAP;
                    uint32_t sslot = di & swm;
                    uint32_t chunk = left;
                    if (chunk > RING_CAP - wslot) {
                        chunk = RING_CAP - wslot;
                    }
                    if (chunk > stage_words - sslot) {
                        chunk = stage_words - sslot;
                    }
                    copy_words(sbase + (uint64_t)sslot * 4, wl1 + (uint64_t)wslot * 4, chunk);
                    si += chunk;
                    di += chunk;
                    left -= chunk;
                }
                prod += run;
                total += run;
                t_copy += rdcycle() - tc;
                heads[L] = tail;
                w32(cbase + r * 4, tail); /* advance worker head -> producer unblocks */
                fence_();                 /* ring data + sticky visible before PROD advances */
                w32(PROD(hartid), prod);  /* publish to the relay */
            }
        }
        if (r64(P_STOP) && (!pending || fullread)) { /* fullread: pending is always 1 -> stop on P_STOP alone */
            break;
        }
    }
    uint64_t t_total = rdcycle() - t0;
    w64(RES_SLOT(hartid) + RES_TOTAL, total * 4ULL);
    w64(RES_SLOT(hartid) + 0x08, t_copy);  /* RES_TCOPY */
    w64(RES_SLOT(hartid) + 0x10, t_total); /* RES_TTOTAL */
    w64(RES_SLOT(hartid) + 0x20, t_wait);  /* RES_TWAIT */
    w64(RES_SLOT(hartid) + 0x28, visits);  /* # drains */
    w64(RES_SLOT(hartid) + 0x30, polls);   /* # tail reads */
    w64(RES_SLOT(hartid) + 0x38, nbulk);   /* # cores drained via bulk (adaptive switch) */
    w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
    fence_();
}

/* ============================== RELAY ============================== */
/* Round-robin the reader SPSCs; copy their words verbatim to the single host ring, flow-controlled by the
 * host's ack (HACKED). Blocks on a full host ring. Runs until P_STOP and all reader SPSCs are drained. */
static void relay_run(
    uint64_t hartid,
    uint64_t host_base,
    uint64_t rlo,
    uint64_t rhi,
    uint64_t hring_words,
    uint64_t pcie_enc,
    uint64_t nohostfc,
    uint64_t bulkcore) {
    uint32_t stage_words = bulkcore ? STAGE_WORDS_BULK : STAGE_WORDS_NORMAL; /* MUST match the reader's SPSC size */
    uint32_t swm = stage_words - 1u;
    /* Drain readers [rlo,rhi) -> their OWN host rings (reader h -> ring h @ host_base + h*hring_bytes, own
     * HSENT/HACKED pair + posted window WRITE_WIN_BASE+h). With ONE relay, [rlo,rhi)=[0,nread) (round-robins
     * all readers). With a relay PER READER ([k,k+1)), the two chip halves are fully decoupled -> the single
     * relay stops being the bottleneck. Each ring carries one reader's self-framed stream (no cross-reader
     * interleave); a per-ring host consumer thread drains it. */
    /* Each ring's sysmem region is [hring_words*4 data][64 B trailer], the trailer holding the SENT pointer
     * IN HOST SYSMEM. The relay publishes SENT through the SAME posted window as the data (ordered after it by
     * the fence + PCIe posted-write ordering), so the host polls its OWN RAM (~ns) instead of reading the
     * pointer from device LIM (~18 us/poll -- the old 215 MB/s wall). HACKED stays in LIM (host->device write
     * is posted/fast). SENT slot @ hbase[h] + hring_words*4. */
    uint64_t hbase[4];
    for (uint64_t h = rlo; h < rhi; h++) {
        uint64_t mhb = host_base + h * ((uint64_t)hring_words * 4 + 64);
        uint32_t win_p = WRITE_WIN_BASE + (uint32_t)h;
        noc_tlb_2m_t wt;
        wt.data[0] = 0;
        wt.data[1] = 0;
        wt.data[2] = 0;
        wt.data[3] = 0;
        wt.addr = mhb >> 21;
        wt.x_end = (uint32_t)(pcie_enc & 0x3f);
        wt.y_end = (uint32_t)((pcie_enc >> 6) & 0x3f);
        wt.x_start = (uint32_t)(pcie_enc & 0x3f);
        wt.y_start = (uint32_t)((pcie_enc >> 6) & 0x3f);
        wt.noc_selector = 0;
        wt.posted = 1;
        (void)noc_configure_tlb_2m_ext(win_p, &wt, 0);
        hbase[h] = NOC_2M_WINDOW_BASE + (uint64_t)win_p * NOC_2M_WINDOW_STRIDE + (mhb & (NOC_2M_WINDOW_STRIDE - 1ULL));
    }
    fence_();

    uint32_t hsent[4] = {0, 0, 0, 0}; /* per-ring published word count */
    uint32_t cons[4] = {0, 0, 0, 0};  /* LOCAL per-reader cons; only WRITTEN to LIM for the reader */
    for (uint64_t h = rlo; h < rhi; h++) {
        w32(hbase[h] + (uint64_t)hring_words * 4, 0); /* SENT pointer in host sysmem (trailer) */
        w32(CONS(h), 0);
    }
    fence_();
    uint64_t total = 0;
    /* PROFILE: t_copy = cycles moving LIM->host; hostfull = passes we skipped on a full host ring (host too
     * slow); idle = full round-robin passes with NO reader data (readers too slow). */
    uint64_t t_copy = 0, hostfull = 0, idle = 0, breach = 0;
    uint64_t t0 = rdcycle();

    for (;;) {
        uint64_t pending = 0;
        for (uint64_t h = rlo; h < rhi; h++) {
            uint32_t prod = r32(PROD(h)); /* the reader's prod, fresh from LIM */
            uint32_t cn = cons[h];        /* our own cons stays local */
            if ((int32_t)(prod - cn) <= 0) {
                continue; /* nothing new (or a transient stale read) -- never underflow into a huge run */
            }
            pending = 1;
            uint64_t sbase = STAGE_BASE + h * STAGE_STRIDE;
            uint32_t avail = prod - cn;
            /* nohostfc (diagnostic): pretend the host ring is always fully drained -> the relay NEVER blocks
             * on the host (writes/overwrites at full rate). Isolates whether the reader is throttled by the
             * host consumer sink. LOSSY on purpose (host reads garbage / isn't run). */
            uint32_t hspace = nohostfc ? (uint32_t)hring_words : (uint32_t)hring_words - (hsent[h] - r32(HACKED(h)));
            if (hspace == 0) {
                hostfull++;
                continue; /* ring h full -> skip to the OTHER reader (its own ring may have space), come back.
                           * SAFE with per-reader rings: reader h's continuation returns to ring h, still
                           * contiguous -- no other reader's frame can land on ring h. (The framing bug only
                           * existed on a single SHARED ring; here each reader owns a ring.) */
            }
            uint32_t run = avail < hspace ? avail : hspace;
            uint64_t tc = rdcycle();
            /* copy reader h's SPSC words -> its OWN ring h, chunked (split at SPSC wrap and ring wrap) */
            uint32_t si = cn, di = hsent[h], leftw = run;
            while (leftw) {
                uint32_t sslot = si & swm;
                uint32_t hslot = di % (uint32_t)hring_words;
                uint32_t chunk = leftw;
                if (chunk > stage_words - sslot) {
                    chunk = stage_words - sslot;
                }
                if (chunk > (uint32_t)hring_words - hslot) {
                    chunk = (uint32_t)hring_words - hslot;
                }
                copy_words(hbase[h] + (uint64_t)hslot * 4, sbase + (uint64_t)sslot * 4, chunk);
                si += chunk;
                di += chunk;
                leftw -= chunk;
            }
            t_copy += rdcycle() - tc;
            cn += run;
            cons[h] = cn;
            hsent[h] += run;
            total += run;
            fence_();         /* ring payload lands (same posted window) before SENT is published */
            w32(CONS(h), cn); /* free the reader SPSC (LIM) */
            w32(hbase[h] + (uint64_t)hring_words * 4, hsent[h]); /* publish SENT to host sysmem (fast poll) */
            (void)breach;
        }
        if (!pending) {
            idle++;
        }
        if (r64(P_STOP) && !pending) {
            /* Only stop once every READER is DONE (RES_DONE) *and* its SPSC is empty -- a reader can leave
             * its SPSC momentarily empty while blocked mid-drain; breaking then would strand its tail. */
            uint64_t all_done = 1;
            for (uint64_t h = rlo; h < rhi; h++) {
                if (r64(RES_SLOT(h) + RES_DONE) != DONE_MAGIC || r32(PROD(h)) != cons[h]) {
                    all_done = 0;
                }
            }
            if (all_done) {
                break;
            }
        }
    }
    uint64_t t_total = rdcycle() - t0;
    w64(RES_SLOT(hartid) + RES_TOTAL, total * 4ULL);
    w64(RES_SLOT(hartid) + 0x08, t_copy);   /* RES_TCOPY */
    w64(RES_SLOT(hartid) + 0x10, t_total);  /* RES_TTOTAL */
    w64(RES_SLOT(hartid) + 0x20, hostfull); /* passes skipped: host ring full */
    w64(RES_SLOT(hartid) + 0x28, idle);     /* passes with no reader data */
    w64(RES_SLOT(hartid) + 0x30, breach);   /* about-to-overwrite-unacked events (stale-high HACKED) */
    w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
    fence_();
}

/* ============================== DIRECT DRAIN (1..N harts) ============================== */
/* Each drain hart reads its OWN slice of worker cores (NoC, uncached -> coherent) and writes its OWN host
 * ring DIRECTLY, injecting a STICKY-SRC before each source's data. No LIM SPSC, no cross-hart handoff ->
 * sidesteps the cross-hart LIM coherence that corrupts the split pipeline's SPSC under saturation. Each hart
 * is a fully independent coherent single-lane pair (own HSENT/HACKED, own ring, own heads region in the
 * direct-mode-unused STAGE space), so N harts add zero shared LIM state. Flow-controlled by host HACKED(h). */
static void drain_direct(
    uint64_t hartid,
    uint64_t ndrain,
    uint64_t num_cores,
    uint64_t prof_l1,
    uint64_t host_base,
    uint64_t hring_words,
    uint64_t pcie_enc,
    uint64_t read_noc,
    uint64_t wnoc) {
    uint64_t q = (num_cores + ndrain - 1) / ndrain; /* contiguous core slice for this hart */
    uint64_t lo = hartid * q, hi = lo + q;
    if (hi > num_cores) {
        hi = num_cores;
    }
    if (lo > num_cores) {
        lo = num_cores;
    }
    uint64_t my_host_base =
        host_base + (uint64_t)hartid * ((uint64_t)hring_words * 4 + 64); /* ring + 64 B SENT trailer */
    uint64_t ctrl_off = prof_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t off_w = my_host_base & (NOC_2M_WINDOW_STRIDE - 1ULL);
    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    /* one read window per core in this slice (index = global core index -> disjoint across harts) */
    for (uint64_t c = lo; c < hi; c++) {
        noc_tlb_2m_t rt;
        rt.data[0] = 0;
        rt.data[1] = 0;
        rt.data[2] = 0;
        rt.data[3] = 0;
        rt.addr = prof_l1 >> 21;
        rt.x_end = coords[c * 2 + 0];
        rt.y_end = coords[c * 2 + 1];
        rt.x_start = coords[c * 2 + 0];
        rt.y_start = coords[c * 2 + 1];
        rt.noc_selector = (uint32_t)read_noc;
        (void)noc_configure_tlb_2m_ext((uint32_t)c, &rt, 0);
    }
    /* one posted write window to THIS hart's host ring (index = WRITE_WIN_BASE+hartid -> disjoint) */
    uint32_t win_p = WRITE_WIN_BASE + (uint32_t)hartid;
    noc_tlb_2m_t wt;
    wt.data[0] = 0;
    wt.data[1] = 0;
    wt.data[2] = 0;
    wt.data[3] = 0;
    wt.addr = my_host_base >> 21;
    wt.x_end = (uint32_t)(pcie_enc & 0x3f);
    wt.y_end = (uint32_t)((pcie_enc >> 6) & 0x3f);
    wt.x_start = (uint32_t)(pcie_enc & 0x3f);
    wt.y_start = (uint32_t)((pcie_enc >> 6) & 0x3f);
    wt.noc_selector = (uint32_t)wnoc; /* route the posted PCIe write over NoC0 (0) or NoC1 (1) */
    wt.posted = 1;
    (void)noc_configure_tlb_2m_ext(win_p, &wt, 0);
    fence_();
    uint64_t hbase = NOC_2M_WINDOW_BASE + (uint64_t)win_p * NOC_2M_WINDOW_STRIDE + off_w;

    /* per-hart heads region in the direct-mode-unused STAGE space (64 KiB/hart -> disjoint cache lines,
     * no cross-hart LIM false sharing at slice boundaries) */
    volatile uint32_t* heads = (volatile uint32_t*)(STAGE_BASE + hartid * STAGE_STRIDE);
    for (uint64_t c = lo; c < hi; c++) {
        for (uint32_t r = 0; r < NRISC; r++) {
            heads[c * NRISC + r] = 0;
        }
    }
    uint32_t hsent = 0;
    w32(hbase + (uint64_t)hring_words * 4, 0); /* SENT pointer in host sysmem (trailer) */
    fence_();
    uint64_t total = 0, t_copy = 0, t_wait = 0;
    uint64_t t0 = rdcycle();

    for (;;) {
        uint64_t pending = 0;
        for (uint64_t c = lo; c < hi; c++) {
            uint64_t cbase = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off;
            uint64_t rbufs = cbase + 128;
            for (uint32_t r = 0; r < NRISC; r++) {
                uint64_t L = c * NRISC + r;
                uint32_t tail = r32(cbase + (5u + r) * 4);
                uint32_t head = heads[L];
                if (tail == head) {
                    continue;
                }
                pending = 1;
                uint32_t run = tail - head;
                uint32_t need = 2u + run;
                uint64_t tw = rdcycle();
                while ((uint32_t)((uint32_t)hring_words - (hsent - r32(HACKED(hartid)))) < need) {
                    cpu_pause();
                }
                t_wait += rdcycle() - tw;
                uint64_t tc = rdcycle();
                /* inject the STICKY-SRC (2 words) into the host ring, then the worker data */
                uint64_t lut = SRCLUT_BASE + L * 8;
                w32(hbase + (uint64_t)(hsent % (uint32_t)hring_words) * 4, r32(lut));
                w32(hbase + (uint64_t)((hsent + 1) % (uint32_t)hring_words) * 4, r32(lut + 4));
                hsent += 2;
                uint64_t wl1 = rbufs + (uint64_t)r * 2048;
                uint32_t si = head, di = hsent, leftw = run;
                while (leftw) {
                    uint32_t wslot = si % RING_CAP;
                    uint32_t hslot = di % (uint32_t)hring_words;
                    uint32_t chunk = leftw;
                    if (chunk > RING_CAP - wslot) {
                        chunk = RING_CAP - wslot;
                    }
                    if (chunk > (uint32_t)hring_words - hslot) {
                        chunk = (uint32_t)hring_words - hslot;
                    }
                    copy_words(hbase + (uint64_t)hslot * 4, wl1 + (uint64_t)wslot * 4, chunk);
                    si += chunk;
                    di += chunk;
                    leftw -= chunk;
                }
                hsent += run;
                total += run;
                heads[L] = tail;
                w32(cbase + r * 4, tail); /* advance worker head -> producer unblocks */
                fence_();                 /* ring payload lands (same posted window) before SENT is published */
                w32(hbase + (uint64_t)hring_words * 4, hsent); /* publish SENT to host sysmem (fast poll) */
                t_copy += rdcycle() - tc;
            }
        }
        if (r64(P_STOP) && !pending) {
            break;
        }
    }
    uint64_t t_total = rdcycle() - t0;
    w64(RES_SLOT(hartid) + RES_TOTAL, total * 4ULL);
    w64(RES_SLOT(hartid) + 0x08, t_copy);
    w64(RES_SLOT(hartid) + 0x10, t_total);
    w64(RES_SLOT(hartid) + 0x20, t_wait);
    w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
    fence_();
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_RUNNING_ACTIVE_FW;
    }
    w64(HARTHB(hartid), 1);
    fence_();

    uint64_t host_base = r64(P_HOST_BASE);
    uint64_t prof_l1 = r64(P_PROF_L1);
    uint64_t num_cores = r64(P_NUM_CORES);
    uint64_t hring_words = r64(P_HRING_WORDS);
    uint64_t pcie_enc = r64(P_PCIE_ENC);
    uint64_t read_noc = r64(P_NONCE) & 1ull;
    uint64_t direct = (r64(P_NONCE) >> 8) & 1ull;   /* NONCE bit 8: DIRECT drain (no reader/relay split) */
    uint64_t splitnoc = (r64(P_NONCE) >> 9) & 1ull; /* NONCE bit 9: each drain hart reads over NoC (hartid&1) */
    uint64_t wnoc = (r64(P_NONCE) >> 11) & 1ull;    /* NONCE bit 11: route the posted PCIe write over NoC1 */
    uint64_t fullread = (r64(P_NONCE) >> 13) & 1ull; /* NONCE bit 13: reader always drains a FULL buffer (bench) */
    uint64_t bulkcore = (r64(P_NONCE) >> 14) & 1ull; /* NONCE bit 14: one bulk NoC read per core (all 5 rings) */
    uint64_t dualrelay = (r64(P_NONCE) >> 15) & 1ull; /* NONCE bit 15: one relay hart PER READER (decouple halves) */
    uint64_t adaptive = (r64(P_NONCE) >> 16) & 1ull;  /* NONCE bit 16: per-core adaptive bulk-vs-per-risc switch */
    /* P_NREAD carries the drain-hart count in direct mode, the reader count in split mode */
    uint64_t nread_or_drain = r64(P_NREAD);
    uint64_t ndrain = 1, nread = 2;
    if (direct) {
        ndrain = nread_or_drain;
        if (ndrain == 0 || ndrain > 4) {
            ndrain = 1;
        }
    } else {
        nread = nread_or_drain;
        if (nread == 0 || nread > 3) {
            nread = 2;
        }
    }
    uint64_t nrelay = dualrelay ? nread : 1;              /* 1 relay for all, or 1 per reader */
    uint64_t nharts = direct ? ndrain : (nread + nrelay); /* direct: ndrain drainers; split: readers + relays */

    volatile uint64_t* rl = (volatile uint64_t*)RES_SLOT(hartid);
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    fence_();

    if (hartid >= nharts) {
        w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
        fence_();
        helper_to_idle_fw();
    }

    w64(HARTHB(hartid), 3);
    fence_();

    if (direct) {
        uint64_t eff_noc = splitnoc ? (hartid & 1ull) : read_noc; /* split reads across NoC0/NoC1 per hart */
        drain_direct(hartid, ndrain, num_cores, prof_l1, host_base, hring_words, pcie_enc, eff_noc, wnoc);
    } else if (hartid < nread) {
        reader_run(hartid, num_cores, prof_l1, nread, read_noc, fullread, bulkcore, adaptive);
    } else {
        uint64_t hri = hartid - nread; /* relay index 0..nrelay-1 */
        uint64_t rlo = (nrelay == 1) ? 0 : hri;
        uint64_t rhi = (nrelay == 1) ? nread : (hri + 1); /* one relay per reader when dualrelay */
        relay_run(
            hartid, host_base, rlo, rhi, hring_words, pcie_enc, (r64(P_NONCE) >> 12) & 1ull, bulkcore || adaptive);
    }

    if (hartid == 0) {
        for (uint64_t h = 1; h < nharts; h++) {
            while (r64(RES_SLOT(h) + RES_DONE) != DONE_MAGIC) {
                cpu_pause();
            }
        }
        return_to_idle_fw();
    }
    helper_to_idle_fw();
    return 0;
}
