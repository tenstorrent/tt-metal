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

#define NRISC 5
#define RING_CAP 512u /* producer L1 ring depth (words) */
/* Idle poll backoff: after a whole pass drains nothing, spin ~this many cycles (X280 ~1 GHz => ~ns)
 * before re-polling. Most passes are idle, so this cuts wasted polls and frees NoC/L1 for producers;
 * a PRODUCTIVE pass skips it, so real bursts still poll at full rate. */
#define POLL_BACKOFF_CYC 500000u
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

/* Emit `seg` raw marker-words (even) into the single SPSC page ring at page `pp`, one 64B page at a
 * time (handles page-ring wrap), zero-padding the tail of a partial last page so its unused marker
 * slots have the valid bit (w0 bit31) clear and the host skips them. Returns pages consumed. */
static inline uint32_t emit_to_single(uint64_t single_store, uint32_t pp, uint64_t src, uint32_t seg) {
    uint32_t npg = (seg + 15u) / 16u;
    uint32_t wleft = seg;
    for (uint32_t p = 0; p < npg; p++) {
        uint64_t dst = single_store + (uint64_t)((pp + p) % SINGLE_NREC) * PAGE;
        uint32_t wc = wleft >= 16u ? 16u : wleft; /* words this page (even: markers are 2 words) */
        if (wc) {
            bulk_copy_words(dst, src, wc);
            src += (uint64_t)wc * 4u;
        }
        for (uint32_t w = wc; w < 16u; w++) {
            w32(dst + (uint64_t)w * 4u, 0); /* zero-pad -> valid bit clear -> host skips */
        }
        wleft -= wc;
    }
    return npg;
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        w64(RES(0x30), 0xB007ULL); /* heartbeat: main() entered (hart 0) */
    }
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
        for (;;) {
            __asm__ volatile("wfi");
        }
    }
    /* LIM overflow guard: too many cores would run the mirrors + single ring past LIM end. */
    if (num_cores > MAX_CORES || single_store + (uint64_t)SINGLE_NREC * PAGE > LIM_END) {
        if (hartid == 0) {
            w64(RES(0x40), 0xBAD00000ULL | num_cores); /* signal misconfig; host sees no DONE_MAGIC */
        }
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

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

        uint64_t dropped = 0;
        uint64_t scratch = SCRATCH(hartid);
        uint64_t bulk_words = 0, segs = 0;
        uint64_t passes = 0, polls = 0;
        uint64_t t_start = rdcycle_();
        for (;;) {
            uint64_t progressed = 0;
            passes++;
            for (uint64_t c = lo; c < hi; c++) {
                uint64_t cbase = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off;
                uint64_t rbufs = cbase + 128;
                /* Poll all 5 RISCs' head/tail in ONE ILP burst: ctrl[0..4]=head, ctrl[5..9]=tail. */
                bulk_copy_words(scratch, cbase, 16);
                fence_();
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
                        while ((uint32_t)(mtail + seg - r32(MHEAD(gi))) > MIRROR_DEPTH) {
                            if (r64(P_STOP)) {
                                goto reader_done;
                            }
                        }
                        uint32_t midx = mtail % MIRROR_DEPTH;
                        uint32_t mfit = MIRROR_DEPTH - midx; /* words until mirror wraps */
                        uint64_t l1src = ring_base + (uint64_t)hidx * 4;
                        if (seg <= mfit) {
                            bulk_copy_words(MSTORE(gi) + (uint64_t)midx * 4, l1src, seg);
                        } else {
                            bulk_copy_words(MSTORE(gi) + (uint64_t)midx * 4, l1src, mfit);
                            bulk_copy_words(MSTORE(gi), l1src + (uint64_t)mfit * 4, seg - mfit);
                        }
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
        w64(READER_DONE(hartid), 1); /* tell the collect hart this reader is finished */
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    if (hartid == nread) {
        /* ============ COLLECT: round-robin mirrors -> single SPSC (raw, no reshape) ============== */
        /* per-mirror consumer head (collect owns MHEAD). Local cache avoids re-reading LIM. */
        static uint32_t chead[MAX_CORES * NRISC];
        for (uint64_t i = 0; i < nmirror; i++) {
            chead[i] = 0;
        }
        uint32_t single_prod = 0;
        uint64_t moved = 0, loops = 0; /* moved = markers */
        uint64_t cyc_copy = 0;         /* time in the mirror->single copy path */
        uint64_t t_start = rdcycle_();
        w64(COLLECT_STATS + 40, 0xC0FFEE01ULL); /* PROBE: "collect entered" sentinel (relay copies to RES 0x138) */
        for (;;) {
            uint64_t progressed = 0;
            loops++;
            for (uint64_t i = 0; i < nmirror; i++) {
                uint32_t mtail = r32(MTAIL(i));
                uint32_t mh = chead[i];
                while (mh != mtail) {
                    uint32_t midx = mh % MIRROR_DEPTH;
                    uint32_t seg = MIRROR_DEPTH - midx; /* contiguous mirror words until wrap */
                    uint32_t rem = mtail - mh;
                    if (seg > rem) {
                        seg = rem;
                    }
                    uint32_t npg = (seg + 15u) / 16u;
                    /* reserve npg pages in the single SPSC; block on the relay (consumer). */
                    while ((uint32_t)(single_prod + npg - r32(S_CONS)) > SINGLE_NREC) {
                        if (r64(P_STOP)) {
                            goto collect_done;
                        }
                    }
                    uint64_t tcp = rdcycle_();
                    single_prod += emit_to_single(single_store, single_prod, MSTORE(i) + (uint64_t)midx * 4, seg);
                    fence_();
                    w32(S_PROD, single_prod);
                    cyc_copy += rdcycle_() - tcp;
                    mh += seg;
                    w32(MHEAD(i), mh); /* free mirror -> reader unblocks */
                    chead[i] = mh;
                    moved += seg / 2;
                    progressed = 1;
                }
            }
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
        const uint64_t cwall = rdcycle_() - t_start;
        w64(COLLECT_STATS + 0, moved);
        w64(COLLECT_STATS + 8, loops);
        w64(COLLECT_STATS + 16, cwall);            /* collect wall */
        w64(COLLECT_STATS + 24, cwall - cyc_copy); /* empty-spin (round-robin over idle mirrors) */
        w64(COLLECT_STATS + 32, cyc_copy);         /* copy time */
        /* PROBE: collect writes loops DIRECTLY to a fresh RES line (0x130, unshared by relay/readers).
         * If the host reads this nonzero, collect's OWN RES writes ARE visible (=> the earlier 0 was a
         * shared-cache-line clobber, not collect failing to run). */
        w64(RES(0x130), loops);
        fence_(); /* stats visible to the relay before the done flag */
        w64(COLLECT_DONE, 1);
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    /* ================= RELAY: single SPSC -> ONE D2H socket FIFO (NOC1) ========================= */
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
    w64(RES(0x138), r64(COLLECT_STATS + 40)); /* PROBE: sentinel relay read from COLLECT_STATS (expect 0xC0FFEE01) */
    w64(RES(0x18), DONE_MAGIC);               /* written LAST: host waits on it before reading results */
    for (;;) {
        __asm__ volatile("wfi");
    }
    return 0;
}
