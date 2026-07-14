/*
 * profzone.c - X280 profiler drainer: 2-reader + 1-relay hart split (production).
 *
 * Decouples the Tensix-L1 ring READ from the D2H-socket WRITE onto separate harts so a slow
 * socket write never stalls the ring reads (the single-hart serial drainer topped out ~250k
 * markers/s and back-pressured the compute cores). Architecture (validated in profcons_split):
 *   - reader harts (0..nread-1): each owns a DISJOINT core subset. Per (core,risc) ring, does the
 *     LOSSLESS direct-read of [head,tail) (no skip -- blocking producer + fence-before-publish keep
 *     every slot this-lap and visible; System Port is uncached so reads are fresh), TRANSFORMS each
 *     2-word marker into a 64B WorkerZoneWire page, and stages it in a per-reader LIM SPSC ring.
 *     Advances the SPSC head only after staging (producer unblocks). LIM is cached + coherent across
 *     X280 harts, so the staging ring is a clean on-chip SPSC.
 *   - relay hart (last): batch-drains each reader's staging ring, copying each pre-built 64B page to
 *     the ONE D2H socket FIFO (reserve on bytes_acked, wrap write_ptr, publish bytes_sent). Sole FIFO
 *     writer => no contention. NOC1 write window keeps relay writes off the readers' NOC0 reads.
 * End-to-end lossless: producers block on full L1 rings; readers block on full staging; the relay
 * blocks on a full D2H FIFO. Nothing is dropped (a lap-guard clamp guards a blocking failure).
 *
 * SPSC L1 contract (profiler_msg_t @ prof_l1): control_vector[32] then buffer[r] @ +128 + r*2048;
 * head=ctrl[r], tail=ctrl[5+r] (monotonic WORD counts), storage idx = count%512.
 * Marker = 2 words: w0 = 0x80000000 | (timer_id<<12) | time_H, w1 = time_L. type = (timer_id>>16)&7.
 * WorkerZoneWire page (64 B, first 28 used): +0x00 header +0x04 core_x +0x08 core_y +0x0C risc
 *   +0x10 timer_id +0x14 time_hi +0x18 time_lo.
 *
 * LIM:
 *   PARAMS  @ 0x08011000 : +0x00 config_addr +0x08 pcie_x +0x10 pcie_y +0x18 prof_l1
 *                          +0x20 num_cores +0x28 stop +0x30 nread +0x38 nharts
 *   RESULTS @ 0x08011040 : relay writes +0x00 total_relayed +0x08 loops +0x18 done +0x20 stalls;
 *                          hart0 writes +0x30 heartbeat(0xB007); readers write +0x50+h*8 dropped.
 *   COORDS  @ 0x08011200 : num_cores x { u32 core_x, u32 core_y } (as relayed; host translates).
 *   STAGECTL@ 0x08018000 : per reader h: +h*32 PROD, +8 CONS, +16 RDONE (page counts).
 *   STAGE   @ 0x08020000 : reader h flit ring at +h*STAGE_STRIDE : NREC x 64 B pages.
 * Host pre-zeros STAGECTL + the SPSC head/tail before boot (deterministic, no init race).
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
 * before re-polling. 99.7% of passes are idle, so this cuts the ~2.6M wasted polls ~10x and frees
 * NoC/L1 for the producers+relay; a PRODUCTIVE pass skips it, so real bursts still poll at full rate. */
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

/* per-reader LIM staging SPSC (64 KiB/reader: NREC=512 x 64B pages = 32 KiB) */
#define STAGECTL 0x08018000UL
#define STAGE_BASE 0x08020000UL
#define STAGE_STRIDE 0x10000UL
#define NREC 512u
#define PROD(h) (STAGECTL + (uint64_t)(h) * 32 + 0)
#define CONS(h) (STAGECTL + (uint64_t)(h) * 32 + 8)
#define RDONE(h) (STAGECTL + (uint64_t)(h) * 32 + 16)
#define SPAGE(h, i) (STAGE_BASE + (uint64_t)(h) * STAGE_STRIDE + ((uint64_t)(i) % NREC) * PAGE)
/* per-reader ILP bulk-read scratch in LIM (RING_CAP words = 2 KiB; 4 KiB stride). Lives between
 * COORDS (~0x08011600) and STAGECTL (0x08018000); safe for nread <= 5. */
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
/* copy one 64 B page (LIM staging -> FIFO) as a single wide vector load+store */
static inline void page_copy(uint64_t src, uint64_t dst) {
    __asm__ volatile("vsetivli zero, 8, e64, m1, ta, ma\n vle64.v v0, (%0)\n vse64.v v0, (%1)\n"
                     :
                     : "r"(src), "r"(dst)
                     : "memory", "v0");
}
/* ILP bulk-read: pull nwords (even) from an uncached NoC System-Port window into a LIM buffer with
 * wide LMUL=8 vector loads so many read beats are in flight (proven ~1.8 GB/s vs ~530 MB/s for a
 * blocking per-word r32). The reshape then reads from LIM (local, fast) instead of the NoC. */
static inline void bulk_read_words(uint64_t dst, uint64_t src, uint32_t nwords) {
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

    if (hartid >= nharts) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }

    if (hartid < nread) {
        /* ================= READER: lossless SPSC read -> WorkerZone page -> LIM staging ========= */
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

        uint32_t prod = 0; /* monotonic staged-page count for this reader */
        uint64_t dropped = 0;
        uint64_t scratch = SCRATCH(hartid);
        uint64_t bulk_words = 0, segs = 0; /* diag: avg seg = bulk_words/segs => is the ILP burst wide? */
        uint64_t passes = 0, polls = 0;    /* count-based (per-iteration rdcycle TRAPS on X280 => contaminates) */
        uint64_t t_start = rdcycle_();     /* ONE rdcycle pair for the whole run: uncontaminated wall */
        for (;;) {
            uint64_t progressed = 0;
            passes++;
            for (uint64_t c = lo; c < hi; c++) {
                uint32_t cx = coords[c * 2 + 0], cy = coords[c * 2 + 1];
                uint64_t cbase = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off;
                uint64_t rbufs = cbase + 128;
                /* Poll ALL 5 RISCs' head/tail in ONE ILP burst: ctrl[0..15] covers head[0..4]=ctrl[0..4]
                 * and tail[0..4]=ctrl[5..9]. Was 10 scalar NoC reads/core/pass, which dominated the run
                 * (rdcycle split: ~1.2s of 2.5s polling). Copy to locals BEFORE the marker bulk-read
                 * below reuses `scratch`. */
                bulk_read_words(scratch, cbase, 16);
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
                    /* SAFETY lap guard (blocking keeps tail-head <= RING_CAP; clamp if it ever fails) */
                    if ((uint32_t)(tail - head) > RING_CAP) {
                        dropped += (uint32_t)(tail - head) - RING_CAP;
                        head = tail - RING_CAP;
                    }
                    uint64_t ring_base = rbufs + (uint64_t)r * 2048;
                    uint32_t h = head;
                    while (h != tail) {
                        /* EXPERIMENT (non-functional; host DROPS the data): flush the raw ring segment
                         * straight into LIM staging as a bulk vector copy -- NO reshape, NO per-marker
                         * loop, NO identity stamping. Raw 2-word markers pack 8 per 64B staging page; the
                         * relay ships those pages unchanged and the host discards them. Isolates whether
                         * the reader per-marker path is what stretches burst-drain latency. */
                        uint32_t hidx = h % RING_CAP;   /* word offset in the ring */
                        uint32_t seg = RING_CAP - hidx; /* words until the ring wraps */
                        uint32_t rem = (uint32_t)(tail - h);
                        if (seg > rem) {
                            seg = rem;
                        }
                        uint32_t npg = (seg + 15u) / 16u; /* 64B pages (8 markers each) to hold seg words */
                        /* one staging-room check per SEGMENT (was per marker); bail on shutdown */
                        while ((uint32_t)(prod + npg - r32(CONS(hartid))) > NREC) {
                            if (r64(P_STOP)) {
                                goto reader_done;
                            }
                        }
                        uint32_t pidx = prod % NREC; /* first staging page for this segment */
                        uint32_t pfit = NREC - pidx; /* pages until the staging ring wraps */
                        uint64_t sbase = STAGE_BASE + (uint64_t)hartid * STAGE_STRIDE;
                        if (npg <= pfit) {
                            bulk_read_words(sbase + (uint64_t)pidx * 64, ring_base + (uint64_t)hidx * 4, seg);
                        } else {
                            uint32_t w1 = pfit * 16u; /* words that fit before the staging wrap (even) */
                            bulk_read_words(sbase + (uint64_t)pidx * 64, ring_base + (uint64_t)hidx * 4, w1);
                            bulk_read_words(sbase, ring_base + (uint64_t)(hidx + w1) * 4, seg - w1);
                        }
                        bulk_words += seg;
                        segs++;
                        prod += npg;
                        h += seg;
                        w32(cbase + CTRL_HEAD(r) * 4, h); /* advance SPSC head (NoC write) -> producer unblocks */
                        fence_();                         /* staged pages visible before PROD advances */
                        w32(PROD(hartid), prod);
                    }
                }
            }
            if (!progressed) {
                if (r64(P_STOP)) {
                    break;
                }
                /* idle: back off so we don't blind-poll ~275 rings at full tilt (see POLL_BACKOFF_CYC) */
                uint64_t tbk = rdcycle_();
                while ((rdcycle_() - tbk) < POLL_BACKOFF_CYC) {
                    if (r64(P_STOP)) {
                        goto reader_done;
                    }
                }
            }
        }
    reader_done:
        fence_();
        w32(PROD(hartid), prod);
        w64(RDONE(hartid), 1);
        w64(RES(0x50 + hartid * 8), dropped);
        w64(RES(0x60 + hartid * 8), bulk_words); /* diag: ILP burst width = bulk_words/segs */
        w64(RES(0x70 + hartid * 8), segs);
        w64(RES(0x80 + hartid * 8), rdcycle_() - t_start); /* uncontaminated total reader wall (X280 ~1 GHz) */
        w64(RES(0x90 + hartid * 8), passes);               /* outer-loop passes */
        w64(RES(0xA0 + hartid * 8), polls);                /* core-polls (1 ctrl bulk-read each) */
        for (;;) {
            __asm__ volatile("wfi");
        }
    } else {
        /* ================= RELAY: LIM staging -> ONE D2H socket FIFO ============================ */
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
        /* Second window for the bytes_sent SIGNAL, NON-POSTED (posted=0): the write completes -- lands in
         * host memory -- before the relay proceeds, so the host's pages_available() observes it promptly.
         * POSTED bytes_sent writes (even batched) landed only in ~15ms bursts => the host under-drained and
         * the relay reserve-stalled (the ~250k/s cap). */
        wt.addr = bsent_addr >> 21;
        wt.posted = 0;
        (void)noc_configure_tlb_2m_ext(WRITE_WIN + 1u, &wt, 0);
        uint64_t wbase_bsent = NOC_2M_WINDOW_BASE + (uint64_t)(WRITE_WIN + 1u) * NOC_2M_WINDOW_STRIDE;
        fence_();

        uint32_t cons[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t sig_ctr = 0;
        uint64_t total = 0, loops = 0, stalls = 0;
        uint64_t cyc_reserve = 0, cyc_copy = 0; /* relay time-split: FIFO-reserve wait vs page copy+signal */
        uint64_t t_start = rdcycle_();          /* total relay wall => empty-spin = wall - reserve - copy */
        for (;;) {
            uint64_t progressed = 0, all_done = 1;
            for (uint64_t h = 0; h < nread; h++) {
                uint32_t pr = r32(PROD(h));
                uint32_t cn = cons[h];
                while (cn != pr) {
                    int stopped = 0;
                    uint64_t rs = 0;
                    uint64_t trs = rdcycle_();
                    for (;;) { /* reserve one page of FIFO space (bytes in flight = bytes_sent-acked) */
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
                        /* Do NOT drop staged pages at shutdown: the host keeps draining the FIFO during
                         * teardown, so keep waiting to flush losslessly. Only give up if the host is truly
                         * gone (FIFO stuck for a very long time) so we cannot wedge forever. */
                        if (r64(P_STOP) && ++rs > 50000000ull) {
                            stopped = 1;
                            break;
                        }
                    }
                    cyc_reserve += rdcycle_() - trs; /* time blocked on host FIFO room (bytes_sent-acked) */
                    if (stopped) {
                        goto relay_done;
                    }
                    uint64_t tcp = rdcycle_();
                    page_copy(SPAGE(h, cn), wbase + fifo_off + write_ptr);
                    write_ptr += PAGE;
                    if (write_ptr >= fifo_total) {
                        write_ptr -= fifo_total;
                    }
                    bytes_sent += PAGE; /* local (drives the reserve check) */
                    /* Signal bytes_sent every 256 pages via the NON-POSTED window so it LANDS promptly (host
                     * acks + frees FIFO room). Batched by 256 so the non-posted round-trips stay cheap
                     * (~631k pages / 256 ~= 2.5k signal writes). */
                    if (++sig_ctr >= 256u) {
                        w32(wbase_bsent + bsent_off, bytes_sent);
                        sig_ctr = 0;
                    }
                    cn++;
                    total++;
                    progressed = 1;
                    cyc_copy += rdcycle_() - tcp; /* page_copy to FIFO + write_ptr advance + bytes_sent signal */
                }
                if (cn != cons[h]) {
                    w32(wbase_bsent + bsent_off, bytes_sent); /* flush the tail (non-posted) */
                    sig_ctr = 0;
                }
                cons[h] = cn;
                w32(CONS(h), cn);
                if (!r64(RDONE(h))) {
                    all_done = 0;
                }
            }
            loops++;
            w64(RES(0x00), total);
            w64(RES(0x08), loops);
            if (all_done && !progressed) {
                break;
            }
        }
    relay_done:
        fence_();
        w64(RES(0x00), total);
        w64(RES(0x20), stalls);
        w64(RES(0xB0), rdcycle_() - t_start); /* relay wall (X280 ~1 GHz); empty-spin = wall-reserve-copy */
        w64(RES(0xB8), cyc_reserve);          /* time blocked waiting on host D2H FIFO room */
        w64(RES(0xC0), cyc_copy);             /* time in page_copy + bytes_sent signal */
        w64(RES(0x18), DONE_MAGIC);           /* written LAST: host waits on it before reading results */
        for (;;) {
            __asm__ volatile("wfi");
        }
    }
    return 0;
}
