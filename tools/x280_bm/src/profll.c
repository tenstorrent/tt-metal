/*
 * profll.c - X280 per-lane LOSSLESS drainer (step-0 active FW).
 *
 * The "new active FW" for the never-linearize design (step-0 path a): each (core,risc) lane is kept
 * SEPARATE end to end. `nharts` harts each own a disjoint slice of cores; per (core,risc) L1 ring they
 * read [head,tail) and POSTED-write the raw words straight to that lane's OWN host pinned-memory slot
 * (slot (c*5+r) at host_base + (c*5+r)*slice_words*4), then advance the ring head so the producer
 * unblocks. reader==relay, no LIM SPSC, no linearization -- lane identity is the slot position, so the
 * drain is FORMAT-AGNOSTIC (the 8 B prof_packet.h words flow through untouched; only the producer and
 * the host interpret them). Lossless by back-pressure: producers block on full L1 rings; the drain
 * advances head only after the host write, and clamps the host write to slice_words (bounded capture).
 *
 * This is the DRAIN half of profcons.c grafted onto the boot-handoff skeleton of profzone.c so it is a
 * proper ACTIVE FW: it is JUMPed to at 0x08001000 by the resident idle FW (boot is baked in, untouched
 * -- see x280_boot.h / x280_driver.hpp), stamps RUNNING, and returns to the idle FW when done.
 *
 * LIM:
 *   PARAMS  @ 0x08011000 : +0x00 pcie_enc +0x08 host_base +0x10 prof_l1
 *                          +0x18 num_cores +0x20 slice_words +0x28 stop +0x30 nonce +0x38 nharts
 *   RESULTS @ 0x08011040 : per-hart slot h @ +h*0x40: +0x00 total_words +0x08 cycles
 *                          +0x10 max_outstanding +0x18 done(=DONE_MAGIC)
 *   HARTHB  @ 0x08011240 : per-hart boot heartbeat (h*8): 1=entered,2=setup,3=draining
 *   COORDS  @ 0x08011200 : num_cores x { u32 noc_x, u32 noc_y } (translated)
 *   HEADS   @ 0x08013000 : num_cores*5 x u32 (consumer head per ring, LIM-local mirror)
 */
#include <stdint.h>

#include "noc.h"
#include "x280_boot.h"

#define MBOX_PARAMS 0x08011000UL
#define MBOX_RESULTS 0x08011040UL
#define MBOX_COORDS 0x08011200UL
#define HEADS_BASE 0x08013000UL

#define P_PCIE_ENC (MBOX_PARAMS + 0x00)
#define P_HOST_BASE (MBOX_PARAMS + 0x08)
#define P_PROF_L1 (MBOX_PARAMS + 0x10)
#define P_NUM_CORES (MBOX_PARAMS + 0x18)
#define P_SLICE_WORDS (MBOX_PARAMS + 0x20)
#define P_STOP (MBOX_PARAMS + 0x28)
#define P_NONCE (MBOX_PARAMS + 0x30)
#define P_NHARTS (MBOX_PARAMS + 0x38)

#define RES_SLOT(h) (MBOX_RESULTS + (uint64_t)(h) * 0x40)
#define RES_TOTAL 0x00
#define RES_CYCLES 0x08
#define RES_MAXOUT 0x10
#define RES_DONE 0x18
#define DONE_MAGIC 0xC0570FFEE1ULL
#define FOOTER_MAGIC 0xC05D09F11E12345ULL
/* boot heartbeat @ 0x08011140 -- in the gap between the RES slots (end ~0x140) and COORDS (0x200).
 * MUST stay below MBOX_COORDS: at +0x200 it aliased coords[core 8..11] (0x...240/248/250/258). */
#define HARTHB(h) (MBOX_RESULTS + 0x100 + (uint64_t)(h) * 8)

#define NRISC 5
#define RING_CAP 512u    /* producer (worker) L1 ring depth, words */
#define HRING_WORDS 512u /* per-lane HOST ring depth, words (the acked D2H FIFO for one lane) */
#define WRITE_WIN_BASE 200u
/* Per-lane host-ring flow control (the acked FIFO). HSENT: X280-authoritative words written to each
 * lane's host ring (also published to host mem, non-posted, as a delivery barrier). HACKED: words the
 * host consumer has drained, written back by the host over UMD; the X280 blocks a lane's drain when its
 * host ring is full (hsent - hacked == HRING_WORDS) -> back-pressure to the worker producer. */
/* NB: 0x08016000 is X280_BOOT_HS_BASE (idle heartbeat / cmd / phase / hart-wake) -- do NOT put data
 * there; HSENT lands past it. */
#define HSENT_BASE 0x08018000UL  /* per-lane u32 (LIM): words X280 has drained to each host ring so far */
#define HACKED_BASE 0x08017000UL /* per-lane u32 (LIM): words host consumer has drained, host writes over UMD */
#define CTRL_HEAD(r) (r)
#define CTRL_TAIL(r) (5u + (r))

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

/* Read 4 independent 64 B flits (ILP 4) from the ring and posted-write them out. */
static inline void copy4_flits(uint64_t rp, uint64_t wp) {
    __asm__ volatile(
        "vsetivli zero, 8, e64, m1, ta, ma\n"
        "vle64.v v0, (%0)\n vle64.v v1, (%1)\n vle64.v v2, (%2)\n vle64.v v3, (%3)\n"
        "vse64.v v0, (%4)\n vse64.v v1, (%5)\n vse64.v v2, (%6)\n vse64.v v3, (%7)\n"
        :
        : "r"(rp), "r"(rp + 64), "r"(rp + 128), "r"(rp + 192), "r"(wp), "r"(wp + 64), "r"(wp + 128), "r"(wp + 192)
        : "memory", "v0", "v1", "v2", "v3");
}
static inline void copy1_flit(uint64_t rp, uint64_t wp) {
    __asm__ volatile("vsetivli zero, 8, e64, m1, ta, ma\n vle64.v v0, (%0)\n vse64.v v0, (%1)\n"
                     :
                     : "r"(rp), "r"(wp)
                     : "memory", "v0");
}
/* copy `n` words src->dst: ILP4 flits, then 1-flit, then scalar tail. src/dst are byte addresses. */
static inline void copy_words(uint64_t dst, uint64_t src, uint32_t n) {
    uint32_t nflits = n / 8, fi = 0;
    for (; fi + 4 <= nflits; fi += 4) {
        copy4_flits(src + (uint64_t)fi * 64, dst + (uint64_t)fi * 64);
    }
    for (; fi < nflits; fi++) {
        copy1_flit(src + (uint64_t)fi * 64, dst + (uint64_t)fi * 64);
    }
    for (uint32_t ww = nflits * 8; ww < n; ww++) {
        w32(dst + (uint64_t)ww * 4, r32(src + (uint64_t)ww * 4));
    }
}

/* ---- boot-handoff helpers (idle FW is baked in; we just re-enter it) ---- */
static inline __attribute__((noreturn)) void x280_jump(uint64_t entry) {
    /* fence.i: the L2CPU is never reset across handoffs, so its I-cache still holds the idle FW's bytes;
     * harmless here (idle code is unchanged) but kept for symmetry with profzone.c. */
    __asm__ volatile("fence ow, ow\n fence.i\n jr %0\n" : : "r"(entry) : "memory");
    __builtin_unreachable();
}
static inline __attribute__((noreturn)) void helper_to_idle_fw(void) { x280_jump(X280_IDLE_FW_LOAD_ADDR); }
static inline __attribute__((noreturn)) void return_to_idle_fw(void) {
    *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_RETURNED_TO_IDLE;
    __asm__ volatile("fence ow, ow");
    x280_jump(X280_IDLE_FW_LOAD_ADDR);
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        /* tell host + idle FW the JUMP landed and the active FW is running */
        *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_RUNNING_ACTIVE_FW;
    }
    w64(HARTHB(hartid), 1);
    fence_();

    uint64_t host_base = r64(P_HOST_BASE);
    uint64_t prof_l1 = r64(P_PROF_L1);
    uint64_t num_cores = r64(P_NUM_CORES);
    (void)r64(P_SLICE_WORDS); /* host ring depth is the compile-time HRING_WORDS now */
    uint64_t nharts = r64(P_NHARTS);
    uint64_t pcie_enc = r64(P_PCIE_ENC);
    uint64_t read_noc = r64(P_NONCE) & 1ull; /* 0x30 slot: read-window noc_selector (0=NoC0, 1=NoC1) */
    if (nharts == 0 || nharts > 3) {
        nharts = 2;
    }

    /* every hart clears its own result slot */
    volatile uint64_t* rl = (volatile uint64_t*)RES_SLOT(hartid);
    for (int i = 0; i < 8; i++) {
        rl[i] = 0;
    }
    fence_();

    if (hartid >= nharts) {
        w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
        fence_();
        helper_to_idle_fw(); /* unused hart: re-arm idle instead of wfi */
    }

    /* this hart's contiguous slice of cores */
    uint64_t q = (num_cores + nharts - 1) / nharts;
    uint64_t lo = hartid * q, hi = lo + q;
    if (hi > num_cores) {
        hi = num_cores;
    }
    if (lo > num_cores) {
        lo = num_cores;
    }

    volatile uint32_t* coords = (volatile uint32_t*)MBOX_COORDS;
    volatile uint32_t* heads = (volatile uint32_t*)HEADS_BASE;
    for (uint64_t c = lo; c < hi; c++) {
        for (uint32_t r = 0; r < NRISC; r++) {
            heads[c * NRISC + r] = 0;
        }
    }

    uint64_t pcie_x = pcie_enc & 0x3f;
    uint64_t pcie_y = (pcie_enc >> 6) & 0x3f;
    uint64_t ctrl_off = prof_l1 & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint64_t off_w = host_base & (NOC_2M_WINDOW_STRIDE - 1ULL);
    uint32_t win_p = WRITE_WIN_BASE + (uint32_t)hartid; /* one POSTED window (non-posted host writes hang here) */

    noc_tlb_2m_t wt;
    wt.data[0] = 0;
    wt.data[1] = 0;
    wt.data[2] = 0;
    wt.data[3] = 0;
    wt.addr = host_base >> 21;
    wt.x_end = (uint32_t)pcie_x;
    wt.y_end = (uint32_t)pcie_y;
    wt.x_start = (uint32_t)pcie_x;
    wt.y_start = (uint32_t)pcie_y;
    wt.noc_selector = 0;
    wt.posted = 1;
    (void)noc_configure_tlb_2m_ext(win_p, &wt, 0);

    /* one read window per core in this hart's slice (index = core index); read_noc selects NoC0/NoC1. */
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
    uint64_t wbase_p = NOC_2M_WINDOW_BASE + (uint64_t)win_p * NOC_2M_WINDOW_STRIDE + off_w;

    /* zero this hart's lanes' host-ring producer pointer (HACKED is zeroed by the host before boot). */
    for (uint64_t c = lo; c < hi; c++) {
        for (uint32_t r = 0; r < NRISC; r++) {
            w32(HSENT_BASE + (c * NRISC + r) * 4, 0);
        }
    }
    fence_();

    w64(HARTHB(hartid), 2);
    fence_();

    uint64_t total = 0, loops = 0, max_out = 0;
    w64(HARTHB(hartid), 3);
    fence_();
    uint64_t t0 = rdcycle();

    for (;;) {
        uint64_t pending = 0; /* any lane still has undrained worker-L1 data (blocked or not) */
        for (uint64_t c = lo; c < hi; c++) {
            uint64_t cbase = NOC_2M_WINDOW_BASE + c * NOC_2M_WINDOW_STRIDE + ctrl_off;
            uint64_t rbufs = cbase + 128;
            for (uint32_t r = 0; r < NRISC; r++) {
                uint64_t L = c * NRISC + r;
                uint32_t hsent = r32(HSENT_BASE + L * 4);
                uint32_t tail = r32(cbase + CTRL_TAIL(r) * 4);
                uint32_t head = heads[L];
                if (tail == head) {
                    continue;
                }
                pending = 1;
                if ((uint64_t)(tail - head) > max_out) {
                    max_out = tail - head;
                }
                uint64_t wl1 = rbufs + (uint64_t)r * 2048;      /* worker L1 ring for this risc */
                uint64_t hring = wbase_p + L * HRING_WORDS * 4; /* this lane's host ring base */
                uint32_t start_hsent = hsent;
                uint32_t h2 = head;
                while (h2 != tail) {
                    uint32_t wstart = h2 % RING_CAP;
                    uint32_t wrun = tail - h2;
                    if (wrun > RING_CAP - wstart) {
                        wrun = RING_CAP - wstart; /* worker-L1 contiguous slab */
                    }
                    uint32_t hacked = r32(HACKED_BASE + L * 4);       /* host has drained this many words (LIM) */
                    uint32_t hspace = HRING_WORDS - (hsent - hacked); /* free space in this lane's host ring */
                    if (hspace == 0) {
                        break; /* host ring full -> back-pressure: leave worker L1 unread (producer blocks) */
                    }
                    uint32_t run = wrun < hspace ? wrun : hspace;
                    uint64_t rp = wl1 + (uint64_t)wstart * 4;
                    uint32_t hoff = hsent % HRING_WORDS;
                    uint32_t to_end = HRING_WORDS - hoff;
                    if (run <= to_end) {
                        copy_words(hring + (uint64_t)hoff * 4, rp, run);
                    } else { /* host-ring wrap: split the copy */
                        copy_words(hring + (uint64_t)hoff * 4, rp, to_end);
                        copy_words(hring, rp + (uint64_t)to_end * 4, run - to_end);
                    }
                    h2 += run;
                    heads[L] = h2;
                    w32(cbase + CTRL_HEAD(r) * 4, h2); /* advance worker-L1 head -> producer unblocks */
                    hsent += run;
                    total += run;
                }
                if (hsent != start_hsent) {
                    fence_();
                    w32(HSENT_BASE + L * 4, hsent); /* authoritative LIM copy; host seqlock-reads it over UMD */
                }
            }
        }
        loops++;
        if (r64(P_STOP) && !pending) {
            break; /* stopped AND all worker-L1 drained (a full host ring keeps `pending` set) */
        }
    }
    uint64_t t1 = rdcycle();
    fence_();
    w64(RES_SLOT(hartid) + RES_TOTAL, total * 4ULL); /* bytes drained */
    w64(RES_SLOT(hartid) + RES_CYCLES, t1 - t0);
    w64(RES_SLOT(hartid) + RES_MAXOUT, max_out);
    fence_();
    w64(RES_SLOT(hartid) + RES_DONE, DONE_MAGIC);
    fence_();

    if (hartid == 0) {
        /* coordinator: wait for the other active harts to finish, then re-arm the idle FW last. */
        for (uint64_t h = 1; h < nharts; h++) {
            while (r64(RES_SLOT(h) + RES_DONE) != DONE_MAGIC) {
                cpu_pause();
            }
        }
        return_to_idle_fw();
    }
    helper_to_idle_fw();
    return 0; /* unreachable */
}
