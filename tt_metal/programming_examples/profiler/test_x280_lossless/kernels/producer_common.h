// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Step-0 throwaway profiler-marker PRODUCER (per-RISC). Emits the compact 8 B per-lane packet format
// straight into this RISC's L1 ring so the X280 can drain it and the host can prove the transport is
// lossless and the sticky/timer decode is correct. NOT kernel_profiler -- a deliberate stand-in that
// writes the ring protocol by hand so we can stress it with deterministic, checkable data.
//
// Wire format (MUST match tools/x280_bm/include/prof_packet.h -- inlined here to dodge kernel include
// paths; keep the two in sync):
//   word0: [31:27] type(5)  [26:0] low27       word1: [31:0] payload32
//   STICKY(6): low27=timer_hi(12 real bits)  payload32=prog_id
//   MARKER(0): low27=zone_srcloc (we stuff a per-lane SEQ here as the loss tag)  payload32=timer_low
//
// Ring layout (MUST match profcons.c / kernel_profiler): control @ PROF_L1, ctrl[r]=head (consumer),
// ctrl[5+r]=tail (this producer), data @ PROF_L1+128, this RISC's ring @ +r*2048, RING_CAP=512 words,
// storage index = count % 512. Lossless by back-pressure: block while the ring can't fit 2 more words.
//
// Host sets via -D defines: PROF_L1 (ring base), PROC_IDX (0..4 = this RISC), N_MARKERS (per lane),
//   TS_STEP (fake-timestamp increment; large enough that timer_hi ticks often to exercise stickies),
//   PROG_ID (op id carried by the sticky).
#pragma once
#include <cstdint>

#ifndef PROF_L1
#error "producer_common.h requires -DPROF_L1=<ring base L1 addr>"
#endif
#ifndef PROC_IDX
#define PROC_IDX 0
#endif
#ifndef N_MARKERS
#define N_MARKERS 4096u
#endif
#ifndef TS_STEP
#define TS_STEP 0x1000000ull /* 2^24 -> timer_hi (bit 32) ticks every 256 markers */
#endif
#ifndef PROG_ID
#define PROG_ID 0xA5A5A5A5u
#endif

#define PP_STICKY 6u
#define PP_ZONE_START 0u
#define RING_CAP 512u

static inline uint32_t pp_w0(uint32_t type, uint32_t low27) { return (type << 27) | (low27 & 0x7FFFFFFu); }

/* PROD_DELAY: busy-wait iterations between markers -> controls the production RATE. 0 = full-burst
 * worst case (emit as fast as the RISC can). Larger = slower producer (closer to real kernel spacing). */
#ifndef PROD_DELAY
#define PROD_DELAY 0u
#endif
/* Per-lane STALL STATS in L1 past the ring region (5 rings end at +0x2880): risc r @ +0x2C00 + r*16 =
 * [events(u32), spins_lo(u32), spins_hi(u32), markers(u32)]. `spins` = total block-on-full loop iterations
 * (proxy for stall time); `events` = markers that had to wait. The host reads these after the run. */
#define STALL_OFF 0x2C00u

static inline void producer_run() {
    const uint32_t r = (uint32_t)PROC_IDX;
    volatile uint32_t* ctrl = (volatile uint32_t*)(uintptr_t)(PROF_L1);
    volatile uint32_t* ring = (volatile uint32_t*)(uintptr_t)(PROF_L1 + 128u + r * 2048u);
    volatile uint32_t* stats = (volatile uint32_t*)(uintptr_t)(PROF_L1 + STALL_OFF + r * 16u);
    const uint32_t HEAD = r;      /* ctrl[r]   : consumer (X280) head */
    const uint32_t TAIL = 5u + r; /* ctrl[5+r] : this producer's tail */

    uint32_t tail = ctrl[TAIL]; /* host pre-zeroes control, so this is 0 on a fresh run */
    /* per-lane fake monotonic 44-bit device timestamp; small distinct base per lane. */
    uint64_t ts = (uint64_t)(r + 1u) << 8;
    uint32_t prev_hi = 0xFFFFFFFFu; /* != any real hi -> forces the kernel-start sticky */
    uint32_t stall_events = 0;
    uint64_t stall_spins = 0;

    for (uint32_t i = 0; i < (uint32_t)N_MARKERS; i++) {
        uint32_t hi = (uint32_t)((ts >> 32) & 0xFFFu); /* 12-bit device hi (WALL_CLOCK high) */
        if (hi != prev_hi) {
            /* STICKY: refresh timer_hi (+ prog_id) whenever the high word ticks. */
            {
                uint32_t sp = 0;
                while ((uint32_t)(tail - ctrl[HEAD]) > (RING_CAP - 2u)) { /* block: ring full */
                    sp++;
                }
                if (sp) {
                    stall_events++;
                    stall_spins += sp;
                }
            }
            ring[tail % RING_CAP] = pp_w0(PP_STICKY, hi);
            ring[(tail + 1u) % RING_CAP] = (uint32_t)PROG_ID;
            /* Do NOT publish TAIL until both words are actually in L1 SRAM: read them back (a load
             * retires only after the store commits), so the X280 sees the TAIL bump strictly AFTER the
             * data is globally visible. A plain fence only ORDERS the stores -- the X280 could still see
             * the bumped TAIL, read the slot over the NoC before the words land, get garbage, advance
             * head, and lose them forever (no content-based valid bit can catch that). */
            {
                volatile uint32_t rb0 = ring[tail % RING_CAP], rb1 = ring[(tail + 1u) % RING_CAP];
                (void)rb0;
                (void)rb1;
            }
            __asm__ volatile("fence" ::: "memory");
            for (volatile int d = 0; d < 512; d++) { /* let the L1 writes drain before publishing TAIL */
            }
            tail += 2u;
            ctrl[TAIL] = tail;
            prev_hi = hi;
        }
        /* MARKER: seq (=i) in zone_srcloc is the loss tag; timer_low is the fake ts low half. */
        {
            uint32_t sp = 0;
            while ((uint32_t)(tail - ctrl[HEAD]) > (RING_CAP - 2u)) { /* block: ring full */
                sp++;
            }
            if (sp) {
                stall_events++;
                stall_spins += sp;
            }
        }
        ring[tail % RING_CAP] = pp_w0(PP_ZONE_START, i & 0x7FFFFFFu);
        ring[(tail + 1u) % RING_CAP] = (uint32_t)(ts & 0xFFFFFFFFu);
        __asm__ volatile("" ::: "memory");
        tail += 2u;
        ctrl[TAIL] = tail;
        ts += (uint64_t)TS_STEP;
        for (volatile uint32_t d = 0; d < (uint32_t)PROD_DELAY; d++) { /* pace the producer */
        }
    }
    /* publish stall stats for the host to read after the run */
    stats[0] = stall_events;
    stats[1] = (uint32_t)(stall_spins & 0xFFFFFFFFu);
    stats[2] = (uint32_t)(stall_spins >> 32);
    stats[3] = (uint32_t)N_MARKERS;
}
