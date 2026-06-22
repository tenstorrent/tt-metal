// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// X280 SPSC prototype — STEP 2: the flit-ring producer.
//
// Runs on BRISC of one Tensix worker core. Treats a region of its own L1 as a
// single-producer/single-consumer ring of 64-byte cells (one NoC flit each).
// Both pointers live in this core's L1:
//
//   w (write index, monotonic u32) : written here (local store), read by the
//                                    X280 consumer over the NoC.
//   r (read index,  monotonic u32) : written by the X280 consumer over the NoC,
//                                    read here (local load) to check free space.
//
// The producer BLOCKS (spins) when the ring is full -- i.e. when advancing w
// would catch up to r (w - r >= RING_CELLS) -- so it never overwrites a cell the
// consumer has not yet drained. This is the backpressure path: a slow consumer
// throttles the producer 1:1 instead of losing records.
//
// Cell payload for index k (64B = 16x u32):
//   word[0]      = k   (the seq; == the absolute ring index)
//   word[1]      = wall-clock low at publish time
//   word[2..15]  = k   (integrity: the consumer checks word[15]==word[0]==k)
// word[0..15] are written first, then a RISC-V `fence`, then w is bumped: a
// consumer that has seen the new w is guaranteed to see a fully-written cell.
//
// Compile defines (supplied by the host launcher):
//   RING_BASE, RING_CELLS (power of 2), W_ADDR, R_ADDR, BLOCK_ADDR, PACE_CYCLES.
// PACE_CYCLES > 0 paces one publish per ~N wall-clock ticks (~400 ns @ 540);
// PACE_CYCLES == 0 produces as fast as possible (stress the ring / measure the
// consumer drain ceiling via backpressure).

#include <cstdint>

#ifndef RING_BASE
#define RING_BASE 0x80000
#endif
#ifndef RING_CELLS
#define RING_CELLS 32  // must be a power of 2
#endif
#ifndef W_ADDR
#define W_ADDR 0x80800
#endif
#ifndef R_ADDR
#define R_ADDR 0x80840
#endif
#ifndef BLOCK_ADDR
#define BLOCK_ADDR 0x80880
#endif
#ifndef PACE_CYCLES
#define PACE_CYCLES 540  // 0 = unpaced (produce as fast as possible)
#endif

static constexpr uint32_t WALL_CLOCK_L = 0xFFB121F0u;  // RISCV_DEBUG_REG_WALL_CLOCK_L
static inline uint32_t wall_lo() { return *reinterpret_cast<volatile uint32_t*>(WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t N = RING_CELLS;
    volatile uint32_t* ring = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(RING_BASE));
    volatile uint32_t* wptr = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(W_ADDR));
    volatile uint32_t* rptr = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(R_ADDR));
    volatile uint32_t* blkptr = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(BLOCK_ADDR));

    // Clean known state at launch. The consumer attaches afterwards and only
    // ever writes rptr from here on; the producer only reads it.
    *rptr = 0;
    *wptr = 0;
    *blkptr = 0;

    uint32_t w = 0;
    uint32_t blocked = 0;  // count of times the ring went full (backpressure events)

#if PACE_CYCLES > 0
    uint32_t deadline = wall_lo() + PACE_CYCLES;
#endif
    while (true) {
#if PACE_CYCLES > 0
        uint32_t now;
        do {
            now = wall_lo();
        } while (static_cast<int32_t>(now - deadline) < 0);
        deadline += PACE_CYCLES;
#endif

        // Backpressure: block while the ring is full. r is published by the X280
        // consumer over the NoC; a local load sees its latest value.
        if ((w - *rptr) >= N) {
            blocked++;
            *blkptr = blocked;
            while ((w - *rptr) >= N) {
                // spin -- cheap local load, no NoC traffic
            }
        }

        // Publish cell for index w (seq == w).
        const uint32_t slot = w & (N - 1);
        volatile uint32_t* cell = ring + slot * 16u;
        const uint32_t ts = wall_lo();
        for (uint32_t i = 0; i < 16u; i++) {
            cell[i] = w;
        }
        cell[1] = ts;

        // Ensure the whole cell is in L1 before the consumer can observe the new
        // w (which it uses to decide the cell is ready).
        asm volatile("fence" ::: "memory");

        w = w + 1;
        *wptr = w;
    }
}
