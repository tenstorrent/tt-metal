// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// X280 SPSC prototype — STEP 1: the producer.
//
// Runs on BRISC (data-movement RISCV_0) of a single Tensix core. Every ~400 ns
// it publishes an 8-byte record (2x uint32) into a fixed L1 location:
//
//     word[1] = wall-clock timestamp (low 32b) at publish time
//     word[0] = seq  (monotonic counter, incremented each tick)  <-- written LAST
//
// word[0] (seq) is written last and acts as the "commit": a consumer polling
// seq will only ever see a fully-written record (timestamp already in place).
// This is the rate-limited single producer the SPSC flit-ring will later feed.
//
// Pacing uses the Tensix wall-clock debug register (ticks at AICLK, ~1.35 GHz on
// Blackhole). 400 ns * 1.35 GHz ~= 540 cycles. PACE_CYCLES is supplied by the
// host as a compile define so the cadence can be tuned without editing here.

#include <cstdint>

#ifndef BUF_ADDR
#define BUF_ADDR 0x80000  // L1 byte offset of the 8B record (NoC-visible)
#endif

#ifndef PACE_CYCLES
#define PACE_CYCLES 540  // wall-clock ticks between publishes (~400 ns @ 1.35 GHz)
#endif

// Blackhole wall-clock low register (RISCV_DEBUG_REGS_START_ADDR | 0x1F0).
// Reading the low half is a single cheap local register load.
static constexpr uint32_t WALL_CLOCK_L = 0xFFB121F0u;

static inline uint32_t wall_lo() { return *reinterpret_cast<volatile uint32_t*>(WALL_CLOCK_L); }

void kernel_main() {
    volatile uint32_t* rec = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(BUF_ADDR));
    rec[0] = 0;  // seq
    rec[1] = 0;  // timestamp

    uint32_t seq = 0;
    uint32_t deadline = wall_lo() + PACE_CYCLES;
    while (true) {
        // Spin until the next ~400 ns deadline. Signed compare tolerates the
        // 32-bit wrap of the wall clock.
        uint32_t now;
        do {
            now = wall_lo();
        } while (static_cast<int32_t>(now - deadline) < 0);

        // Publish: timestamp first, then seq (the commit word).
        rec[1] = now;
        rec[0] = ++seq;

        deadline += PACE_CYCLES;
    }
}
