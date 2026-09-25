// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One Tensix core of the broadcast test. Two sources, at the two corners of the worker grid, take turns: odd rounds go
// out from source 0 on NoC 0, even rounds from source 1 on NoC 1, each a 4 B broadcast of the round number into every
// other Tensix core's flag. Every core, the sources included, spins on its flag and stamps MC_RX when a round arrives,
// with no branch between the two that the sources, which see only the other NoC's rounds, would predict differently
// from the receivers. A source issues its round a gap plus a pseudo-random pad after the other's arrived, so the
// pollers' phase against an arrival is spread over their loop. The round numbers are written once at start and never
// modified, so the NIU cannot fetch a stale value, and sit 16 B apart because an L1-to-L1 NoC write needs source and
// destination congruent mod 16. A spin gives up after kSpinLimit polls, leaving the round it was waiting for at
// flag_addr + 4 for the host.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t kSpinLimit = 1u << 24;
constexpr uint32_t kGapCycles = 40000;

inline uint32_t wall_lo() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);  // 0 or 1: the source of that NoC's rounds; 2: a receiver
    const uint32_t x_start = get_arg_val<uint32_t>(1);
    const uint32_t y_start = get_arg_val<uint32_t>(2);
    const uint32_t x_end = get_arg_val<uint32_t>(3);
    const uint32_t y_end = get_arg_val<uint32_t>(4);
    const uint32_t num_dests = get_arg_val<uint32_t>(5);
    const uint32_t flag_addr = get_arg_val<uint32_t>(6);
    const uint32_t values_addr = get_arg_val<uint32_t>(7);
    const uint32_t rounds = get_arg_val<uint32_t>(8);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    volatile tt_l1_ptr uint32_t* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(values_addr);
    noc_local_state_init(1);
    if (role < 2) {
        for (uint32_t r = 0; r <= rounds; r++) {
            values[4 * r] = r;
        }
        asm volatile("fence" ::: "memory");
    }
    uint32_t lfsr = 0xACE1u + role;
    uint32_t last = wall_lo();
    for (uint32_t r = 1; r <= rounds; r++) {
        const uint32_t src = (r & 1u) ? 0u : 1u;
        if (src == role) {
            lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
            const uint32_t target = last + kGapCycles + (lfsr & 255u);
            while (static_cast<int32_t>(wall_lo() - target) < 0) {
            }
            const uint64_t dst = get_noc_multicast_addr(x_start, y_start, x_end, y_end, flag_addr, src);
            noc_semaphore_set_multicast(values_addr + 16 * r, dst, num_dests, false, src);
            continue;
        }
        for (uint32_t polls = 0; *flag < r; polls++) {
            invalidate_l1_cache();
            if (polls == kSpinLimit) {
                flag[1] = r;
                noc_async_write_barrier(0);
                noc_async_write_barrier(1);
                return;
            }
        }
        {
            DeviceZoneScopedN("MC_RX");
        }
        last = wall_lo();
    }
    noc_async_write_barrier(0);
    noc_async_write_barrier(1);
}
