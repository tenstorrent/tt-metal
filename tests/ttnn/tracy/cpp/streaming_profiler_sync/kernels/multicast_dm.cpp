// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The multicast test's kernel: the source core multicasts round numbers over one NoC, and every receiver records an
// MC_RX zone when each round arrives.

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "workload_layout.hpp"

constexpr uint32_t kGapCycles = 40000;
// A 16-bit Galois LFSR, x^16 + x^14 + x^13 + x^11 + 1, jitters each gap by up to 255 cycles.
constexpr uint32_t kLfsrSeed = 0xACE1u, kLfsrTaps = 0xB400u;

inline uint32_t wall_lo() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    const auto role = static_cast<MulticastRole>(get_arg_val<uint32_t>(0));
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
    if (role != MulticastRole::Receiver) {
        // Written once and never modified, so the NIU cannot fetch a stale value.
        for (uint32_t round = 0; round <= rounds; round++) {
            values[round * kRoundValueStrideBytes / sizeof(uint32_t)] = round;
        }
        std::atomic_thread_fence(std::memory_order_release);
    }
    uint32_t lfsr = kLfsrSeed + static_cast<uint32_t>(role);
    uint32_t last_arrival = wall_lo();
    for (uint32_t round = 1; round <= rounds; round++) {
        const uint32_t source_noc = (round & 1u) ? 0u : 1u;
        if (source_noc == static_cast<uint32_t>(role)) {
            lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & kLfsrTaps);
            const uint32_t target = last_arrival + kGapCycles + (lfsr & 255u);
            while (static_cast<int32_t>(wall_lo() - target) < 0) {
            }
            const uint64_t dest_addr = get_noc_multicast_addr(x_start, y_start, x_end, y_end, flag_addr, source_noc);
            noc_semaphore_set_multicast(
                values_addr + kRoundValueStrideBytes * round, dest_addr, num_dests, false, source_noc);
            continue;
        }
        for (uint32_t polls = 0; flag[kRoundWord] < round; polls++) {
            invalidate_l1_cache();
            if (polls == kSpinLimit) {
                flag[kGaveUpRoundWord] = round;
                noc_async_write_barrier(0);
                noc_async_write_barrier(1);
                return;
            }
        }
        {
            DeviceZoneScopedN("MC_RX");
        }
        last_arrival = wall_lo();
    }
    noc_async_write_barrier(0);
    noc_async_write_barrier(1);
}
