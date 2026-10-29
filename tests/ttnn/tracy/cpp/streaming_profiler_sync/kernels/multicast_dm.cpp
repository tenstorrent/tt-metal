// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two source cores take turns multicasting round numbers, one over NoC 0 on odd rounds and the other over NoC 1 on even
// rounds. Every core records an MC_RX zone when a round it didn't send arrives.

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "sync_workload.hpp"

constexpr uint32_t kGapCycles = 40000;
// A 16-bit Galois LFSR, x^16 + x^14 + x^13 + x^11 + 1, jitters each gap by up to 255 cycles.
constexpr uint32_t kLfsrSeed = 0xACE1u, kLfsrTaps = 0xB400u;

void kernel_main() {
    using namespace sync_workload;
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
        // A NoC write can send the previous value of a source the core has just stored to, so each round's value gets
        // its own slot.
        for (uint32_t round = 0; round <= rounds; round++) {
            values[round * kRoundValueStrideBytes / sizeof(uint32_t)] = round;
        }
        std::atomic_thread_fence(std::memory_order_release);
    }
    uint32_t lfsr = kLfsrSeed + static_cast<uint32_t>(role);
    uint32_t last_arrival = get_timestamp_32b();
    for (uint32_t round = 1; round <= rounds; round++) {
        const uint32_t source_noc = (round & 1u) ? 0u : 1u;
        if (source_noc == static_cast<uint32_t>(role)) {
            lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & kLfsrTaps);
            const uint32_t target = last_arrival + kGapCycles + (lfsr & 255u);
            while (static_cast<int32_t>(get_timestamp_32b() - target) < 0) {
            }
            const uint64_t dest_addr = get_noc_multicast_addr(x_start, y_start, x_end, y_end, flag_addr, source_noc);
            noc_semaphore_set_multicast(
                values_addr + kRoundValueStrideBytes * round, dest_addr, num_dests, /*linked=*/false, source_noc);
            continue;
        }
        if (!wait_for_round(flag, round)) {
            break;
        }
        {
            DeviceZoneScopedN("MC_RX");
        }
        last_arrival = get_timestamp_32b();
    }
    noc_async_write_barrier(0);
    noc_async_write_barrier(1);
}
