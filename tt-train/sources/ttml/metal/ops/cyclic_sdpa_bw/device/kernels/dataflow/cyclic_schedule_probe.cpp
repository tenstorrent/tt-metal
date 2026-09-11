// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Evaluate the cyclic schedule on the core and write it to DRAM.
//
// This is the first device step of the relay port and it deliberately does
// nothing else: every core runs the full t = 0..T loop, computes the
// quantities the relay will need at each timestep, and writes one page. There
// is no core-to-core traffic and nothing waits on anything, so it cannot
// hang. What it does establish is everything underneath the protocol --
// region selection, the core-id-to-coordinate placement, per-core runtime
// arguments, the persistent loop, and that the schedule headers behave the
// same compiled for RISC-V as they do on the host.
//
// One page per logical core, kFields uint32 per timestep:
//   0 row i                     4 next consumer (0 = streak end, spill)
//   1 column j                  5 1 if t begins a later streak of row i
//   2 producer core             6 endpoint threshold, else 0
//   3 producer is internal      7 spill endpoint, else 0
//
// Runtime args: output base address, this core's 1-based logical id.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"

namespace {
constexpr uint32_t kFields = 8;
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_core = get_arg_val<uint32_t>(1);

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_scratch = tt::CBIndex::c_0;
    constexpr uint32_t kTimesteps = 2u * kCores + 1u;
    constexpr uint32_t kPageBytes = kFields * sizeof(uint32_t) * kTimesteps;

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores);

    const uint32_t l1_addr = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);

    uint32_t k = 0;
    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);
        const auto producer = sched.producer(my_core, t);
        const bool later = sched.is_later_streak_start(pair.i, t);

        out[k++] = pair.i;
        out[k++] = pair.j;
        out[k++] = producer.core;
        out[k++] = producer.internal ? 1u : 0u;
        out[k++] = sched.next_consumer(pair.i, t);
        out[k++] = later ? 1u : 0u;
        out[k++] = later ? sched.endpoint_threshold(pair.i, t) : 0u;
        out[k++] = later ? sched.spill_endpoint(pair.i, t) : 0u;
    }

    const auto out_accessor = TensorAccessor(out_args, out_addr, kPageBytes);
    noc_async_write_page(my_core - 1u, out_accessor, l1_addr);
    noc_async_write_barrier();
}
