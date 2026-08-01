// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/tensor/tensor_accessor.h"

// Number of get_noc_addr calls measured inside a single profiler zone. Must match
// GET_NOC_ADDR_BENCHMARK_ITERS in the post-processor
// (tests/tt_metal/tt_metal/perf_microbenchmark/tensor_accessor/accessor_postprocess.py),
// which divides the zone cost by this to report cycles/call.
#ifndef GET_NOC_ADDR_BENCHMARK_ITERS
#define GET_NOC_ADDR_BENCHMARK_ITERS 4096
#endif

void kernel_main() {
    constexpr uint32_t base_idx_cta = 0;
    constexpr uint32_t base_idx_crta = 0;

    auto args = TensorAccessorArgs<base_idx_cta, base_idx_crta>();
    auto tensor_accessor = TensorAccessor(args, 0);

#if INTERLEAVED_LAYOUT
    // Interleaved accessors have no dspec(); the host pushes the page count as the
    // compile-time arg right after the accessor args.
    const uint32_t num_pages = get_compile_time_arg_val(args.next_compile_time_args_offset());
#else
    // Sharded layout: page count is known from the distribution spec.
    const uint32_t num_pages = tensor_accessor.dspec().tensor_volume();
#endif

    // Measure get_noc_addr with ONE profiler zone around a large call loop and report
    // cycles/call, instead of one zone per call: a per-call DeviceZoneScopedN adds ~18
    // cycles of Tracy overhead, which is the same order of magnitude as the address
    // calculation itself and swamps the signal. The EMPTY zone captures the fixed zone
    // enter/exit overhead so the post-processor can subtract it:
    //   cycles/call = (ACCESSOR_zone - EMPTY_zone) / N
    constexpr uint32_t N = GET_NOC_ADDR_BENCHMARK_ITERS;

    // Baseline: cost of an empty zone (zone enter/exit overhead only).
    {
        DeviceZoneScopedN("EMPTY");
    }

    // Measured: N address calculations under a single zone. `volatile` keeps the compiler
    // from hoisting/eliminating the call.
    {
        DeviceZoneScopedN(ACCESSOR_CONFIG_NAME);
        for (uint32_t i = 0; i < N; ++i) {
            volatile auto _ = tensor_accessor.get_noc_addr(i % num_pages);
        }
    }
}
