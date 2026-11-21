// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Verify KV write (Laguna; verify_kv_write.py), compute: the LLK sequence of paged_cache's
// paged_fused_update_cache.cpp (the legacy kernel at 78608b96a50^, lines 36-63; the same calls as the Metal 2.0 port on
// main): untilize this row's input block once (init only), then per head untilize the cache block (cb_cache ->
// cb_untilized_cache) and, once the writer has patched the row(s) in, tilize it back (cb_untilized_cache2 -> cb_out,
// in the cache's format), so the cache bits come out of the same untilize / pack path as paged_fused_update_cache.
//
// Compile-time args: cb_in_k, cb_in_v, cb_cache, cb_untilized_cache, cb_untilized_cache2, cb_untilized_input, cb_out,
// Wt, num_heads. Runtime args: set (0 = K: input cb_in_k, 1 = V: input cb_in_v).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t in1_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in2_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(4);
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(5);
    constexpr uint32_t out_cb = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t num_heads = get_compile_time_arg_val(8);

    const bool is_input1 = get_arg_val<uint32_t>(0) == 0;
    const uint32_t in_cb = is_input1 ? in1_cb : in2_cb;

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

    // Untilize input (single block, init only - no uninit needed)
    if (!is_input1) {
        compute_kernel_lib::untilize<
            Wt,
            in2_cb,
            untilized_in_cb,
            compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    } else {
        compute_kernel_lib::untilize<
            Wt,
            in1_cb,
            untilized_in_cb,
            compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    }

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration from previous iteration
        compute_kernel_lib::untilize<Wt, cache_cb, untilized_cache_cb>(1);

        // Wait on writer to update block. Tilize with reconfiguration
        compute_kernel_lib::tilize<Wt, untilized_cache2_cb, out_cb>(1);
    }
}
