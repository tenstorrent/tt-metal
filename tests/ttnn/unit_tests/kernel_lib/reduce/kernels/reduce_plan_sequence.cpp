// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    constexpr std::uint32_t kernel_owned_arg = get_compile_time_arg_val(0);
    constexpr std::uint32_t reduce_args_offset = 1;
    constexpr std::uint32_t call_count = get_compile_time_arg_val(reduce_args_offset);
    static_assert(kernel_owned_arg == 17, "The reduce args must preserve the kernel-owned prefix");
    static_assert(call_count == 2, "This sanity kernel explicitly places two planned reduce calls");

    using First = ttnn::kernel_lib::ReduceCallArgs<
        reduce_args_offset + ttnn::kernel_lib::reduce_plan_args::call_count_word_count>;
    using Second = ttnn::kernel_lib::ReduceCallArgs<First::next_compile_time_args_offset()>;
    static_assert(First::path == ttnn::kernel_lib::ReducePath::Tiled);
    static_assert(Second::path == ttnn::kernel_lib::ReducePath::Tiled);
    static_assert(First::input_cb_id == Second::input_cb_id, "Both calls must reuse the same input CB");
    static_assert(First::has_accumulator && Second::has_accumulator);
    static_assert(First::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::Intermediate);
    static_assert(First::accumulation_index == 0);
    static_assert(Second::accumulation_mode == ttnn::kernel_lib::ReduceAccumulationMode::Final);
    static_assert(Second::accumulation_index == 1);
    static_assert(First::partial_mode == compute_kernel_lib::ReducePartialMode::None);
    static_assert(Second::partial_mode == compute_kernel_lib::ReducePartialMode::None);
    static_assert(First::auxiliary_tile_offset == 0);
    static_assert(Second::auxiliary_tile_offset == First::auxiliary_tile_offset);

    constexpr std::uint32_t startup_src_b = First::algorithm == compute_kernel_lib::ReduceAlgorithm::AccumulateViaAdd
                                                ? First::input_cb_id
                                                : First::auxiliary_cb_id;
    compute_kernel_hw_startup(First::input_cb_id, startup_src_b, First::output_cb_id);

    // Call 0: create the raw running value in the accumulator CB.
    compute_kernel_lib::reduce<First>();

    // The kernel owns sequencing. Arbitrary compute or synchronization may be placed here.

    // Call 1: reload the running value, finalize it, and write the real output.
    static_assert(Second::auxiliary_cb_id == First::auxiliary_cb_id);
    compute_kernel_lib::reduce<Second>();
}
