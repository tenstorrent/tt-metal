// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr std::uint32_t kernel_owned_arg = get_compile_time_arg_val(0);
    constexpr std::uint32_t reduce_args_offset = 1;
    constexpr std::uint32_t call_count = get_compile_time_arg_val(reduce_args_offset);
    static_assert(kernel_owned_arg == 17, "The reduce args must preserve the kernel-owned prefix");
    static_assert(call_count == 2, "This sanity kernel explicitly prepares two planned reduce calls");

    using First = ttnn::kernel_lib::ReduceCallArgs<
        reduce_args_offset + ttnn::kernel_lib::reduce_plan_args::call_count_word_count>;
    using Second = ttnn::kernel_lib::ReduceCallArgs<First::next_compile_time_args_offset()>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<First>();
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Second>();
}
