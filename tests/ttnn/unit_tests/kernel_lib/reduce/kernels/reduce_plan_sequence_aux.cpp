// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr std::uint32_t kernel_owned_arg = get_compile_time_arg_val(0);
    using Auxiliary = ttnn::kernel_lib::ReduceAuxiliaryArgs<1>;
    static_assert(kernel_owned_arg == 23, "The auxiliary args must preserve the kernel-owned prefix");
    constexpr auto auxiliary_cb = Auxiliary::num_tiles == 0 ? ttnn::kernel_lib::reduce_plan_args::no_cb_id : 1U;
    static_assert(Auxiliary::cb_id == auxiliary_cb, "The aggregate recipe must carry its optional auxiliary CB ID");
    using BoundAuxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<Auxiliary, auxiliary_cb>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<BoundAuxiliary>();
}
