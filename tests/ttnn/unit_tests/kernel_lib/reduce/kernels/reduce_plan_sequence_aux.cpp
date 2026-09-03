// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr std::uint32_t kernel_owned_arg = get_compile_time_arg_val(0);
    using Auxiliary = ttnn::kernel_lib::ReduceAuxiliaryArgs<1>;
    static_assert(kernel_owned_arg == 23, "The auxiliary args must preserve the kernel-owned prefix");
    static_assert(Auxiliary::cb_id == 1, "The aggregate recipe must carry its auxiliary CB ID");
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();
}
