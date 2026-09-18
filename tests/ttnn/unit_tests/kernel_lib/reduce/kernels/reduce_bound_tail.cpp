// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include <type_traits>

void kernel_main() {
    using LogicalCall = ttnn::kernel_lib::ReduceCallArgs<2, RUNTIME_ARG_OFFSET>;
    // Exercise rebinding on both runtime alternatives, with no caller branch.
    using Call = ttnn::kernel_lib::BoundReduceCallArgs<LogicalCall, 3, 1, 16>;
    static_assert(Call::has_tail_variant);
    static_assert(Call::auxiliary_tile_count == 0);
    static_assert(Call::Tail::auxiliary_tile_count > 0);
    static_assert(Call::algorithm != Call::Tail::algorithm);
    static_assert(std::is_empty_v<Call>);
    static_assert(!std::is_constructible_v<Call, ttnn::kernel_lib::ReduceRuntimeShape>);
    compute_kernel_hw_startup(Call::input_cb_id, Call::output_cb_id);
    compute_kernel_lib::reduce<Call>();
}
