// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of deepseek_v3_b1 ReduceToOneB1's compute (#58724 fourth review): the Op of
// models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp itself with TW_R2O_PR (the PR's per-tile add), else the
// PR head ec7714f90f8's copy of that header (reduce_to_one_b1_head.hpp, the API add with no define: main's per-face program),
// after the kernels' deepseek_compute_kernel_init, run TWIN_ITERS times as a ROOT3, ROOT2 or ROOT1 worker.
// Compile args: num_tiles, local_cb, received_cb, scratch_cb, device role (1 ROOT3, 2 ROOT2, 3 ROOT1), iterations.
#include <cstdint>
#include "../../../../models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#if defined(TW_R2O_PR)
#include "../../../../models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp"
#else
#include "../../../../models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1_head.hpp"
#endif

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t local_cb = get_compile_time_arg_val(1);
    constexpr uint32_t received_cb = get_compile_time_arg_val(2);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(3);
    constexpr uint32_t role = get_compile_time_arg_val(4);
    constexpr uint32_t twin_iters = get_compile_time_arg_val(5);
    using R = deepseek_b1_ops::ReduceToOneB1;
    using CT = R::ComputeCTArgs<role, num_tiles, local_cb, received_cb, scratch_cb, scratch_cb, 0>;

    deepseek_compute_kernel_init();
    R::Op<CT, true, true> op;
    for (uint32_t it = 0; it < twin_iters; ++it) {
        op(R::ComputeArgs{});
    }
}
