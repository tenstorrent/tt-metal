// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "experiments/sdpa-l2/bfp4-lofi-v2/safe_rescale.hpp"
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "experiments/sdpa-l2/bfp4-lofi-v2/exp_refiner.hpp"
#include "experiments/sdpa-l2/bfp4-lofi-v2/fast_correction.hpp"
#include "experiments/sdpa-l2/compute-sprint-v1/lowp/combined_fence/compute_streaming.hpp"

void kernel_main() {
    constexpr uint32_t scale = get_compile_time_arg_val(0);
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 0, 16);
    for (uint32_t t = 0; t < count; ++t) {
        cb_wait_front(0, 1);
        cb_reserve_back(16, 1);
        // Exact frozen streaming subtraction, accurate correction and BF16 pack.
        sub_exp_first_col_blocks<false, scale>(0, 0, 16, 0, 1);
        cb_push_back(16, 1);
        cb_pop_front(0, 1);
    }
}
