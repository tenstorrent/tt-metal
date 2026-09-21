// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "experiments/sdpa-l2/bfp4-lofi-v2/safe_rescale.hpp"
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "experiments/sdpa-l2/bfp4-lofi-v2/exp_refiner.hpp"
#include "experiments/sdpa-l2/bfp4-lofi-v2/fast_correction.hpp"
#ifdef SDPA_SPRINT_CANDIDATE_HEADER
#include SDPA_SPRINT_CANDIDATE_HEADER
#else
#include "experiments/sdpa-l2/compute-sprint-v1/lowp/combined_fence/compute_streaming.hpp"
#endif

void kernel_main() {
    constexpr uint32_t jobs = get_compile_time_arg_val(0);
    constexpr uint32_t k_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t scale = get_compile_time_arg_val(2);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    cb_wait_front(0, 32);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    {
        DeviceZoneScopedN("SDPA_SPRINT_COMP_V3");
        sdpa_standard_v2<8, 16, 16 * k_chunks, 4, 4, scale, 2, 4, 2, 4, false,
                         0, 1, 2, 6, 3, 14, 4, 5, 16, 15>(jobs, k_chunks, 8, 9, 10, 11, 12, 13);
    }
}

