// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#ifdef SDPA_LOFI_SAFE_RESCALE
#include "../safe_rescale.hpp"
#endif
#ifdef RESIDENT_MAIN
#include "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#elif defined(SDPA_STREAMING_ACCURACY)
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "../exp_refiner.hpp"
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#else
#include "experiments/sdpa-l2/hybrid-mixed-v1/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "../exp_refiner.hpp"
#include "compute_streaming.hpp"
#endif

void kernel_main() {
    constexpr uint32_t q_repeats = get_compile_time_arg_val(0);
    constexpr uint32_t k_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t scale = get_compile_time_arg_val(2);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    // Reader publishes Q only after all resident input slots are initialized.
    cb_wait_front(0, 16);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    {
        DeviceZoneScopedN("SDPA_RESIDENT");
        sdpa_standard_v2<
            4,
            16,
            16 * k_chunks,
            4,
            4,
            scale,
#ifdef SDPA_FP32_STREAMING
            1,
            4,
            1,
            4,
#else
            2,
            4,
            2,
            4,
#endif
            false,
            0,
            1,
            2,
            6,
            3,
            14,
            4,
            5,
            16,
            15>(q_repeats, k_chunks, 8, 9, 10, 11, 12, 13);
    }
}
