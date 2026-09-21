// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#ifndef SDPA_K_CHUNK_TILES
#define SDPA_K_CHUNK_TILES 16
#endif
#ifdef SDPA_LOFI_SAFE_RESCALE
#include "experiments/sdpa-l2/bfp4-lofi-v2/safe_rescale.hpp"
#endif
#ifdef RESIDENT_MAIN
#include "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#elif defined(SDPA_STREAMING_ACCURACY)
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "experiments/sdpa-l2/bfp4-lofi-v2/exp_refiner.hpp"
#ifdef SDPA_LOFI_FIX_CORRECTION
#include "experiments/sdpa-l2/bfp4-lofi-v2/fast_correction.hpp"
#endif
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#else
#include "experiments/sdpa-l2/hybrid-mixed-v1/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "experiments/sdpa-l2/bfp4-lofi-v2/exp_refiner.hpp"
#ifdef SDPA_SPRINT_CANONICAL
#include "experiments/sdpa-l2/bfp4-lofi-v2/streaming/compute_streaming.hpp"
#else
#ifdef SDPA_V2_BASELINE
#include "experiments/sdpa-l2/compute-sprint-v1/fp32/compute_streaming.hpp"
#else
#include "experiments/sdpa-l2/compute-sprint-v2/fp32/compute_streaming.hpp"
#endif
#endif
#endif
void kernel_main() {
    constexpr uint32_t jobs = get_compile_time_arg_val(0);
    constexpr uint32_t k_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t scale = get_compile_time_arg_val(2);
    constexpr uint32_t q_tiles = 8;
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    cb_wait_front(0, q_tiles * 4);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    {
        DeviceZoneScopedN("SDPA_FULLCHIP_LOFI");
        sdpa_standard_v2<
            q_tiles,
            SDPA_K_CHUNK_TILES,
            SDPA_K_CHUNK_TILES * k_chunks,
            4,
            4,
            scale,
#ifdef SDPA_FP32_STREAMING
#ifdef SDPA_V2_QK22
            2,
            2,
#else
            1,
            4,
#endif
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
            15>(jobs, k_chunks, 8, 9, 10, 11, 12, 13);
    }
}
