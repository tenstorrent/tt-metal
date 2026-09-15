// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#if !defined(ARCH_BLACKHOLE) || DST_ACCUM_MODE
#error "Final reciprocal experiment requires Blackhole BF16 destination"
#endif
#ifdef SDPA_LOFI_SAFE_RESCALE
#include "../safe_rescale.hpp"
#endif
#ifdef RESIDENT_MAIN
#include "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#elif defined(SDPA_STREAMING_ACCURACY)
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#ifdef SDPA_LOFI_FIX_CORRECTION
#include "../fast_correction.hpp"
#endif
#include "../exp_refiner.hpp"
#else
#error "Select MAIN or FAST BF16"
#endif

// Do not redefine APPROX, legacy defaults or any common-header entrypoint.
// In each frozen streaming header, the only Blackhole recip_tile calls are
// the final normalization's init<false>() and calculate<false>(0, C).
#ifdef SDPA_FINAL_RECIP8
#include "recip_override.hpp"
#define recip_tile_init sdpa_final_recip8_tile_init
#define recip_tile sdpa_final_recip8_tile
#endif
#ifdef SDPA_FINAL_SCALE_HIFI4
#include "final_scale.hpp"
#undef mul_bcast_cols_init
#undef mul_tiles_bcast_cols
#define mul_bcast_cols_init sdpa_final_scale_mul_bcast_cols_init
#define mul_tiles_bcast_cols sdpa_final_scale_mul_tiles_bcast_cols
#endif
#ifdef RESIDENT_MAIN
#include "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#else
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#endif
#ifdef SDPA_FINAL_RECIP8
#undef recip_tile_init
#undef recip_tile
#endif
#ifdef SDPA_FINAL_SCALE_HIFI4
#undef mul_bcast_cols_init
#undef mul_tiles_bcast_cols
#define mul_bcast_cols_init lofi_safe_mul_bcast_cols_init
#define mul_tiles_bcast_cols lofi_safe_mul_tiles_bcast_cols
#endif

void kernel_main() {
    constexpr uint32_t k_chunks = get_compile_time_arg_val(0);
    constexpr uint32_t scale = get_compile_time_arg_val(1);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(2);
    static_assert(q_tiles == 8);
    const uint32_t jobs = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    cb_wait_front(0, 32);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    {
        DeviceZoneScopedN("SDPA_FINAL_RECIP8_FULLCHIP");
        sdpa_standard_v2<8, 16, 16 * k_chunks, 4, 4, scale,
                         2, 4, 2, 4, false, 0, 1, 2, 6, 3, 14, 4, 5, 16, 15>(
            jobs, k_chunks, 8, 9, 10, 11, 12, 13);
    }
}
