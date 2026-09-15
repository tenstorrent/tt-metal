// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#if DST_ACCUM_MODE
#error "V-axis experiment is BF16 destination only"
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
#ifdef SDPA_V_TRANSPOSED
#include "pv_transpose.hpp"
#define mm_no_mop_init_short vtransposed_mm_init
#define mm_no_mop_reinit_short vtransposed_mm_reinit
#endif
// The existing API definition is already parsed. Rename only calls made
// inside the frozen streaming header, never the original API or our helper.
#ifdef SDPA_LOFI_EXP_GRID7
#include "../exp_grid7.hpp"
#define exp_packthread_tile_init lofi_grid7_exp_packthread_tile_init
#endif
#ifdef RESIDENT_MAIN
#include "experiments/sdpa-l2/single-core-resident-v1/main/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#else
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#endif
#ifdef SDPA_LOFI_EXP_GRID7
#undef exp_packthread_tile_init
#endif

#ifdef SDPA_V_TRANSPOSED
#undef mm_no_mop_init_short
#undef mm_no_mop_reinit_short
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
        DeviceZoneScopedN("SDPA_V_AXIS_FULLCHIP");
        sdpa_standard_v2<8, 16, 16 * k_chunks, 4, 4, scale,
                         2, 4, 2, 4, false, 0, 1, 2, 6, 3, 14, 4, 5, 16, 15>(
            jobs, k_chunks, 8, 9, 10, 11, 12, 13);
    }
}
