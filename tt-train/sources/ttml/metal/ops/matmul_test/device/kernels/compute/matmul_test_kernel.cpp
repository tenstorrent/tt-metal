// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/ops/common/compute_utils.hpp"

namespace NAMESPACE {

// CBs - for reading inputs
constexpr auto cb_input_a_idx = tt::CBIndex::c_0;       // BF16
constexpr auto cb_input_b_idx = tt::CBIndex::c_1;       // BF16
constexpr auto cb_input_a_fp32_idx = tt::CBIndex::c_2;  // FP32 (if needed)
constexpr auto cb_input_b_fp32_idx = tt::CBIndex::c_3;  // FP32 (if needed)
constexpr auto cb_output_idx = tt::CBIndex::c_4;

// Compile-time flags indicating whether to use FP32 versions
#ifdef USE_FP32_A
constexpr bool use_fp32_a = (USE_FP32_A == 1);
#else
constexpr bool use_fp32_a = false;
#endif

#ifdef USE_FP32_B
constexpr bool use_fp32_b = (USE_FP32_B == 1);
#else
constexpr bool use_fp32_b = false;
#endif

inline void MAIN {
    init_sfpu(cb_input_a_idx, cb_input_b_idx);
    binary_op_init_common(cb_input_a_idx, cb_input_b_idx, cb_output_idx);

    // Determine which CBs to use for the matmul
    constexpr uint32_t cb_a = use_fp32_a ? cb_input_a_fp32_idx : cb_input_a_idx;
    constexpr uint32_t cb_b = use_fp32_b ? cb_input_b_fp32_idx : cb_input_b_idx;
    mm_init(cb_a, cb_b, cb_output_idx, 0);

    // Constants for clarity
    constexpr uint32_t one_tile = 1U;
    constexpr uint32_t tile_a_idx = 0U;
    constexpr uint32_t tile_b_idx = 0U;
    constexpr uint32_t output_reg_idx = 1U;
    constexpr uint32_t conversion_reg_idx = 0U;

    // Wait for input tiles (always in BF16 CBs from reader)
    cb_wait_front(cb_input_a_idx, one_tile);
    cb_wait_front(cb_input_b_idx, one_tile);

    // Convert BF16 to FP32 if needed by copying through tile registers
    // The pack operation will automatically convert formats based on dest CB format
    if constexpr (use_fp32_a) {
        reconfig_data_format(cb_input_a_idx, cb_input_a_idx);
        copy_tile_init(cb_input_a_idx);
        tile_regs_acquire();
        copy_tile(cb_input_a_idx, tile_a_idx, conversion_reg_idx);
        tile_regs_commit();
        pack_and_push(conversion_reg_idx, cb_input_a_fp32_idx);
        cb_wait_front(cb_input_a_fp32_idx, one_tile);
    }

    if constexpr (use_fp32_b) {
        reconfig_data_format(cb_input_b_idx, cb_input_b_idx);
        copy_tile_init(cb_input_b_idx);
        tile_regs_acquire();
        copy_tile(cb_input_b_idx, tile_b_idx, conversion_reg_idx);
        tile_regs_commit();
        pack_and_push(conversion_reg_idx, cb_input_b_fp32_idx);
        cb_wait_front(cb_input_b_fp32_idx, one_tile);
    }

    // Initialize matmul with proper format configuration and zero accumulator
    mm_init_short(cb_a, cb_b, 0);

    // Acquire tile registers for matmul
    tile_regs_acquire();
    reconfig_data_format(cb_a, cb_b);

    // Perform matmul: C = A @ B (single tile)
    // This is where the bug manifests: mixed format operands produce garbage
    matmul_tiles(cb_a, cb_b, tile_a_idx, tile_b_idx, output_reg_idx, false);

    // Commit the matmul result
    tile_regs_commit();

    // Pack result to output CB and push
    pack_and_push(output_reg_idx, cb_output_idx);

    // Pop input CBs
    cb_pop_front(cb_input_a_idx, one_tile);
    cb_pop_front(cb_input_b_idx, one_tile);
    if constexpr (use_fp32_a) {
        cb_pop_front(cb_input_a_fp32_idx, one_tile);
    }
    if constexpr (use_fp32_b) {
        cb_pop_front(cb_input_b_fp32_idx, one_tile);
    }
}

}  // namespace NAMESPACE
