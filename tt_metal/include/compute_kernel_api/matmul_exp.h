// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_sfpu_exp_api.h"
#endif
#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif
namespace ckernel {

// ================== Fused MATMUL + EXP (PACK drives SFPU, MATH drives FPU) ==================

// clang-format off
/**
 * Fused initialization for matmul+exp block operation. Configures:
 *  - UNPACK: AB matmul unpacking
 *  - MATH:   MATMUL compute
 *  - PACK:   SFPU EXP configured (FAST APPROX defaulted)
 */
// clang-format on
ALWI void mm_exp_block_init_short(
    // matmul init configurations
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1,
    // exp init configurations
    bool approx = true,
    bool fast_and_approx = true,
    uint32_t scale = 0x3F800000) {
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    PACK((llk_math_eltwise_unary_sfpu_exponential_init<approx, fast_and_approx, scale>()));
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
}

// clang-format off
/**
 * Fused matmul+exp on a block tile (basic variant):
 *  - UNPACK issues AB unpack
 *  - MATH issues matmul on current DST
 *  - PACK applies EXP on the same DST (no overlap)
 */
// clang-format on
ALWI void matmul_exp_block(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, transpose, ct_dim, rt_dim, kt_dim)));
}
