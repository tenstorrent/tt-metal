// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "experimental/llk_math_matmul_custom_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_api.h"
#endif

// defines the default throttle level for no-mop matmul kernels (default 0)
#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif

namespace ckernel {

// clang-format off
/**
 * Short initialization for the no-MOP matmul block operation. Configures only the unpacker and math
 * engine, without touching hardware configuration or pack. Safe to call at any point mid-kernel.
 *
 * Return value: None
 *
 * | Argument  | Description                                                   | Type     | Valid Range                                      | Required |
 * |-----------|---------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | in0_cb_id | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                          | True     |
 * | in1_cb_id | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                          | True     |
 * | transpose | The transpose flag for performing transpose operation on B    | bool     | Must be true or false                            | False    |
 * | ct_dim    | The number of columns of the output matrix in tiles           | uint32_t | 1 to 2^32-1                                      | False    |
 * | rt_dim    | The number of rows of the output matrix in tiles              | uint32_t | 1 to 2^32-1                                      | False    |
 * | kt_dim    | The inner dim of the input matrices in tiles                  | uint32_t | 1 to 2^32-1                                      | False    |
 */
// clang-format on
ALWI void mm_no_mop_init_short(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const bool transpose = false,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init_no_mop<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
}

// clang-format off
/**
 * Performs a block-sized matrix multiplication C=A*B using direct replay buffer execution instead of
 * a MOP. Accumulates the result into DST (DST += C). The DST register buffer must be in acquired
 * state via acquire_dst call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile in block A from the first input CB                | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile in block B from the second input CB               | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
ALWI void matmul_block_no_mop(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const bool transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_no_mop<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
}

// clang-format off
/**
 * Reinitializes address modifiers for the no-MOP matmul operation without a full re-init.
 * Useful when resuming matmul after an interruption that may have modified address modifier registers.
 * Must be called from the math engine only (TRISC_MATH context).
 *
 * Return value: None
 */
// clang-format on
ALWI void mm_no_mop_configure_addrmod_reinit(const bool transpose = false) {
    MATH((llk_math_matmul_configure_addrmod_reinit<MATH_FIDELITY, MM_THROTTLE>(transpose)));
}

// clang-format off
/**
 * Lightweight no-MOP matmul reinit for steady-state loops where tile formats/dim assumptions
 * are unchanged. Reprograms unpack matmul setup and restores math addrmods without full init.
 *
 * Return value: None
 */
// clang-format on
ALWI void mm_no_mop_reinit_short(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const bool transpose = false,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_reinit_no_mop<MATH_FIDELITY, MM_THROTTLE>(transpose)));
}

// clang-format off
/**
 * Restores no-MOP matmul math-side state only (addrmods/counters), without unpack re-init
 * and without replay program reconfiguration. Use after ops that touch math addrmods while
 * matmul replay configuration remains valid.
 *
 * Return value: None
 */
// clang-format on
ALWI void mm_no_mop_reinit_addrmods_only(const bool transpose = false) {
    MATH((llk_math_matmul_reinit_no_mop<MATH_FIDELITY, MM_THROTTLE>(transpose)));
}

}  // namespace ckernel
