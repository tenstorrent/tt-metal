// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif
#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif
namespace ckernel {

// clang-format off
/**
 * Initializes the matmul operation for subsequent tile operations. Must be called before matmul_tile.
 * This function is safe to call multiple times in fused kernels and only configures matmul-specific
 * hardware settings that differ from the generic compute_kernel_hw_startup(). Common MMIO configurations
 * like PACK settings are handled by compute_kernel_hw_startup().
 *
 * Return value: None
 *
 * | Param Type | Name      | Description                                                 | Type     | Valid Range                                      | Required |
 * |------------|-----------|-------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | Function   | in0_cb_id | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                          | True     |
 * | Function   | in1_cb_id | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                          | True     |
 * | Function   | transpose | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set| False    |
 */
// clang-format on
ALWI void matmul_init(uint32_t in0_cb_id, uint32_t in1_cb_id, const uint32_t transpose = 0) {
    // CRITICAL: Only matmul-specific hardware configs that differ from generic startup
    UNPACK((llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));

    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
    // CRITICAL: Matmul-specific math hw config (no template params vs generic with <false, false>)
    MATH((llk_math_hw_configure_disaggregated(in0_cb_id, in1_cb_id)));
}

// clang-format off
/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles in two
 * specified input CBs and writes the result to DEST. The DEST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                                                               | Type     | Valid Range                                      | Required |
 * |------------|------------------|---------------------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | Function   | in0_cb_id        | The identifier of the first input circular buffer (CB)                   | uint32_t | 0 to 31                                          | True     |
 * | Function   | in1_cb_id        | The identifier of the second input circular buffer (CB)                  | uint32_t | 0 to 31                                          | True     |
 * | Function   | in0_tile_index   | The index of the tile A from the first input CB                          | uint32_t | Must be less than the size of the CB            | True     |
 * | Function   | in1_tile_index   | The index of the tile B from the second input CB                         | uint32_t | Must be less than the size of the CB            | True     |
 * | Function   | dest_tile_index  | The index of the tile in DEST REG to which the result C will be written  | uint32_t | Must be less than the acquired size of DEST REG | True     |
 * | Function   | transpose        | The transpose flag for performing transpose operation on B               | uint32_t | Any positive value will indicate transpose is set| False    |
 */
// clang-format on
ALWI void matmul_tile(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t dest_tile_index,
    const uint32_t transpose = 0) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index)));
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(dest_tile_index, transpose)));
}

// clang-format off
/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles
 * located in SRCA and SRCB and writes the result to DEST. The DEST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                                                               | Type     | Valid Range                                      | Required |
 * |------------|------------------|---------------------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | Template   | num_faces        | Number of faces to process                                                | uint32_t | 1 to 4                                           | False    |
 * | Function   | dest_tile_index  | The index of the tile in DEST REG to which the result C will be written  | uint32_t | Must be less than the acquired size of DEST REG | True     |
 */
// clang-format on
template <uint32_t num_faces = 4>
ALWI void matmul_tile_math(uint32_t dest_tile_index) {
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE, num_faces>(dest_tile_index)));
}

// clang-format off
/**
 * Initializes the matmul operation with data format reconfiguration for srcA.
 * This function should be used when switching from another operation that used
 * a different data format for srcA. Safe to call multiple times in fused kernels.
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                                 | Type     | Valid Range                                      | Required |
 * |------------|--------------|-------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | Function   | in0_cb_id    | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                          | True     |
 * | Function   | in1_cb_id    | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                          | True     |
 * | Function   | old_srca_cb  | The identifier of the old srcA circular buffer (CB)        | uint32_t | 0 to 31                                          | True     |
 * | Function   | transpose    | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set| False    |
 */
// clang-format on
ALWI void matmul_init_reconfig_data_format(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t old_srca_cb, const uint32_t transpose = 0) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_srca_cb, in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_srca_cb, in1_cb_id)));
    matmul_init(in0_cb_id, in1_cb_id, transpose);
}

// clang-format off
/**
 * Initializes the matmul block operation for subsequent block operations. Must be called before matmul_block.
 * This function is safe to call multiple times in fused kernels and only configures matmul-specific
 * hardware settings that differ from the generic compute_kernel_hw_startup(). Common MMIO configurations
 * like PACK settings are handled by compute_kernel_hw_startup().
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                                 | Type     | Valid Range                                         | Required |
 * |------------|----------------|-------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | Function   | in0_cb_id      | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                             | True     |
 * | Function   | in1_cb_id      | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                             | True     |
 * | Function   | transpose      | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set   | False    |
 * | Function   | block_ct_dim   | The number of columns of the output matrix in tiles        | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode| False    |
 * | Function   | block_rt_dim   | The number of rows of the output matrix in tiles           | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode| False    |
 * | Function   | block_kt_dim   | The inner dim of the input matrices in tiles               | uint32_t | 1 to 2^32-1                                         | False    |
 */
// clang-format on
ALWI void matmul_block_init(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t block_ct_dim = 1,
    uint32_t block_rt_dim = 1,
    uint32_t block_kt_dim = 1) {
    // CRITICAL: Only matmul-specific hardware configs that differ from generic startup
    UNPACK((llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, block_ct_dim, block_rt_dim, block_kt_dim)));

    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(
        in0_cb_id, in1_cb_id, transpose, block_ct_dim, block_rt_dim, block_kt_dim)));
    // CRITICAL: Matmul-specific math hw config (no template params vs generic with <false, false>)
    MATH((llk_math_hw_configure_disaggregated(in0_cb_id, in1_cb_id)));
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DEST. The DEST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                                                               | Type     | Valid Range                                      | Required |
 * |------------|------------------|---------------------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | Function   | in0_cb_id        | The identifier of the first input circular buffer (CB)                   | uint32_t | 0 to 31                                          | True     |
 * | Function   | in1_cb_id        | The identifier of the second input circular buffer (CB)                  | uint32_t | 0 to 31                                          | True     |
 * | Function   | in0_tile_index   | The index of the tile in block A from the first input CB                 | uint32_t | Must be less than the size of the CB            | True     |
 * | Function   | in1_tile_index   | The index of the tile in block B from the second input CB                | uint32_t | Must be less than the size of the CB            | True     |
 * | Function   | dest_tile_index  | The index of the tile in DEST REG to which the result C will be written  | uint32_t | Must be less than the acquired size of DEST REG | True     |
 * | Function   | transpose        | The transpose flag for performing transpose operation on tiles in B       | uint32_t | Any positive value will indicate transpose is set| False    |
 * | Function   | block_ct_dim     | The column dimension for the output block                                 | uint32_t | Must be equal to block B column dimension        | True     |
 * | Function   | block_rt_dim     | The row dimension for the output block                                    | uint32_t | Must be equal to block A row dimension           | True     |
 * | Function   | block_kt_dim     | The inner dimension                                                       | uint32_t | Must be equal to block A column dimension        | True     |
 */
// clang-format on
ALWI void matmul_block(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t dest_tile_index,
    const uint32_t transpose,
    uint32_t block_ct_dim,
    uint32_t block_rt_dim,
    uint32_t block_kt_dim) {
    UNPACK((llk_unpack_AB_matmul(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, block_ct_dim, block_rt_dim, block_kt_dim)));
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(
        dest_tile_index, transpose, block_ct_dim, block_rt_dim, block_kt_dim)));
}

// clang-format off
/**
 * Initializes the matmul block operation with data format reconfiguration for srcA.
 * This function should be used when switching from another operation that used
 * a different data format for srcA. Safe to call multiple times in fused kernels.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                                 | Type     | Valid Range                                         | Required |
 * |------------|----------------|-------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | Function   | in0_cb_id      | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                             | True     |
 * | Function   | in1_cb_id      | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                             | True     |
 * | Function   | old_in1_cb_id  | The identifier of the old in1_cb_id circular buffer (CB)   | uint32_t | 0 to 31                                             | True     |
 * | Function   | transpose      | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set   | False    |
 * | Function   | block_ct_dim   | The column dimension for the output block                  | uint32_t | Must be equal to block B column dimension           | False    |
 * | Function   | block_rt_dim   | The row dimension for the output block                     | uint32_t | Must be equal to block A row dimension              | False    |
 * | Function   | block_kt_dim   | The inner dimension                                         | uint32_t | Must be equal to block A column dimension           | False    |
 */
// clang-format on
ALWI void matmul_block_init_reconfig_data_format(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t old_in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t block_ct_dim = 1,
    uint32_t block_rt_dim = 1,
    uint32_t block_kt_dim = 1) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_in1_cb_id, in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_in1_cb_id, in1_cb_id)));
    matmul_block_init(in0_cb_id, in1_cb_id, transpose, block_ct_dim, block_rt_dim, block_kt_dim);
}

}  // namespace ckernel
