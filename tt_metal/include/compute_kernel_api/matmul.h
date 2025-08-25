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
 * hardware settings that differ from the generic compute_kernel_hw_startup().
 *
 * - Unpacker: Configures matmul-specific dual-operand unpacking with CB data format mapping:
 *   * in0_cb_id → srcB (supports partial face for 8x32 tiles)
 *   * in1_cb_id → srcA (transpose operations applied here)
 * - Math Engine: Initializes FPU with MATH_FIDELITY precision and MM_THROTTLE performance settings
 * - Uses disaggregated configuration for matmul-specific tile dimensions and operand mappings
 *
 * Requires compute_kernel_hw_startup() to be called first for general UNPACK/MATH/PACK initialization.
 * CB data formats and tile dimensions must match those configured in compute_kernel_hw_startup().
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
ALWI void matmul_init(uint32_t in0_cb_id, uint32_t in1_cb_id, const bool transpose = 0) {
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
 * Unpacker: Loads tiles from CBs into srcA/srcB registers with configured data format conversion
 * Math Engine: Executes matrix multiplication using configured fidelity and throttle settings
 *
 * - in0_cb_id[in0_tile_index] goes into srcB (operand B, may be transposed)
 * - in1_cb_id[in1_tile_index] goes into srcA (operand A)
 * - srcA * srcB goes into DEST[dest_tile_index]
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
    const bool transpose = 0) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index)));
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(dest_tile_index, transpose)));
}

// clang-format off
/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles
 * already loaded in SRCA and SRCB registers and writes the result to DEST.
 * The DEST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * This is a math-only operation that operates on pre-loaded data in srcA/srcB registers.
 * Use after unpacker has loaded tiles via matmul_tile() unpacker phase, or when manually
 * managing unpacker and math phases separately for performance optimization.
 *
 * Math Engine: Executes matrix multiplication using configured MATH_FIDELITY and MM_THROTTLE
 * Face Processing: Handles variable number of tile faces (1-4) for different tile sizes
 * Row-major destination layout with configured accumulation mode
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
 * IMPORTANT: This function only reconfigures srcA data format. Use matmul_init_reconfig_data_format()
 * if you need to reconfigure both srcA and srcB data formats.
 *
 * Unpacker: Reconfigures data format mapping from old_srca_cb to in1_cb_id for srcA path only
 * Math Engine: Updates data format configuration for srcA operand processing only
 * Calls matmul_init() for complete matmul-specific hardware setup after reconfiguration
 *
 * Use case:
 * Essential for fused kernels where the same srcA register was previously configured
 * for a different operation (e.g., elementwise) with different data formats, and now
 * needs to be reconfigured for matmul operation with potentially different CB and format.
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
ALWI void matmul_init_reconfig_data_format_srca(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t old_srca_cb, const bool transpose = 0) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_srca_cb, in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_srca_cb, in1_cb_id)));
    matmul_init(in0_cb_id, in1_cb_id, transpose);
}

// clang-format off
/**
 * Initializes the matmul operation with data format reconfiguration for BOTH srcA and srcB.
 * This function should be used when switching from another operation that used
 * different data formats for both operands. Safe to call multiple times in fused kernels.
 *
 * IMPORTANT: This function reconfigures BOTH srcA and srcB data formats. Use
 * matmul_init_reconfig_data_format_srca() if you only need to reconfigure srcA.
 *
 * Unpacker: Reconfigures data format mapping for both srcA and srcB operand paths
 * Math Engine: Updates data format configuration for both operands
 * Calls matmul_init() for complete matmul-specific hardware setup after reconfiguration
 *
 * Use case:
 * Essential for fused kernels where both operands were previously configured for different
 * operations with different data formats or CB mappings. Used when transitioning from
 * operations that affect both input data paths.
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                                 | Type     | Valid Range                                      | Required |
 * |------------|--------------|-------------------------------------------------------------|----------|--------------------------------------------------|----------|
 * | Function   | in0_cb_id    | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                          | True     |
 * | Function   | in1_cb_id    | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                          | True     |
 * | Function   | old_in0_cb   | The identifier of the old in0_cb_id circular buffer (CB)   | uint32_t | 0 to 31                                          | True     |
 * | Function   | old_srca_cb  | The identifier of the old srcA circular buffer (CB)        | uint32_t | 0 to 31                                          | True     |
 * | Function   | transpose    | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set| False    |
 */
// clang-format on
ALWI void matmul_init_reconfig_data_format(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t old_in0_cb, uint32_t old_srca_cb, const bool transpose = 0) {
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE>(old_srca_cb, in1_cb_id, old_in0_cb, in0_cb_id)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE>(old_srca_cb, in1_cb_id, old_in0_cb, in0_cb_id)));
    matmul_init(in0_cb_id, in1_cb_id, transpose);
}

// clang-format off
/**
 * Initializes the matmul block operation for subsequent block operations. Must be called before matmul_block.
 * This function is safe to call multiple times in fused kernels and only configures matmul-specific
 * hardware settings that differ from the generic compute_kernel_hw_startup().
 *
 * Unpacker: Configures block-aware dual-operand unpacking with multi-tile block dimensions
 *   * in0_cb_id → srcB (supports partial face, handles rt_dim × kt_dim blocks)
 *   * in1_cb_id → srcA (transpose operations, handles kt_dim × ct_dim blocks)
 * Math Engine: Initializes FPU for block operations with MATH_FIDELITY and MM_THROTTLE
 * Block Optimization: Configures reuse patterns based on block dimensions for efficiency
 *
 * Block dimensions:
 * - block_ct_dim: Column tiles in output block (affects srcB reuse pattern)
 * - block_rt_dim: Row tiles in output block (affects srcA reuse pattern)
 * - block_kt_dim: Inner dimension tiles (accumulation depth)
 * - Total output: block_rt_dim × block_ct_dim tiles per block
 *
 * Requires compute_kernel_hw_startup() to be called first.
 * Block dimensions must match subsequent matmul_block() calls.
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
    const bool transpose = 0,
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
 * Unpacker: Loads multi-tile blocks from CBs using block-aware addressing and stride patterns
 * Math Engine: Executes block matrix multiplication with optimized data reuse
 * Block Processing: Handles rt_dim × ct_dim output blocks with kt_dim inner dimension
 * Automatic tile indexing within blocks based on configured block dimensions
 *
 * - in0_cb_id[in0_tile_index + block offsets] goes into srcB (rt_dim × kt_dim block)
 * - in1_cb_id[in1_tile_index + block offsets] goes into srcA (kt_dim × ct_dim block)
 * - Produces rt_dim × ct_dim output tiles in DEST starting at dest_tile_index
 *
 * Optimized data reuse patterns based on block dimensions
 * Uses fidelity and throttle settings from matmul_block_init()
 * Row-major destination tile arrangement
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
    const bool transpose,
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
 * IMPORTANT: This function only reconfigures srcA data format. Use matmul_block_init_reconfig_data_format()
 * if you need to reconfigure both srcA and srcB data formats.
 *
 * Unpacker: Reconfigures data format mapping from old_in1_cb_id to in1_cb_id for srcA path only
 * Math Engine: Updates data format configuration for srcA operand in block operations only
 * Calls matmul_block_init() for complete block-aware matmul hardware setup after reconfiguration
 * Preserves block dimension configurations (ct_dim, rt_dim, kt_dim) across reconfiguration
 *
 * Use case:
 * Essential for fused kernels performing multiple block operations where srcA was previously
 * configured for a different operation with different data formats or CB mappings. Commonly
 * used in attention mechanisms where intermediate results are reused with format changes.
 *
 * Block context:
 * Maintains block-aware unpacker configuration with updated data format mappings
 * Supports complex fused operation sequences with varying data formats
 * Optimizes hardware reconfiguration by only updating necessary format mappings
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
ALWI void matmul_block_init_reconfig_data_format_srca(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t old_in1_cb_id,
    const bool transpose = 0,
    uint32_t block_ct_dim = 1,
    uint32_t block_rt_dim = 1,
    uint32_t block_kt_dim = 1) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_in1_cb_id, in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_in1_cb_id, in1_cb_id)));
    matmul_block_init(in0_cb_id, in1_cb_id, transpose, block_ct_dim, block_rt_dim, block_kt_dim);
}

// clang-format off
/**
 * Initializes the matmul block operation with data format reconfiguration for BOTH srcA and srcB.
 * This function should be used when switching from another operation that used
 * different data formats for both operands. Safe to call multiple times in fused kernels.
 *
 * IMPORTANT: This function reconfigures BOTH srcA and srcB data formats. Use
 * matmul_block_init_reconfig_data_format_srca() if you only need to reconfigure srcA.
 *
 * Unpacker: Reconfigures data format mapping for both srcA and srcB operand paths
 * Math Engine: Updates data format configuration for both operands in block operations
 * Calls matmul_block_init() for complete block-aware matmul hardware setup after reconfiguration
 * Preserves block dimension configurations (ct_dim, rt_dim, kt_dim) across reconfiguration
 *
 * Use case:
 * Essential for fused kernels where both operands were previously configured for different
 * operations with different data formats or CB mappings. Used when transitioning from
 * operations that affect both input data paths (e.g., conv to matmul transitions).
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                                 | Type     | Valid Range                                         | Required |
 * |------------|----------------|-------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | Function   | in0_cb_id      | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                             | True     |
 * | Function   | in1_cb_id      | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                             | True     |
 * | Function   | old_in0_cb_id  | The identifier of the old in0_cb_id circular buffer (CB)   | uint32_t | 0 to 31                                             | True     |
 * | Function   | old_in1_cb_id  | The identifier of the old in1_cb_id circular buffer (CB)   | uint32_t | 0 to 31                                             | True     |
 * | Function   | transpose      | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set   | False    |
 * | Function   | ct_dim         | The column dimension for the output block                  | uint32_t | Must be equal to block B column dimension           | False    |
 * | Function   | rt_dim         | The row dimension for the output block                     | uint32_t | Must be equal to block A row dimension              | False    |
 * | Function   | kt_dim         | The inner dimension                                         | uint32_t | Must be equal to block A column dimension           | False    |
 */
// clang-format on
ALWI void matmul_block_init_reconfig_data_format(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t old_in0_cb_id,
    uint32_t old_in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE>(old_in1_cb_id, in1_cb_id, old_in0_cb_id, in0_cb_id)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE>(old_in1_cb_id, in1_cb_id, old_in0_cb_id, in0_cb_id)));
    matmul_block_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
}

}  // namespace ckernel
