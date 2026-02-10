// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_custom_mm_api.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_custom_mm_api.h"
#endif
namespace ckernel {

// clang-format off
/**
 * Full initialization for custom_mm_block operation. Must be called before custom_mm_block and only once at the beggining of the kernel.
 * For initializing custom_mm_block in the middle of the kernel, please use custom_mm_block_init_short.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument       | Description                                                                            | Type     | Valid Range                  | Required              |
 * |----------------|----------------------------------------------------------------------------------------|----------|------------------------------|-----------------------|
 * | transpose      | The transpose flag for performing transpose operation on in1                           | bool     | true/false                   | False (default false) |
 * | split_acc      | Wether to accumulate partials within a single tile in different dest locations         | bool     | true/false                   | False (default false) |
 * | dense_packing  | Whether to pack consecutive tiles 32 rows apart (instead of 64, doubles dest capacity) | bool     | true/false                   | False (default false) |
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                                 | uint32_t | 0 to 31                      | True                  |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                                | uint32_t | 0 to 31                      | True                  |
 * | out_cb_id      | The identifier of the output circular buffer (CB)                                      | uint32_t | 0 to 31                      | True                  |
 * | ct_dim         | The width of the output matrix in tiles                                                | uint32_t | {1, 2, 4, 8, 10, 12, 14, 16} | False (default 1)     |
 */
// clang-format on
template <bool transpose = false, bool split_acc = false, bool dense_packing = false, bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void custom_mm_block_init(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    // Intentionally swap in0 and in1 as operation specific hw_configures are deprecated
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_custom_mm_init<transpose>(in0_cb_id, in1_cb_id, ct_dim)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_custom_mm_init<transpose, split_acc, dense_packing>(in0_cb_id, in1_cb_id, ct_dim)));

    PACK((llk_pack_dest_init<fp32_dest_acc_en, false>()));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_init<false, false>(out_cb_id)));
    if constexpr (dense_packing) {
        // Reduce packing stride from tile to tile to 32 rows instead of 64
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

// clang-format off
/**
 * Short initialization for custom_mm_block operation. Must be called before custom_mm_block and is safe to call at any point in the kernel.
 * For initializing custom_mm_block at the beginning of the kernel, please use custom_mm_block_init.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument       | Description                                                                            | Type     | Valid Range                  | Required              |
 * |----------------|----------------------------------------------------------------------------------------|----------|------------------------------|-----------------------|
 * | transpose      | The transpose flag for performing transpose operation on in1                           | bool     | true/false                   | False (default false) |
 * | split_acc      | Wether to accumulate partials within a single tile in different dest locations         | bool     | true/false                   | False (default false) |
 * | dense_packing  | Whether to pack consecutive tiles 32 rows apart (instead of 64, doubles dest capacity) | bool     | true/false                   | False (default false) |
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                                 | uint32_t | 0 to 31                      | True                  |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                                | uint32_t | 0 to 31                      | True                  |
 * | out_cb_id      | The identifier of the output circular buffer (CB)                                      | uint32_t | 0 to 31                      | True                  |
 * | ct_dim         | The width of the output matrix in tiles                                                | uint32_t | {1, 2, 4, 8, 10, 12, 14, 16} | False (default 1)     |
 */
// clang-format on
template <bool transpose = false, bool split_acc = false, bool dense_packing = false>
ALWI void custom_mm_block_init_short(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm_init<transpose>(in0_cb_id, in1_cb_id, ct_dim)));

    MATH((llk_math_custom_mm_init<transpose, split_acc, dense_packing>(in0_cb_id, in1_cb_id, ct_dim)));

    PACK((llk_pack_init<false, false>(out_cb_id)));
    if constexpr (dense_packing) {
        // Reduce packing stride from tile to tile to 32 rows instead of 64
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument        | Description                                                                                                    | Type     | Valid Range                                      | Required              |
 * |-----------------|----------------------------------------------------------------------------------------------------------------|----------|--------------------------------------------------|-----------------------|
 * | finalize        | Wether to perform the finalization step which merges split_accumulation partials                               | bool     | true/false (must be false if split_acc is false) | False (default true)  |
 * | read_transposed | Wether to read in1 tiles in transposed order (read ct tiles with a stride of kt, then move over a single tile) | bool     | true/false                                       | False (default false) |
 * | in0_cb_id       | The identifier of the first input circular buffer (CB)                                                         | uint32_t | 0 to 31                                          | True                  |
 * | in1_cb_id       | The identifier of the second input circular buffer (CB)                                                        | uint32_t | 0 to 31                                          | True                  |
 * | in0_tile_index  | The index of the tile in block A from the first input CB                                                       | uint32_t | Must be less than the size of the CB             | True                  |
 * | in1_tile_index  | The index of the tile in block B from the second input CB                                                      | uint32_t | Must be less than the size of the CB             | True                  |
 * | dst_index       | The index of the tile in DST REG to which the result C will be written                                         | uint32_t | Must be less than the acquired size of DST REG   | True                  |
 * | kt_dim          | The inner dimension in tiles                                                                                   | uint32_t | Must be an even number from 2 to 256 (inclusive) | True                  |
 * | ct_dim          | The width of the output matrix in tiles                                                                        | uint32_t | {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}              | False (default 1)     |
 */
// clang-format on
template <bool finalize = true, bool read_transposed = false>
ALWI void custom_mm_block(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm<read_transposed>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim, ct_dim)));
    MATH((llk_math_custom_mm<finalize>(in0_cb_id, in1_cb_id, dst_index, kt_dim, ct_dim)));
}

// clang-format off
/**
 * Performs unpack part of block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument        | Description                                                                                                    | Type     | Valid Range                                      | Required              |
 * |-----------------|----------------------------------------------------------------------------------------------------------------|----------|--------------------------------------------------|-----------------------|
 * | read_transposed | Wether to read in1 tiles in transposed order (read ct tiles with a stride of kt, then move over a single tile) | bool     | true/false                                       | False (default false) |
 * | in0_cb_id       | The identifier of the first input circular buffer (CB)                                                         | uint32_t | 0 to 31                                          | True                  |
 * | in1_cb_id       | The identifier of the second input circular buffer (CB)                                                        | uint32_t | 0 to 31                                          | True                  |
 * | in0_tile_index  | The index of the tile in block A from the first input CB                                                       | uint32_t | Must be less than the size of the CB             | True                  |
 * | in1_tile_index  | The index of the tile in block B from the second input CB                                                      | uint32_t | Must be less than the size of the CB             | True                  |
 * | kt_dim          | The inner dimension in tiles                                                                                   | uint32_t | Must be an even number from 2 to 256 (inclusive) | True                  |
 * | ct_dim          | The width of the output matrix in tiles                                                                        | uint32_t | {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}              | False (default 1)     |
 */
// clang-format on
template <bool read_transposed = false>
ALWI void custom_mm_block_unpack(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm<read_transposed>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim, ct_dim)));
}

// clang-format off
/**
 * Performs math part of block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Return value: None
 *
 * | Argument        | Description                                                                                                    | Type     | Valid Range                                      | Required              |
 * |-----------------|----------------------------------------------------------------------------------------------------------------|----------|--------------------------------------------------|-----------------------|
 * | finalize        | Wether to perform the finalization step which merges split_accumulation partials                               | bool     | true/false (must be false if split_acc is false) | False (default true)  |
 * | in0_cb_id       | The identifier of the first input circular buffer (CB)                                                         | uint32_t | 0 to 31                                          | True                  |
 * | in1_cb_id       | The identifier of the second input circular buffer (CB)                                                        | uint32_t | 0 to 31                                          | True                  |
 * | dst_index       | The index of the tile in DST REG to which the result C will be written                                         | uint32_t | Must be less than the acquired size of DST REG   | True                  |
 * | kt_dim          | The inner dimension in tiles                                                                                   | uint32_t | Must be an even number from 2 to 256 (inclusive) | True                  |
 * | ct_dim          | The width of the output matrix in tiles                                                                        | uint32_t | {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}              | False (default 1)     |
 */
// clang-format on
template <bool finalize = true>
ALWI void custom_mm_block_math(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    MATH((llk_math_custom_mm<finalize>(in0_cb_id, in1_cb_id, dst_index, kt_dim, ct_dim)));
}

// clang-format off
/**
 * Uninitializes the custom_mm_block operation, must be called after the final custom_mm_block call in a sequence and before initializing another operation.
 *
 *
 * Return value: None
 *
 * | Argument       | Description                                                                            | Type     | Valid Range                  | Required              |
 * |----------------|----------------------------------------------------------------------------------------|----------|------------------------------|-----------------------|
 * | dense_packing  | Whether to pack consecutive tiles 32 rows apart (instead of 64, doubles dest capacity) | bool     | true/false                   | False (default false) |
 */
// clang-format on
template <bool dense_packing = false>
ALWI void custom_mm_block_uninit() {
    if constexpr (dense_packing) {
        // Restore default packing stride of 64 rows between tiles
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

}  // namespace ckernel
