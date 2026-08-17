// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#include "tensor_shape.h"
#ifdef TRISC_MATH
#include "llk_math_reduce_api.h"
#include "experimental/llk_math_reduce_custom_api.h"
#include "experimental/llk_math_reduce_custom_runtime_api.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_AB_reduce_custom_api.h"
#include "experimental/llk_unpack_AB_reduce_custom_runtime_api.h"
#endif

#ifdef TRISC_PACK
#include "llk_pack_reduce_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initialization for reduce_block_max_row operation. Must be called before reduce_block_max_row.
 * Processes a block of tiles in the width dimension, reducing each row across the block.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32 (num_faces=4) or 16x32 (num_faces=2, a single face-row)
 * - Can work on both 16-bit or 32-bit DEST register modes based on DST_ACCUM_MODE
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native reduce_init API.
 * Use the standard reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>() with reduce_tile() in a loop
 * for general-purpose reduction across multiple tiles.
 *
 * respect_trigger parameter enables an optimization used in SDPA (Scaled Dot-Product Attention)
 * kernels to increase utilization. When enabled, it splits the unpack MOP (Macro Operation) into two halves
 * with hardware semaphore synchronization, allowing better pipelining and avoiding a more costly circular buffer
 * synchronization. The same value has to be passed to init, execute and uninit functions for this to take effect.
 *
 * NOTE: Be extra careful when setting respect_trigger to true. This feature breaks the LLK API contract in
 * the following way: the llk-lib layer in reduce_block_max_row is waiting and acquiring the semaphore,
 * but posting it is expected to be done by the packer in the compute kernel, i.e. 2 layers above.
 * Number of semposts must match the number of calls to reduce_block_max_row_uninit.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | block_ct_dim              | The number of tiles in the width dimension to process as a block                        | uint32_t  | 1 to 2^32-1                                   | True     |
 * | Template   | respect_trigger           | Triggers MOP split optimization                                                         | bool      | {true, false}                                  | False    |
 * | Function   | tensor_shape              | Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile)            | ckernel::TensorShape | N/A                                 | True     |
 * | Function   | ocb                       | The identifier of the output circular buffer (CB)                                       | uint32_t  | 0 to 31                                        | True     |
 */
// clang-format on
template <std::uint32_t block_ct_dim, bool respect_trigger = false>
ALWI void reduce_block_max_row_init(const ckernel::TensorShape& tensor_shape, std::uint32_t ocb) {
    UNPACK((llk_unpack_AB_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE, respect_trigger>(tensor_shape)));
    MATH((llk_math_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE>(tensor_shape)));
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(ocb)));
}

// num_faces convenience overload: constructs a TensorShape from a flat face count (2 or 4).
template <std::uint32_t block_ct_dim, bool respect_trigger = false, std::uint32_t num_faces = 4>
ALWI void reduce_block_max_row_init(std::uint32_t ocb) {
    reduce_block_max_row_init<block_ct_dim, respect_trigger>(ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces), ocb);
}

// clang-format off
/**
 * Performs block-based max row reduction operation on a block of tiles in the width dimension.
 * Reduces each row across the block of tiles and writes the result to DST register.
 * The DST register buffer must be in acquired state via acquire_dst call.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32 (num_faces=4) or 16x32 (num_faces=2, a single face-row)
 * - Can work on both 16-bit or 32-bit DEST register modes based on DST_ACCUM_MODE
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native reduce_tile API.
 * Use the standard reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>() with reduce_tile() in a loop
 * for general-purpose reduction across multiple tiles.
 *
 * respect_trigger parameter enables an optimization used in SDPA (Scaled Dot-Product Attention)
 * kernels to increase utilization. When enabled, it splits the unpack MOP (Macro Operation) into two halves
 * with hardware semaphore synchronization, allowing better pipelining and avoiding a more costly circular buffer
 * synchronization. The same value has to be passed to init, execute and uninit functions for this to take effect.
 *
 * NOTE: Be extra careful when setting respect_trigger to true. This feature breaks the LLK API contract in
 * the following way: the llk-lib layer in reduce_block_max_row is waiting and acquiring the semaphore,
 * but posting it is expected to be done by the packer in the compute kernel, i.e. 2 layers above.
 * Number of semposts must match the number of calls to reduce_block_max_row_uninit.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | block_ct_dim              | The number of tiles in the width dimension to process as a block                        | uint32_t  | 1 to 2^32-1                                   | True     |
 * | Template   | respect_trigger           | Triggers MOP split optimization                                                         | bool      | {true, false}                                  | False    |
 * | Function   | tensor_shape              | Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile)            | ckernel::TensorShape | N/A                                 | True     |
 * | Function   | icb                       | The identifier of the circular buffer (CB) containing operand A                         | uint32_t  | 0 to 31                                        | True     |
 * | Function   | icb_scaler                | CB holding scaling factors                                                              | uint32_t  | 0 to 31                                        | True     |
 * | Function   | row_start_index           | The starting tile index for the row being processed                                     | uint32_t  | Must be less than the size of the CB           | True     |
 * | Function   | idst                      | The index of the tile in DST REG for the result                                         | uint32_t  | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <std::uint32_t block_ct_dim, bool respect_trigger = false>
ALWI void reduce_block_max_row(
    const ckernel::TensorShape& tensor_shape,
    std::uint32_t icb,
    std::uint32_t icb_scaler,
    std::uint32_t row_start_index,
    std::uint32_t idst) {
    UNPACK((llk_unpack_AB_reduce_block_max_row<block_ct_dim, respect_trigger>(icb, icb_scaler, row_start_index)));
    MATH((llk_math_reduce_block_max_row<block_ct_dim, DST_ACCUM_MODE>(idst, tensor_shape)));
}

// num_faces convenience overload: constructs a TensorShape from a flat face count (2 or 4).
template <std::uint32_t block_ct_dim, bool respect_trigger = false, std::uint32_t num_faces = 4>
ALWI void reduce_block_max_row(
    std::uint32_t icb, std::uint32_t icb_scaler, std::uint32_t row_start_index, std::uint32_t idst) {
    reduce_block_max_row<block_ct_dim, respect_trigger>(
        ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces), icb, icb_scaler, row_start_index, idst);
}

#ifdef ARCH_BLACKHOLE
// clang-format off
/**
 * Lightweight Blackhole-only reinit path used when reduce follows custom SDPA sub path.
 * Reprograms reduce MOP and restores only the reduce addrmods.
 *
 * respect_trigger parameter enables an optimization used in SDPA (Scaled Dot-Product Attention)
 * kernels to increase utilization. When enabled, it splits the unpack MOP (Macro Operation) into two halves
 * with hardware semaphore synchronization, allowing better pipelining and avoiding a more costly circular buffer
 * synchronization. The same value has to be passed to init, execute and uninit functions for this to take effect.
 *
 * NOTE: Be extra careful when setting respect_trigger to true. This feature breaks the LLK API contract in
 * the following way: the llk-lib layer in reduce_block_max_row is waiting and acquiring the semaphore,
 * but posting it is expected to be done by the packer in the compute kernel, i.e. 2 layers above.
 * Number of semposts must match the number of calls to reduce_block_max_row_uninit.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | block_ct_dim              | The number of tiles in the width dimension to process as a block                        | uint32_t  | 1 to 2^32-1                                   | True     |
 * | Template   | respect_trigger           | Triggers MOP split optimization                                                         | bool      | {true, false}                                  | False    |
 * | Function   | tensor_shape              | Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile)            | ckernel::TensorShape | N/A                                 | True     |
 * | Function   | ocb                       | The identifier of the output circular buffer (CB)                                       | uint32_t  | 0 to 31                                        | True     |
 */
// clang-format on
template <std::uint32_t block_ct_dim, bool respect_trigger = false>
ALWI void reduce_block_max_row_reinit_short(const ckernel::TensorShape& tensor_shape, std::uint32_t ocb) {
    UNPACK((llk_unpack_AB_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE, respect_trigger>(tensor_shape)));
    MATH((llk_math_reduce_block_max_row_reinit_with_mop<block_ct_dim>(tensor_shape)));
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(ocb)));
}

// num_faces convenience overload: constructs a TensorShape from a flat face count (2 or 4).
template <std::uint32_t block_ct_dim, bool respect_trigger = false, std::uint32_t num_faces = 4>
ALWI void reduce_block_max_row_reinit_short(std::uint32_t ocb) {
    reduce_block_max_row_reinit_short<block_ct_dim, respect_trigger>(
        ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces), ocb);
}
#endif

#ifdef ARCH_BLACKHOLE
/**
 * Minimal reinit: only ADDR_MOD_1 + ADDR_MOD_2 + ADDR_MOD_6. Requires copy_tile_custom
 * (which uses ADDR_MOD_4) so ADDR_MOD_3 is preserved from the previous reduce.
 */
template <std::uint32_t block_ct_dim, bool respect_trigger = false>
ALWI void reduce_block_max_row_reinit_minimal(const ckernel::TensorShape& tensor_shape, std::uint32_t ocb) {
    UNPACK((llk_unpack_AB_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE, respect_trigger>(tensor_shape)));
    MATH((llk_math_reduce_block_max_row_reinit_minimal()));
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(ocb)));
}

// num_faces convenience overload: constructs a TensorShape from a flat face count (2 or 4).
template <std::uint32_t block_ct_dim, bool respect_trigger = false, std::uint32_t num_faces = 4>
ALWI void reduce_block_max_row_reinit_minimal(std::uint32_t ocb) {
    reduce_block_max_row_reinit_minimal<block_ct_dim, respect_trigger>(
        ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces), ocb);
}

/**
 * Minimal reinit (runtime variant): only ADDR_MOD_1 + ADDR_MOD_2 + ADDR_MOD_6.
 * Requires copy_tile_custom (which uses ADDR_MOD_4) so ADDR_MOD_3 is preserved
 * from the previous reduce.
 */
ALWI void reduce_block_max_row_reinit_minimal_runtime(
    const ckernel::TensorShape& tensor_shape,
    std::uint32_t ocb,
    std::uint32_t block_ct_dim,
    bool respect_trigger = false) {
    UNPACK(
        (llk_unpack_AB_reduce_block_max_row_init_runtime<DST_ACCUM_MODE>(block_ct_dim, respect_trigger, tensor_shape)));
    MATH((llk_math_reduce_block_max_row_reinit_minimal_runtime()));
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(ocb)));
}

// num_faces convenience overload: constructs a TensorShape from a flat face count (2 or 4).
ALWI void reduce_block_max_row_reinit_minimal_runtime(
    std::uint32_t ocb, std::uint32_t block_ct_dim, bool respect_trigger = false, std::uint32_t num_faces = 4) {
    reduce_block_max_row_reinit_minimal_runtime(
        ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces), ocb, block_ct_dim, respect_trigger);
}

/**
 * Short reinit (runtime variant): Reprograms reduce MOP and restores addrmods.
 * Used when reduce follows custom SDPA sub path with runtime block_ct_dim.
 */
ALWI void reduce_block_max_row_reinit_short_runtime(
    const ckernel::TensorShape& tensor_shape,
    std::uint32_t ocb,
    std::uint32_t block_ct_dim,
    bool respect_trigger = false) {
    UNPACK(
        (llk_unpack_AB_reduce_block_max_row_init_runtime<DST_ACCUM_MODE>(block_ct_dim, respect_trigger, tensor_shape)));
    MATH((llk_math_reduce_block_max_row_reinit_short_runtime<DST_ACCUM_MODE>(block_ct_dim, tensor_shape)));
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(ocb)));
}

// num_faces convenience overload: constructs a TensorShape from a flat face count (2 or 4).
ALWI void reduce_block_max_row_reinit_short_runtime(
    std::uint32_t ocb, std::uint32_t block_ct_dim, bool respect_trigger = false, std::uint32_t num_faces = 4) {
    reduce_block_max_row_reinit_short_runtime(
        ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces), ocb, block_ct_dim, respect_trigger);
}
#endif

// clang-format off
/**
 * Uninitializes the block-based reduce_max_row operation. Needs to be called after the last call to `reduce_block_max_row` before initializing another operation.
 * This version is for block-based reduction across multiple tiles processed together.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on clear_fp32_accumulation flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native reduce_uninit API.
 * Use the standard reduce_uninit() for general-purpose reduction cleanup.
 *
 * respect_trigger parameter enables an optimization used in SDPA (Scaled Dot-Product Attention)
 * kernels to increase utilization. When enabled, it splits the unpack MOP (Macro Operation) into two halves
 * with hardware semaphore synchronization, allowing better pipelining and avoiding a more costly circular buffer
 * synchronization. The same value has to be passed to init, execute and uninit functions for this to take effect.
 *
 * NOTE: Be extra careful when setting respect_trigger to true. This feature breaks the LLK API contract in
 * the following way: the llk-lib layer in reduce_block_max_row is waiting and acquiring the semaphore,
 * but posting it is expected to be done by the packer in the compute kernel, i.e. 2 layers above.
 * Number of semposts must match the number of calls to reduce_block_max_row_uninit.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | respect_trigger           | Triggers MOP split optimization                                                         | bool      | {true, false}                                  | False    |
 * | Function   | icb                       | The identifier of the circular buffer (CB) containing operand A                         | uint32_t  | 0 to 31                                        | False    |
 */
// clang-format on
template <bool respect_trigger = false>
ALWI void reduce_block_max_row_uninit(std::uint32_t icb) {
#ifdef ARCH_BLACKHOLE
    MATH((llk_math_reduce_uninit()));
#else
    // Required because MOVB2D/D2B depends on SrcA ALU Format - Hi/Lo16 does not work with Tf32 (only on WH)
    // This is needed because FP32 data from L1 that is unpacked to Src registers is reduced to Tf32
    // See _llk_math_reduce_init_ for more details
    MATH((llk_math_reduce_uninit(icb)));
#endif
    PACK((llk_pack_reduce_mask_clear()));
    UNPACK((llk_unpack_AB_reduce_block_max_row_uninit<respect_trigger>()));
}

// Runtime variants - block_ct_dim and respect_trigger are runtime parameters.
ALWI void reduce_block_max_row_init_runtime(
    const ckernel::TensorShape& tensor_shape,
    std::uint32_t ocb,
    std::uint32_t block_ct_dim,
    bool respect_trigger = false) {
    UNPACK(
        (llk_unpack_AB_reduce_block_max_row_init_runtime<DST_ACCUM_MODE>(block_ct_dim, respect_trigger, tensor_shape)));
    MATH((llk_math_reduce_block_max_row_init_runtime<DST_ACCUM_MODE>(block_ct_dim, tensor_shape)));
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(ocb)));
}

// num_faces convenience overload: constructs a TensorShape from a flat face count (2 or 4).
ALWI void reduce_block_max_row_init_runtime(
    std::uint32_t ocb, std::uint32_t block_ct_dim, bool respect_trigger = false, std::uint32_t num_faces = 4) {
    reduce_block_max_row_init_runtime(
        ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces), ocb, block_ct_dim, respect_trigger);
}

ALWI void reduce_block_max_row_runtime(
    const ckernel::TensorShape& tensor_shape,
    std::uint32_t icb,
    std::uint32_t icb_scaler,
    std::uint32_t row_start_index,
    std::uint32_t idst,
    bool respect_trigger = false,
    bool overlap_first_half = false) {
    UNPACK((llk_unpack_AB_reduce_block_max_row_runtime(
        icb, icb_scaler, row_start_index, respect_trigger, overlap_first_half)));
    MATH((llk_math_reduce_block_max_row_runtime<DST_ACCUM_MODE>(idst, tensor_shape)));
}

// num_faces convenience overload: constructs a TensorShape from a flat face count (2 or 4).
ALWI void reduce_block_max_row_runtime(
    std::uint32_t icb,
    std::uint32_t icb_scaler,
    std::uint32_t row_start_index,
    std::uint32_t idst,
    bool respect_trigger = false,
    bool overlap_first_half = false,
    std::uint32_t num_faces = 4) {
    reduce_block_max_row_runtime(
        ckernel::tensor_shape_from_num_faces(ckernel::MAX_FACE_R_DIM, num_faces),
        icb,
        icb_scaler,
        row_start_index,
        idst,
        respect_trigger,
        overlap_first_half);
}

ALWI void reduce_block_max_row_uninit_runtime(
    std::uint32_t icb, bool respect_trigger = false, bool overlap_first_half = false) {
#ifdef ARCH_BLACKHOLE
    MATH((llk_math_reduce_uninit()));
#else
    MATH((llk_math_reduce_uninit(icb)));
#endif
    PACK((llk_pack_reduce_mask_clear()));
    UNPACK((llk_unpack_AB_reduce_block_max_row_uninit_runtime(respect_trigger, overlap_first_half)));
}

}  // namespace ckernel
