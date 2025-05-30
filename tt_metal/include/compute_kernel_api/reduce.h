// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_reduce_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the necessary hardware and software initialization for reduce operation. Needs to be called before reduce_tile.
 *
 * The `icb1` circular buffer must contain the scaling factors for the reduction:
 * - If `reduce_type = SUM`, all scaling factors should be 1.
 * - If `reduce_type = AVG`, all scaling factors should be 1/N (where N is the number of elements being averaged).
 * - If `reduce_type = MAX`, the scaling factors in `icb1` are ignored.
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                                     | Type      | Valid Range                                    | Required |
 * |------------|--------------|-----------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | reduce_type  | The type of reduce op - sum, average or maximum                 | PoolType  | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim   | The dimension of reduce op - row, column or both                | ReduceDim | {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Function   | icb0         | The identifier of the circular buffer (CB) containing operand A | uint32_t  | 0 to 31                                        | True     |
 * | Function   | icb1         | CB holding scaling factors (see above)                          | uint32_t  | 0 to 31                                        | True     |
 * | Function   | ocb          | The identifier of the output circular buffer (CB)               | uint32_t  | 0 to 31                                        | True     |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_init(uint32_t icb, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_AB_reduce_init<reduce_dim>(icb, icb1)));
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, MATH_FIDELITY>()));
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, reduce_dim>()));
}

// clang-format off
/**
 * Reverts the packer edge mask configuration to its default state by clearing any previously set masks. Needed to be called after
 * reduce op to reset the packer state to default.
 *
 * | Param Type | Name | Description                                      | Type | Valid Range | Required |
 * |------------|------|--------------------------------------------------|------|-------------|----------|
 * | Function   | —    | No parameters                                    |  —   |      —      |    —     |
 */
// clang-format on
ALWI void reduce_revert_delta() { PACK((llk_pack_reduce_mask_clear())); }

// clang-format off
/**
 * Performs a reduction operation *B = reduce(A)* using reduce_func for dimension reduction on a tile in the CB at a given index and writes the
 * result to the DST register at index *dst_tile_index*. Reduction can be of type *Reduce::R*, *Reduce::C*, or *Reduce::RC*, identifying the
 * dimension(s) to be reduced in size to 1. The DST register buffer must be in acquired state via *acquire_dst* call.
 *
 * The `icb1` circular buffer must contain the scaling factors for the reduction:
 * - If `reduce_type = SUM`, all scaling factors should be 1.
 * - If `reduce_type = AVG`, all scaling factors should be 1/N (where N is the number of elements being averaged).
 * - If `reduce_type = MAX`, the scaling factors in `icb1` are ignored.
 *
 * The templates take `reduce_type` which can be `ReduceFunc::Sum`, `ReduceFunc::Avg`, or `ReduceFunc::Max` and `reduce_dim` which can be `Reduce::R`, `Reduce::C`, or
 * `Reduce::RC`. They can also be specified by defines REDUCE_OP and REDUCE_DIM.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * | Param Type | Name     | Description                                                     | Type     | Valid Range                                    | Required |
 * |------------|----------|-----------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | Template   | reduce_type | The type of reduce op - sum, average or maximum              | PoolType | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim  | The dimension of reduce op - row, column or both             | ReduceDim| {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Function   | icb0     | The identifier of the circular buffer (CB) containing operand A | uint32_t | 0 to 31                                        | True     |
 * | Function   | icb1     | CB holding scaling factors (see above)                          | uint32_t | 0 to 31                                        | True     |
 * | Function   | itile0   | The index of the tile within the first CB                       | uint32_t | Must be less than the size of the CB           | True     |
 * | Function   | itile1   | The index of the tile within the scaling factor CB.             | uint32_t | Must be less than the size of the CB           | True     |
 * | Function   | idst     | The index of the tile in DST REG for the result                 | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_tile(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_reduce<reduce_type, reduce_dim, MATH_FIDELITY, DST_ACCUM_MODE>(icb0, icb1, idst)));
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
}

// clang-format off
/**
 * Performs a math-only reduction operation on a tile in the DST register. Assumes that source tiles are already in source registers.
 *
 * | Param Type | Name         | Description                                                     | Type      | Valid Range                                    | Required |
 * |------------|--------------|-----------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | reduce_type  | The type of reduce op - sum, average or maximum                 | PoolType  | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim   | The dimension of reduce op - row, column or both                | ReduceDim | {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Function   | idst         | The index of the tile in DST REG for the result                 | uint32_t  | Must be less than the acquired size of DST REG | True     |
 * | Function   | num_faces    | Number of faces to reduce (optional, default 4)                 | uint32_t  | >= 1                                           | False    |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4) {
    MATH((llk_math_reduce<reduce_type, reduce_dim, MATH_FIDELITY, DST_ACCUM_MODE>(idst, num_faces)));
}

}  // namespace ckernel
