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
 * Performs the necessary hardware and software initialization for reduce operation for provided circular buffer identifiers (CB IDs). In order for reduce
 * operation to be performed, this function call must be followed by a call to `reduce_tile` or `reduce_tile_math`.
 * If another reduce op is needed which uses different CB IDs, then another reduce_init needs to be called as a part of that reduce operation.
 *
 * The `icb_scaler` circular buffer must contain the scaling factors for the reduction. The most straightforward way of filling the `icb_scaler` with the scaling
 * factors is to populate first row of each face with the followin values:
 * - If `reduce_type = SUM`, all scaling factors should preferably be 1.
 * - If `reduce_type = AVG`, all scaling factors should preferably be 1/N (where N is the number of elements being averaged, except if the reduction dimension is scalar,
 * in which case the scaling factor should be 1/sqrt(N)).
 * - If `reduce_type = MAX`, all scaling factors should preferably be 1.
 *
 * NOTE: For SUM and AVG operations, the value in `icb_scaler` is a scaling factor of the final sum of values across rows/columns/both, so there is no real constraint in terms
 * of it's value. For MAX operation, maximum value will be obtained as expected, but it will be scaled by the values in `icb_scaler`. In any case, it is recommended to use the
 * above-mentioned scaling factors to ensure that operations function as intended. Refer to ISA documentation for more details.
 * NOTE: For other valid ways of populating the `icb_scaler`, refer to the ISA documentation.
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                                     | Type      | Valid Range                                    | Required |
 * |------------|--------------|-----------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | reduce_type  | The type of reduce op - sum, average or maximum                 | PoolType  | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim   | The dimension of reduce op - row, column or both                | ReduceDim | {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Function   | icb          | The identifier of the circular buffer (CB) containing operand A | uint32_t  | 0 to 31                                        | True     |
 * | Function   | icb_scaler   | CB holding scaling factors (see above)                          | uint32_t  | 0 to 31                                        | True     |
 * | Function   | ocb          | The identifier of the output circular buffer (CB)               | uint32_t  | 0 to 31                                        | True     |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb) {
    UNPACK((llk_unpack_AB_reduce_init<reduce_dim>(icb, icb_scaler)));
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, MATH_FIDELITY>()));
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, reduce_dim>()));
}

// clang-format off
/**
 * Resets the packer edge mask configuration to its default state by clearing any previously set masks. Needs to be called after
 * reduce_tile if the next operation requires default packer state. In case that the next operation is reduce operation across the
 * same dimension, this call can be omitted. If this function is not called, the packer will continue to use the edge masks set
 * by the latest reduce_init call, which may lead to incorrect packing behavior in subsequent operations.
 *
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025.
 *
 * | Param Type | Name | Description                                      | Type | Valid Range | Required |
 * |------------|------|--------------------------------------------------|------|-------------|----------|
 * | Function   | —    | No parameters                                    |  —   |      —      |    —     |
 */
// clang-format on
ALWI void reduce_uninit() { PACK((llk_pack_reduce_mask_clear())); }

// clang-format off
/**
 * Performs a reduction operation *B = reduce(A)* using reduce_func for dimension reduction on a tile in the CB at a given index and writes the
 * result to the DST register at index *dst_tile_index*. Reduction can be of type *Reduce::R*, *Reduce::C*, or *Reduce::RC*, identifying the
 * dimension(s) to be reduced in size to 1. The DST register buffer must be in acquired state via *acquire_dst* call.
 *
 * The `icb_scaler` circular buffer must contain the scaling factors for the reduction. The most straightforward way of filling the `icb_scaler` with the scaling
 * factors is to populate first row of each face with the followin values:
 * - If `reduce_type = SUM`, all scaling factors should preferably be 1.
 * - If `reduce_type = AVG`, all scaling factors should preferably be 1/N (where N is the number of elements being averaged, except if the reduction dimension is scalar,
 * in which case the scaling factor should be 1/sqrt(N)).
 * - If `reduce_type = MAX`, all scaling factors should preferably be 1.
 *
 * The templates take `reduce_type` which can be `ReduceFunc::Sum`, `ReduceFunc::Avg`, or `ReduceFunc::Max` and `reduce_dim` which can be `Reduce::R`, `Reduce::C`, or
 * `Reduce::RC`. They can also be specified by defines REDUCE_OP and REDUCE_DIM.
 *
 * NOTE: Before the next operation is initialized, the `reduce_uninit` function must be called to reset the packer state to default.
 * NOTE: For SUM and AVG operations, the value in `icb_scaler` is a scaling factor of the final sum of values across rows/columns/both, so there is no real constraint in terms
 * of it's value. For MAX operation, maximum value will be obtained as expected, but it will be scaled by the values in `icb_scaler`. In any case, it is recommended to use the
 * above-mentioned scaling factors to ensure that operations function as intended. Refer to ISA documentation for more details.
 * NOTE: For other valid ways of populating the `icb_scaler`, refer to the ISA documentation.
 * Return value: None
 *
 * | Param Type | Name     | Description                                                     | Type     | Valid Range                                    | Required |
 * |------------|----------|-----------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | Template   | reduce_type | The type of reduce op - sum, average or maximum              | PoolType | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim  | The dimension of reduce op - row, column or both             | ReduceDim| {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Function   | icb      | The identifier of the circular buffer (CB) containing operand A | uint32_t | 0 to 31                                        | True     |
 * | Function   | icb_scaler  | CB holding scaling factors (see above)                          | uint32_t | 0 to 31                                        | True     |
 * | Function   | itile    | The index of the tile within the first CB                       | uint32_t | Must be less than the size of the CB           | True     |
 * | Function   | itile_sclaer | The index of the tile within the scaling factor CB.             | uint32_t | Must be less than the size of the CB           | True     |
 * | Function   | idst     | The index of the tile in DST REG for the result                 | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_tile(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_sclaer, uint32_t idst) {
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(icb, icb_scaler, idst)));
    UNPACK((llk_unpack_AB(icb, icb_scaler, itile, itile_sclaer)));
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
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(idst, num_faces)));
}

}  // namespace ckernel
