// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/global_config_state.h"
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
 * This version incorporates global configuration state management from Issue #22904. The function now checks the current
 * packer edge mask state and only reconfigures when necessary, eliminating the need for explicit `reduce_uninit` calls.
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
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | reduce_type               | The type of reduce op - sum, average or maximum                                         | PoolType  | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim                | The dimension of reduce op - row, column or both                                        | ReduceDim | {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Template   | enforce_fp32_accumulation | Enable accumulation of reduction in full FP32 precision (Requires DST_ACCUM_MODE==true) | bool      | {true, false}                                  | True     |
 * | Function   | icb                       | The identifier of the circular buffer (CB) containing operand A                         | uint32_t  | 0 to 31                                        | True     |
 * | Function   | icb_scaler                | CB holding scaling factors (see above)                                                  | uint32_t  | 0 to 31                                        | True     |
 * | Function   | ocb                       | The identifier of the output circular buffer (CB)                                       | uint32_t  | 0 to 31                                        | True     |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
ALWI void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb) {
    // Initialize unpacker for reduce operation
    UNPACK((llk_unpack_AB_reduce_init<reduce_dim, BroadcastType::NONE, enforce_fp32_accumulation>(icb, icb_scaler)));

    // Initialize math unit for reduce operation
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, enforce_fp32_accumulation>()));

    // Check if packer edge mask reconfiguration is needed
    constexpr PackEdgeMaskConfig required_mask_config = reduce_dim_to_edge_mask_config<reduce_dim>();

    if (pack_edge_mask_needs_reconfig(required_mask_config)) {
        // Only reconfigure packer edge mask if current state is different
        PACK((llk_pack_reduce_mask_config<false /*untilize*/, reduce_dim>()));

        // Update global state to reflect the new configuration
        set_pack_edge_mask_config_state(required_mask_config);
    }

    // Note: Packer strides remain in contiguous mode for reduce operations
    // Only edge masking changes, so we ensure packer strides are in the correct state
    if (pack_strides_needs_reconfig(PackStridesConfig::PACK_STRIDES_CONTIGUOUS)) {
        // This should be rare since most operations use contiguous strides,
        // but we include it for completeness
        PACK((llk_pack_init(ocb)));
        set_pack_strides_config_state(PackStridesConfig::PACK_STRIDES_CONTIGUOUS);
    }
}

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
 * NOTE: With the new global state management (Issue #22904), there is no longer a need to call `reduce_uninit`
 * before the next operation. The next operation's init function will automatically handle any necessary state transitions.
 *
 * NOTE: For SUM and AVG operations, the value in `icb_scaler` is a scaling factor of the final sum of values across rows/columns/both, so there is no real constraint in terms
 * of it's value. For MAX operation, maximum value will be obtained as expected, but it will be scaled by the values in `icb_scaler`. In any case, it is recommended to use the
 * above-mentioned scaling factors to ensure that operations function as intended. Refer to ISA documentation for more details.
 * NOTE: For other valid ways of populating the `icb_scaler`, refer to the ISA documentation.
 * Return value: None
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | reduce_type               | The type of reduce op - sum, average or maximum                                         | PoolType  | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim                | The dimension of reduce op - row, column or both                                        | ReduceDim | {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Template   | enforce_fp32_accumulation | Enable accumulation of reduction in full FP32 precision (Requires DST_ACCUM_MODE==true) | bool      | {true, false}                                  | True     |
 * | Function   | icb                       | The identifier of the circular buffer (CB) containing operand A                         | uint32_t  | 0 to 31                                        | True     |
 * | Function   | icb_scaler                | CB holding scaling factors (see above)                                                  | uint32_t  | 0 to 31                                        | True     |
 * | Function   | itile                     | The index of the tile within the first CB                                               | uint32_t  | Must be less than the size of the CB           | True     |
 * | Function   | itile_scaler              | The index of the tile within the scaling factor CB.                                     | uint32_t  | Must be less than the size of the CB           | True     |
 * | Function   | idst                      | The index of the tile in DST REG for the result                                         | uint32_t  | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
ALWI void reduce_tile(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst) {
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, false, enforce_fp32_accumulation>(
        icb, icb_scaler, idst)));
    UNPACK((llk_unpack_AB(icb, icb_scaler, itile, itile_scaler)));
}

// clang-format off
/**
 * Performs a math-only reduction operation on a tile in the DST register. Assumes that source tiles are already in source registers.
 *
 * | Param Type | Name                      | Description                                                                             | Type      | Valid Range                                    | Required |
 * |------------|---------------------------|-----------------------------------------------------------------------------------------|-----------|------------------------------------------------|----------|
 * | Template   | reduce_type               | The type of reduce op - sum, average or maximum                                         | PoolType  | {SUM, AVG, MAX}                                | True     |
 * | Template   | reduce_dim                | The dimension of reduce op - row, column or both                                        | ReduceDim | {REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR}        | True     |
 * | Template   | enforce_fp32_accumulation | Enable accumulation of reduction in full FP32 precision (Requires DST_ACCUM_MODE==true) | bool      | {true, false}                                  | True     |
 * | Function   | idst                      | The index of the tile in DST REG for the result                                         | uint32_t  | Must be less than the acquired size of DST REG | True     |
 * | Function   | num_faces                 | Number of faces to reduce (optional, default 4)                                         | uint32_t  | 1 to 4                                         | False    |
 */
// clang-format on
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false>
ALWI void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4) {
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, false, enforce_fp32_accumulation>(
        idst, num_faces)));
}

// clang-format off
/**
 * Legacy function for backward compatibility.
 *
 * DEPRECATED: This function is now a no-op as it is no longer needed with the new global configuration
 * state management (Issue #22904). The function is kept for backward compatibility but does nothing.
 *
 * In the new design, the next operation's init function will automatically handle any necessary
 * state transitions, eliminating the need for explicit uninit calls.
 *
 * NOTE: This function will be completely removed in a future release. Please update your code
 * to remove calls to reduce_uninit().
 *
 * | Param Type | Name | Description                                      | Type | Valid Range | Required |
 * |------------|------|--------------------------------------------------|------|-------------|----------|
 * | Function   | —    | No parameters                                    |  —   |      —      |    —     |
 */
// clang-format on
[[deprecated(
    "reduce_uninit is no longer needed with global config state management. This function will be removed in a future "
    "release.")]]
ALWI void reduce_uninit() {
    // No-op: Global state management handles transitions automatically
    // The next operation's init function will reconfigure as needed
}

}  // namespace ckernel
