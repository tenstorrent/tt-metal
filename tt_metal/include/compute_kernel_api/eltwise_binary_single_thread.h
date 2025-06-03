// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "llk_math_binary_api.h"
#include "llk_unpack_AB_api.h"
#include "llk_unpack_A_api.h"
#include "llk_pack_api.h"

namespace ckernel {

// clang-format off
/**
 * Init function for all binary ops
 * Followed by the specific init required with an opcode (binrary_op_specific_init)
 *
 * | Argument       | Description                                                   | Type     | Valid Range                | Required |
 * |----------------|---------------------------------------------------------------|----------|----------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31                    | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31                    | True     |
 * | ocb            | The identifier of the circular buffer (CB) containing output  | uint32_t | 0 to 31, defaults to CB 16 | True     |
 */
// clang-format on
ALWI void binary_op_init_common_(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1)));
    // UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));

    UNPACK((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    UNPACK((llk_math_hw_configure_disaggregated(icb0, icb1)));

    UNPACK((llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(ocb)));
    UNPACK((llk_pack_init_st(ocb)));
    UNPACK((llk_pack_dest_init<false, DST_ACCUM_MODE>()));
}

// clang-format off
 /**
 * Template for initializing element-wise binary operations.
 * Template parameters:
 * full_init: if true, the full init is performed (unpack+math), otherwise only math init is performed
 * eltwise_binary_type: the binary operation type
 *
 * Function
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A [+,-,x] B + dst_tile_idx of *_tiles, depending on the eltwise_binary_type | bool | 0,1  | False |
 */
// clang-format on
template <bool full_init, EltwiseBinaryType eltwise_binary_type>
ALWI void binary_tiles_init_(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false) {
    UNPACK((llk_unpack_AB_init_st(icb0, icb1, 0 /*transpose*/, acc_to_dest)));
}

// clang-format off
/**
 * Performs element-wise addition C=A+B of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
ALWI void add_tiles_(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    // This will do the task of both the unpacker and math.
    // We can configure only one mop per thread and even if we could configure two mops
    // And launch them in sequence, unpacker's mop would wait for clear signal from
    // the math mop but math mop would not execute because it would trail the unpacker
    // mop in the instruction pipeline, leading to a deadlock.
    UNPACK((llk_unpack_AB_st(icb0, icb1, itile0, itile1, idst)));
}

}  // namespace ckernel
