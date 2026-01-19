// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#include "llk_unpack_A_api.h"
#endif

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
ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
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
ALWI void binary_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false) {
    MATH((
        llk_math_eltwise_binary_init_with_operands<eltwise_binary_type, NONE, MATH_FIDELITY>(icb0, icb1, acc_to_dest)));

    if constexpr (full_init) {
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, 0 /*transpose*/)));
    }
}

// clang-format off
/**
 * Short init function
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void mul_tiles_init(uint32_t icb0, uint32_t icb1) { binary_tiles_init<true, ELWMUL>(icb0, icb1); }

// clang-format off
/**
 * Short init function
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A + B + dst_tile_idx of add_tiles | bool     | 0,1         | False |
 */
// clang-format on
ALWI void add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false) {
    binary_tiles_init<true, ELWADD>(icb0, icb1, acc_to_dest);
}

// clang-format off
/**
 * Short init function
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31     | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | If true, operation = A - B + dst_tile_idx of sub_tiles | bool     | 0,1         | False |
 */
// clang-format on
ALWI void sub_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false) {
    binary_tiles_init<true, ELWSUB>(icb0, icb1, acc_to_dest);
}

// clang-format off
/**
 * Performs element-wise multiplication C=A*B of tiles in two CBs at given
 * indices and writes the result to the DST register at index dst_tile_index.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
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
ALWI void mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    // static bool first = true; // TODO(AP): static initializer causes a hang, possibly investigate
    // if (first)
    //  one possible solution is to add a local context in the kernel, pass it around and store init flags in it
    //  this way the compiler should be able to perform loop hoisting optimization
    //  - might need to add __attribute__((pure)) to init calls for this to work
    //  Also pass -fmove-loop-invariants to g++
    // mul_tiles_initf();
    // first = false;

    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<ELWMUL, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, true)));
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
ALWI void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, true)));
}

// clang-format off
/**
 * Performs element-wise subtraction C=A-B of tiles in two CBs at given indices
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
ALWI void sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<ELWSUB, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, true)));
}

/**
 * Please refer to documentation for any_init.
 */
template <
    EltwiseBinaryType eltwise_binary_type = ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles_init(uint32_t icb0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, binary_reuse_dest>(false, false, icb0)));
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, NONE, MATH_FIDELITY, binary_reuse_dest>(false)));
}

// clang-format off
/**
 * Performs element-wise binary operations, such as multiply, add, or sub of tiles.
 * If binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA, then the tile specified by idst will be loaded from
 * the DST register buffer into SRCA. The binary operation will operate on SRCA & SRCB inputs, and the result will be
 * written back to the DST register buffer specified by idst. Similar to DEST_TO_SRCA, if binary_reuse_dest =
 * EltwiseBinaryReuseDestType::DEST_TO_SRCB, then tile specified by idst will be loaded from the DST into SRCB register
 * buffer.
 *
 * EltwiseBinaryReuseDestType::DEST_TO_SRCA and EltwiseBinaryReuseDestType::DEST_TO_SRCB assume that another operation has
 * populated the dest register, otherwise dest will contain zeroes.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                                               | Type     | Valid Range                                    | Required |
 * |----------------|-----------------------------------------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A                                                   | uint32_t | 0 to 31                                        | True     |
 * | in_tile_index  | The index of tile A within the first CB                                                                   | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of tile B that will be moved to Src reg, and the index of the tile in DST REG for the result C  | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = ELWADD,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, true, binary_reuse_dest>(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_binary<eltwise_binary_type, NONE, DST_ACCUM_MODE, MATH_FIDELITY, binary_reuse_dest>(
        in_cb_id, in_cb_id, dst_tile_index, true)));
}

}  // namespace ckernel
