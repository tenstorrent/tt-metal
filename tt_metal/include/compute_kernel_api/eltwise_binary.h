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

/**
 * Init function for all binary ops
 * Followed by the specific init required with an opcode (binrary_op_specific_init)
 * | Argument       | Description                                                   | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31                                        | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31                                        | True     |
 * | ocb            | The identifier of the circular buffer (CB) containing output  | uint32_t | 0 to 31, defaults to CB 16                     | True     |
 */
ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb=16)
{
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1) ));

    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>() ));
    MATH(( llk_math_hw_configure_disaggregated(icb0, icb1) ));

    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(ocb) ));
    PACK(( llk_pack_init(ocb) ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>() ));
}


/**
 * Please refer to documentation for any_init.
 * f means high fidelity with resepect to accuracy
 * this is set during createprogram
 */
ALWI void mul_tiles_init_f() { MATH(( llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>() )); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void mul_tiles_init(uint32_t icb0 = 0, uint32_t icb1 = 1) {
    MATH(( llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1) ));
}

/**
 * Please refer to documentation for any_init.
 * nof means low fidelity with resepect to accuracy
 * this is set during createprogram
 */
ALWI void add_tiles_init_nof() { MATH(( llk_math_eltwise_binary_init<ELWADD, NONE>() )); }


/**
 * Short init function
 * | Argument       | Description                                                   | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31                                        | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31                                        | True     |
 * | acc_to_dest    | If true, operation = A + B + dst_tile_idx of add_tiles        | bool     | 0,1                                            | False    |
 */
ALWI void add_tiles_init(uint32_t icb0 = 0, uint32_t icb1 = 1, bool acc_to_dest = false) {
    MATH(( llk_math_eltwise_binary_init<ELWADD, NONE>(0/*transpose*/, acc_to_dest) ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, 0/*transpose*/, acc_to_dest) ));
}

/**
 * Please refer to documentation for any_init.
 * nof means low fidelity with respect to accuracy
 * this is set during createprogram
 */
ALWI void sub_tiles_init_nof() { MATH(( llk_math_eltwise_binary_init<ELWSUB, NONE>() )); }


/**
 * Short init function
 * | Argument       | Description                                                   | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t | 0 to 31                                        | True     |
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t | 0 to 31                                        | True     |
 * | acc_to_dest    | If true, operation = A - B + dst_tile_idx of sub_tiles        | bool     | 0,1                                            | False    |
 */
ALWI void sub_tiles_init(uint32_t icb0 = 0, uint32_t icb1 = 1, bool acc_to_dest = false) {
    MATH(( llk_math_eltwise_binary_init<ELWSUB, NONE>(0/*transpose*/, acc_to_dest) ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, 0/*transpose*/, acc_to_dest) ));
}


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
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void mul_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    //static bool first = true; // TODO(AP): static initializer causes a hang, possibly investigate
    //if (first)
    // one possible solution is to add a local context in the kernel, pass it around and store init flags in it
    // this way the compiler should be able to perform loop hoisting optimization
    // - might need to add __attribute__((pure)) to init calls for this to work
    // Also pass -fmove-loop-invariants to g++
    //mul_tiles_initf();
    //first = false;

    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));
    MATH(( llk_math_eltwise_binary<ELWMUL, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE, DST_ACCUM_MODE>(icb0, icb1, idst) ));
}

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
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void add_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));
    MATH(( llk_math_eltwise_binary<ELWADD, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE, DST_ACCUM_MODE>(icb0, icb1, idst) ));
}

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
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void sub_tiles( uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1)  ));
    MATH(( llk_math_eltwise_binary<ELWSUB, NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE, DST_ACCUM_MODE>(icb0, icb1, idst) ));
}

/**
 * Init function with a specified op
 * template parameters:
 * full_init: if true, the full init is performed (unpack+math), otherwise a nof init is performed (only math)
 * eltwise_binary_op_type: the binary operation type
 */
template<bool full_init = false, EltwiseBinaryType eltwise_binary_op_type = ELWADD>
ALWI void binary_op_specific_init() // TODO(AP): better naming
{
    if constexpr (full_init) {
        if constexpr (eltwise_binary_op_type == ELWADD) {
            add_tiles_init();
        } else if constexpr (eltwise_binary_op_type == ELWSUB) {
            sub_tiles_init();
        } else if constexpr (eltwise_binary_op_type == ELWMUL) {
            mul_tiles_init();
        }
    } else {
        if constexpr (eltwise_binary_op_type == ELWADD) {
            add_tiles_init_nof();
        } else if constexpr (eltwise_binary_op_type == ELWSUB) {
            sub_tiles_init_nof();
        } else if constexpr (eltwise_binary_op_type == ELWMUL) {
            mul_tiles_init_f();
        }
    }
}

/**
 * Please refer to documentation for any_init.
 */
template<
EltwiseBinaryType eltwise_binary_type = ELWADD,
EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles_init(uint32_t icb0) {
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, binary_reuse_dest>(false, false, icb0) ));
    MATH(( llk_math_eltwise_binary_init<eltwise_binary_type, NONE, MATH_FIDELITY, binary_reuse_dest>(false, false) ));
}


/**
 * Performs element-wise binary operations, such as multiply, add, or sub of tiles.
 * If binary_reuse_dest = EltwiseBinaryReuseDestType::DST_TO_SRCA, then the tile specified by idst will be loaded from the DST register buffer
 * into SRCA. The binary operation will operate on SRCA & SRCB inputs, and the result will be written back to the DST register buffer specified by idst.
 * Similar to DST_TO_SRCA, if binary_reuse_dest = EltwiseBinaryReuseDestType::DST_TO_SRCB, then tile specified by idst will be loaded from the DST
 * into SRCB register buffer. DST_TO_SRCB feature is not available for Grayskull, only Wormhole.
 *
 * EltwiseBinaryReuseDestType::DST_TO_SRCA and EltwiseBinaryReuseDestType::DST_TO_SRCB assume that another operation has populated
 * the dest register, otherwise dest will contain zeroes.
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
template<
EltwiseBinaryType eltwise_binary_type = ELWADD,
EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles( uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)
{
    UNPACK(( llk_unpack_A<BroadcastType::NONE, true, binary_reuse_dest>(in_cb_id, in_tile_index)  ));
    MATH(( llk_math_eltwise_binary<eltwise_binary_type, NONE, MATH_FIDELITY, binary_reuse_dest, DST_ACCUM_MODE>(in_tile_index, in_tile_index, dst_tile_index) ));
}


} // namespace ckernel
