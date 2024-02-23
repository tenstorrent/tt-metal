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

#if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)

template<bool at_start>
ALWI void reduce_init(PoolType reduce_op, ReduceDim dim, uint32_t icb, uint32_t icb_scaler, uint32_t ocb = 16)
{
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb, icb_scaler) ));
    UNPACK(( llk_unpack_AB_reduce_init<REDUCE_DIM>(icb, icb_scaler) ));

    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, at_start, false, DST_ACCUM_MODE>(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>() ));
}

ALWI void reduce_init_short(PoolType reduce_op, ReduceDim dim, uint32_t icb, uint32_t icb_scaler, uint32_t ocb = 16) {

    UNPACK(( llk_unpack_AB_reduce_init<REDUCE_DIM>(icb, icb_scaler) ));
    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, false, false, DST_ACCUM_MODE>(ocb) ));
}

// Delta from binary_op_init_common
template<bool at_start>
ALWI void reduce_init_delta(PoolType reduce_op, ReduceDim dim, uint32_t ocb = 16, uint32_t icb0 = 0, uint32_t icb1 = 1)
{
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_reduce_init<REDUCE_DIM>(icb0, icb1) ));

    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));

    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, at_start, false, DST_ACCUM_MODE>(ocb) ));
}

ALWI void reduce_init_delta_no_pack(uint32_t icb0 = 0, uint32_t icb1 = 1)
{
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<>(icb0, icb1) ));

    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
}

ALWI void reduce_init_delta_math()
{
    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
}

ALWI void reduce_revert_delta(uint32_t ocb = 16)
{
    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, false, true, DST_ACCUM_MODE>(ocb) ));
}

/**
 * Performs a reduction operation *B = reduce(A)* using reduce_func for
 * dimension reduction on a tile in the CB at a given index and writes the
 * result to the DST register at index *dst_tile_index*. Reduction can be
 * either of type *Reduce::R*, *Reduce::C* or *Reduce::RC*, identifying the
 * dimension(s) to be reduced in size to 1. The DST register buffer must be in
 * acquired state via *acquire_dst* call.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                     | Type     | Valid Range                                    | Required |
 * |----------------|-----------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | reduce_func    | Enum value, specifying the type of reduce function to perform.  | uint32_t | One of ReduceFunc::Sum, ReduceFunc::Max        | True     |
 * | dim            | Dimension id, identifying the dimension to reduce in size to 1. | uint32_t | One of Reduce::R, Reduce::C, Reduce::RC        | True     |
 * | icb0           | The identifier of the circular buffer (CB) containing A         | uint32_t | 0 to 31                                        | True     |
 * | icb1           | CB for Scaling factor applied to each element of the result.    | uint32_t | 0 to 31                                        | True     |
 * | itile0         | The index of the tile within the first CB                       | uint32_t | Must be less than the size of the CB           | True     |
 * | itile1         | The index of the tile within the scaling factor CB              | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG for the result                 | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
ALWI void reduce_tile(PoolType reduce_op, ReduceDim dim, uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_reduce<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY, DST_ACCUM_MODE>(idst) ));
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1) ));
}

ALWI void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4)
{
    MATH(( llk_math_reduce<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY, DST_ACCUM_MODE>(idst, num_faces) ));
}
#endif




} // namespace ckernel
