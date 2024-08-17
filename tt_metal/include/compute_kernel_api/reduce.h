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

template<bool at_start, PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb = 16)
{
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb, icb_scaler) ));
    UNPACK(( llk_unpack_AB_reduce_init<reduce_dim>(icb, icb_scaler) ));

    MATH(( llk_math_reduce_init<reduce_type, reduce_dim, MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>() ));
    MATH(( llk_math_hw_configure_disaggregated() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_reduce_config_v2<reduce_dim, at_start, false, DST_ACCUM_MODE>(ocb) ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>() ));
}

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_init_short(uint32_t icb, uint32_t icb_scaler, uint32_t ocb = 16) {

    UNPACK(( llk_unpack_AB_reduce_init<reduce_dim>(icb, icb_scaler) ));
    MATH(( llk_math_reduce_init<reduce_type, reduce_dim, MATH_FIDELITY>() ));
    PACK(( llk_pack_reduce_config_v2<reduce_dim, false, false, DST_ACCUM_MODE>(ocb) ));
}

template<bool at_start, PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_init_delta(uint32_t ocb = 16, uint32_t icb0 = 0, uint32_t icb1 = 1)
{
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_reduce_init<reduce_dim>(icb0, icb1) ));

    MATH(( llk_math_reduce_init<reduce_type, reduce_dim, MATH_FIDELITY>() ));

    PACK(( llk_pack_reduce_config_v2<reduce_dim, at_start, false, DST_ACCUM_MODE>(ocb) ));
}

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_init_delta_no_pack(uint32_t icb0 = 0, uint32_t icb1 = 1)
{
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<>(icb0, icb1) ));

    MATH(( llk_math_reduce_init<reduce_type, reduce_dim, MATH_FIDELITY>() ));
}

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_init_delta_math()
{
    MATH(( llk_math_reduce_init<reduce_type, reduce_dim, MATH_FIDELITY>() ));
}

template<ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_revert_delta(uint32_t ocb = 16)
{
    PACK(( llk_pack_reduce_config_v2<reduce_dim, false, true, DST_ACCUM_MODE>(ocb) ));
}

/**
 * Performs a reduction operation *B = reduce(A)* using reduce_func for
 * dimension reduction on a tile in the CB at a given index and writes the
 * result to the DST register at index *dst_tile_index*. Reduction can be
 * either of type *Reduce::R*, *Reduce::C* or *Reduce::RC*, identifying the
 * dimension(s) to be reduced in size to 1. The DST register buffer must be in
 * acquired state via *acquire_dst* call.
 *
 * The templates takes reduce_type which can be ReduceFunc::Sum, ReduceFunc::Max
 * and reduce_dim which can be Reduce::R, Reduce::C, Reduce::RC.
 * They can also be specified by defines REDUCE_OP and REDUCE_DIM.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                     | Type     | Valid Range                                    | Required |
 * |----------------|-----------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A         | uint32_t | 0 to 31                                        | True     |
 * | icb1           | CB for Scaling factor applied to each element of the result.    | uint32_t | 0 to 31                                        | True     |
 * | itile0         | The index of the tile within the first CB                       | uint32_t | Must be less than the size of the CB           | True     |
 * | itile1         | The index of the tile within the scaling factor CB              | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG for the result                 | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
 template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_tile(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_reduce<reduce_type, reduce_dim, MATH_FIDELITY, DST_ACCUM_MODE>(idst) ));
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1) ));
}

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4)
{
    MATH(( llk_math_reduce<reduce_type, reduce_dim, MATH_FIDELITY, DST_ACCUM_MODE>(idst, num_faces) ));
}

} // namespace ckernel
