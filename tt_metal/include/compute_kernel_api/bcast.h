#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif
#include "compute_kernel_api/llk_pack_includes.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_AB.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif


namespace ckernel {

/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
ALWI void sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWSUB, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
ALWI void mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
ALWI void mul_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1) ));
}

/**
 * Please refer to documentation for sub_tiles_bcast
 */
ALWI void add_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::ROW, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1) ));
}

/**
 * Please refer to documentation for sub_tiles_bcast
 */
ALWI void add_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<EltwiseBinaryType::ELWADD, BroadcastType::COL, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1) ));
}


template<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
void init_bcast(uint32_t icb0, uint32_t icb1)
{
    if constexpr (tBcastOp == ELWMUL)
        MATH(( llk_math_eltwise_binary_init<tBcastOp, tBcastDim, MATH_FIDELITY>() ));
    else
        MATH(( llk_math_eltwise_binary_init<tBcastOp, tBcastDim>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_init<tBcastDim>() )); // TODO(AP): move out init
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<tBcastDim>(icb0, icb1) ));
    // TODO(AP): running this specific init after common AB init causes a hang

    // clone of general init for AB TODO(AP): commonize
    //UNPACK(( llk_setup_operands() ));
    //UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
    //UNPACK(( llk_unpack_AB_hw_configure_disaggregated<BroadcastType::NONE>(icb0, icb1) ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    MATH(( llk_math_pack_sync_init<SyncHalf>() ));
}

template<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
ALWI void any_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<tBcastOp, tBcastDim, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<tBcastDim>(icb0, icb1, itile0, itile1) ));
}

/**
 * This documentation applies to either one of the 3 broadcast operation variants -
 * *add_tiles_bcast*, *sub_tiles_bcast* and *mul_tiles_bcast*.
 *
 * The description below describes *add_tiles_bcast*, the other 2 operations
 * use the same definition with the corresponding substitution of the math
 * operator.
 *
 * Performs a broadcast-operation *C=A+B* of tiles in two CBs at given indices
 * and writes the result to the DST register at index dst_tile_index. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call
 * is blocking and is only available on the compute engine.
 *
 * Broadcasting semantics are defined as follows:
 *
 * For *dim==BroadcastType::COL*, the input in *B* is expected to be a single tile with a
 * filled 0-column and zeros elsewhere.  The result is *C[h, w] = A[h,w] + B[w]*
 *
 * For *dim==Dim::C*, the input in *B* is expected to be a single tile with a
 * filled 0-row, and zeros elsewhere.  The result is *C[h, w] = A[h,w] + B[h]*
 *
 * For *dim==Dim::RC*, the input in *B* is expected to be a single tile with a
 * filled single value at location [0,0], and zeros elsewhere.  The result is
 * *C[h, w] = A[h,w] + B[0,0]*
 *
 * Return value: None
 *
 * DOX-TODO(AP): verify that the bcast tile is actually required to be filled
 * with zeros.
 *
 * | Argument       | Description                                              | Type          | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|---------------|------------------------------------------------|----------|
 * | tBcastDim      | Broadcast dimension                                      | BroadcastType | One of Dim::R, Dim::C, Dim::RC.                | True     |
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t      | 0 to 31                                        | True     |
 * | in1_cb_id      | The indentifier of the circular buffer (CB) containing B | uint32_t      | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t      | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t      | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
template<BroadcastType tBcastDim>
ALWI void add_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWADD, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template<BroadcastType tBcastDim>
ALWI void sub_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWSUB, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template<BroadcastType tBcastDim>
ALWI void mul_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{ any_tiles_bcast<EltwiseBinaryType::ELWMUL, tBcastDim>(icb0, icb1, itile0, itile1, idst); }

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_rows to be executed correctly.
 */
ALWI void add_bcast_rows_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::ROW>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_cols to be executed correctly.
 */
ALWI void add_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWADD, BroadcastType::COL>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed correctly.
 */
ALWI void mul_tiles_bcast_scalar_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>() )); // TODO(AP)
    UNPACK(( llk_unpack_AB_init<BroadcastType::SCALAR>() ));
}

/**
 * Performs a broadcast-multiply of a tile from icb0[itile0] with a scalar encoded as a tile from icb1[itile1].
 */
ALWI void mul_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_eltwise_binary<ELWMUL, BroadcastType::SCALAR, SyncHalf, MATH_FIDELITY, false>(idst) ));
    UNPACK(( llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1) ));
}


/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed correctly.
 */
ALWI void mul_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::COL, MATH_FIDELITY>() )); // TODO(AP)
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
}

/**
 * Performs a switch-from-another-op tile hw reconfiguration step needed for mul_bcast_rows to be executed correctly.
 */
ALWI void mul_bcast_rows_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWMUL, BroadcastType::ROW, MATH_FIDELITY>() ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_bcast_cols to be executed correctly.
 */
ALWI void sub_bcast_cols_init_short()
{
    MATH(( llk_math_eltwise_binary_init<ELWSUB, BroadcastType::COL, MATH_FIDELITY>() )); // TODO(AP)
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>() ));
}

} // namespace ckernel
