// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_binary_api.h"
#include "llk_math_matmul_api.h"
#include "llk_math_common.h"
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_api.h"
#include "llk_unpack_A_api.h"
#include "llk_unpack_common_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack.h"
#include "llk_pack_common.h"
#endif

namespace ckernel {

template <BroadcastType bcast_type>
ALWI void unary_bcast_init(uint32_t icb, uint32_t ocb) {
    // 32bit formats are implemented using unpack to dest, since SrcB is only 19bits wide
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);

    // Will configure A & B in similar way
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb)));

    if (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A_init<bcast_type, false, EltwiseBinaryReuseDestType::NONE, true>(
            false, false /*transpose within 16x16 face*/, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, bcast_type>(icb)));
    } else {
        UNPACK((llk_unpack_A_init<bcast_type, false, EltwiseBinaryReuseDestType::NONE, false>(
            false, false /*transpose within 16x16 face*/, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<B2D, DST_ACCUM_MODE, bcast_type>(icb)));
    }
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
#endif

    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_init<false>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}

template <BroadcastType bcast_type>
ALWI void unary_bcast(uint32_t icb, uint32_t in_tile_index, uint32_t dst_tile_index) {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    // 32bit formats are implemented using unpack to dest, since SrcB is only 19bits wide
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);

    if (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A<bcast_type, false, EltwiseBinaryReuseDestType::NONE, true>(icb, in_tile_index)));
        MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, bcast_type, true>(dst_tile_index, icb)));
    } else {
        UNPACK((llk_unpack_A<bcast_type, false, EltwiseBinaryReuseDestType::NONE, false>(icb, in_tile_index)));
        MATH((llk_math_eltwise_unary_datacopy<B2D, DST_ACCUM_MODE, bcast_type, false>(dst_tile_index, icb)));
    }
#endif
}

template <BroadcastType bcast_type>
ALWI void unary_bcast_uninit(uint32_t icb) {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);

    UNPACK((llk_unpack_A_uninit<bcast_type>(icb)));

    if (enable_unpack_to_dest) {
        MATH((llk_math_eltwise_unary_datacopy_uninit<bcast_type, true>()));
    } else {
        MATH((llk_math_eltwise_unary_datacopy_uninit<bcast_type, false>()));
    }
#endif
}

template <BroadcastType old_bcast_type, BroadcastType new_bcast_type>
void reconfigure_unary_bcast(uint32_t old_icb, uint32_t new_icb, uint32_t old_ocb, uint32_t new_ocb) {
#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    // Pass through uses A2D and potentially direct unpack to dest.
    const auto data_copy_type = (new_bcast_type == BroadcastType::NONE) ? A2D : B2D;
    const bool enable_unpack_to_dest = data_copy_type == A2D;
    const std::uint32_t new_operand_id = get_operand_id(new_icb);
    const std::uint32_t old_operand_id = get_operand_id(old_icb);
    bool unpacker_src_format_change = unpack_src_format[new_operand_id] != unpack_src_format[old_operand_id];
    bool unpacker_dst_format_change = unpack_dst_format[new_operand_id] != unpack_dst_format[old_operand_id];
    bool bcast_type_change = (old_bcast_type != new_bcast_type);

    if (unpacker_src_format_change || unpacker_dst_format_change) {
        // Will configure A & B in similar way
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(new_icb)));
    }

    if (unpacker_src_format_change || unpacker_dst_format_change || bcast_type_change) {
        UNPACK((llk_unpack_A_init<new_bcast_type, false, EltwiseBinaryReuseDestType::NONE, enable_unpack_to_dest>(
            false, false /*transpose within 16x16 face*/, new_icb)));
    }

    if (unpacker_dst_format_change) {
        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(new_icb, new_icb)));
    }

    if (unpacker_dst_format_change || bcast_type_change) {
        MATH((llk_math_eltwise_unary_datacopy_init<data_copy_type, DST_ACCUM_MODE, new_bcast_type>(new_icb)));
    }
#endif

    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(old_ocb, new_ocb)));
}

/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
ALWI void sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::COL,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}

/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
ALWI void sub_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::SCALAR,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
ALWI void mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::COL,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
ALWI void mul_tiles_bcast_rows(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1, bcast_row_idx)));
}

/**
 * Please refer to documentation for sub_tiles_bcast
 */
ALWI void add_tiles_bcast_rows(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1, bcast_row_idx)));
}

/**
 * Please refer to documentation for sub_tiles_bcast
 */
ALWI void add_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::COL,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}

/**
 * Please refer to documentation for add_tiles_bcast
 */
ALWI void add_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::SCALAR,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

// clang-format off
/**
 * Associated init function that must be called before calling a bcast op.
 *
 * Return value: None
 *
 *
 * | Argument       | Description                                                   | Type          | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|---------------|-------------|----------|
 * | icb0           | The identifier of the circular buffer (CB) containing A       | uint32_t      | 0 to 31     | True     |
 * | icb1           | The indentifier of the circular buffer (CB) containing B      | uint32_t      | 0 to 31     | True     |
 * | ocb            | The indentifier of the circular buffer (CB) containing output | uint32_t      | 0 to 31     | False    |
 */
 // clang-format on
template <EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
void init_bcast(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    if constexpr (tBcastOp == ELWMUL) {
        MATH((llk_math_eltwise_binary_init<tBcastOp, tBcastDim, MATH_FIDELITY>()));
    } else {
        MATH((llk_math_eltwise_binary_init<tBcastOp, tBcastDim>()));
    }

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<tBcastDim>(icb0, icb1)));

    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
}

/*
Internal helper function for all broadcast ops
*/
template <EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
ALWI void any_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    MATH((llk_math_eltwise_binary<tBcastOp, tBcastDim, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<tBcastDim>(icb0, icb1, itile0, itile1, bcast_row_idx)));
}

// clang-format off
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
// clang-format on
template <BroadcastType tBcastDim>
ALWI void add_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWADD, tBcastDim>(icb0, icb1, itile0, itile1, idst, bcast_row_idx);
}

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template <BroadcastType tBcastDim>
ALWI void sub_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWSUB, tBcastDim>(icb0, icb1, itile0, itile1, idst, bcast_row_idx);
}

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template <BroadcastType tBcastDim>
ALWI void mul_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWMUL, tBcastDim>(icb0, icb1, itile0, itile1, idst, bcast_row_idx);
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_rows to be executed
 * correctly. Required to be called before add_tiles_bcast if using column as broadcast type
 */
ALWI void add_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) {
    MATH((llk_math_eltwise_binary_init_with_operands<ELWADD, BroadcastType::ROW, MATH_FIDELITY>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_cols to be executed
 * correctly. Required to be called before add_tiles_bcast if using column as broadcast type
 */
ALWI void add_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) {
    MATH((llk_math_eltwise_binary_init<ELWADD, BroadcastType::COL>()));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_scalar to be
 * executed correctly.
 */
ALWI void add_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1) {
    MATH((llk_math_eltwise_binary_init<ELWADD, BroadcastType::SCALAR, MATH_FIDELITY>()));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed
 * correctly.
 */
ALWI void mul_tiles_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1) {
    MATH((llk_math_eltwise_binary_init_with_operands<ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>(icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

/**
 * Performs a broadcast-multiply of a tile from icb0[itile0] with a scalar encoded as a tile from icb1[itile1].
 */
ALWI void mul_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          ELWMUL,
          BroadcastType::SCALAR,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed
 * correctly.
 */
ALWI void mul_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) {
    MATH((llk_math_eltwise_binary_init<ELWMUL, BroadcastType::COL, MATH_FIDELITY>()));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1)));
}

/**
 * Performs a switch-from-another-op tile hw reconfiguration step needed for mul_bcast_rows to be executed correctly.
 */
ALWI void mul_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) {
    MATH((llk_math_eltwise_binary_init<ELWMUL, BroadcastType::ROW, MATH_FIDELITY>()));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_bcast_cols to be executed
 * correctly.
 */
ALWI void sub_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) {
    MATH((llk_math_eltwise_binary_init_with_operands<ELWSUB, BroadcastType::COL, MATH_FIDELITY>(icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_tiles_bcast_scalar to be
 * executed correctly.
 */
ALWI void sub_tiles_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1) {
    MATH((llk_math_eltwise_binary_init<ELWSUB, BroadcastType::SCALAR, MATH_FIDELITY>()));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

}  // namespace ckernel
