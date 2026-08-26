// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "llk_assert.h"
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
#endif
#ifdef TRISC_PACK
#include "llk_pack.h"
#include "llk_pack_common.h"
#endif

namespace ckernel {

// BroadcastType::NONE is a pass through: llk_unpack_A leaves the tile in SrcA and never raises a SrcB
// data valid (Tensix only zero-fills SrcB), so the math thread must read it back with A2D. Using B2D
// would copy zeros, wait on a SrcB data valid that never arrives and never clear SrcA's data valid,
// hanging the unpacker on the next tile. The broadcast modes leave the tile in SrcB, so they use B2D.
template <BroadcastType bcast_type>
constexpr DataCopyType unary_bcast_data_copy_type =
    (bcast_type == BroadcastType::NONE) ? DataCopyType::A2D : DataCopyType::B2D;

template <BroadcastType bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_bcast_init(uint32_t icb) {
    // NOTE: no call_line parameter here — a defaulted call_line would make this 1-arg overload
    // ambiguous with the [[deprecated]] (icb, ocb) full init below. The sentinel still tracks the
    // operand; only the source line for this specific call is attributed to bcast.h.
    state_configure(icb, __builtin_LINE());

#ifndef ARCH_QUASAR
    // 32bit formats are implemented using unpack to dest, since SrcB is only 19bits wide
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);

    if (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A_init<bcast_type, false, EltwiseBinaryReuseDestType::NONE, true>(
            false, false /*transpose within 16x16 face*/, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, is_fp32_dest_acc_en, bcast_type>(icb)));
    } else {
        UNPACK((llk_unpack_A_init<bcast_type, false, EltwiseBinaryReuseDestType::NONE, false>(
            false, false /*transpose within 16x16 face*/, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<unary_bcast_data_copy_type<bcast_type>, is_fp32_dest_acc_en, bcast_type>(
            icb)));
    }
#endif
#else
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    // 32bit formats require the A2D unpack-to-dest path (SrcB is only 19 bits wide), which is not
    // implemented for Quasar yet; only the B2D path is supported here.
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest =
        (dst_format == (std::uint32_t)DataFormat::Float32) || (dst_format == (std::uint32_t)DataFormat::Int32);
    LLK_ASSERT(!enable_unpack_to_dest, "32-bit unary broadcast (unpack-to-dest) not supported on Quasar");
    UNPACK((llk_unpack_A_init<
            bcast_type,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            false /*unpack_to_dest*/>(false /*transpose_of_faces*/, false /*within_face_16x16_transpose*/, icb)));
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::B2D, false /*EN_32BIT_DEST*/, bcast_type>(icb)));
#endif
#endif
}

// Deprecated full init: fused hardware startup + op-specific short init. Superseded by the
// compute_kernel_hw_startup(icb, ocb) + unary_bcast_init(icb) programming model, mirroring the
// matmul (#46346) / transpose (#23835) / eltwise (#22943) cleanups under umbrella #22219.
template <BroadcastType bcast_type>
[[deprecated(
    "Use compute_kernel_hw_startup(icb, ocb) once at the top of the kernel, then unary_bcast_init(icb). "
    "The unary_bcast_init(icb, ocb) full init will be removed after September 15th, 2026 (tt-metal#49924).")]]
ALWI void unary_bcast_init(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);
    compute_kernel_hw_startup(icb, ocb);
    unary_bcast_init<bcast_type>(icb);
}

template <BroadcastType bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_bcast(uint32_t icb, uint32_t in_tile_index, uint32_t dst_tile_index) {
#ifndef ARCH_QUASAR
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    // 32bit formats are implemented using unpack to dest, since SrcB is only 19bits wide
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);

    if (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A<bcast_type, false, EltwiseBinaryReuseDestType::NONE, true>(icb, in_tile_index)));
        MATH((
            llk_math_eltwise_unary_datacopy<DataCopyType::A2D, is_fp32_dest_acc_en, bcast_type, true>(dst_tile_index, icb)));
    } else {
        UNPACK((llk_unpack_A<bcast_type, false, EltwiseBinaryReuseDestType::NONE, false>(icb, in_tile_index)));
        MATH((llk_math_eltwise_unary_datacopy<
              unary_bcast_data_copy_type<bcast_type>,
              is_fp32_dest_acc_en,
              bcast_type,
              false>(dst_tile_index, icb)));
    }
#endif
#else
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    // Broadcast mode and B2D vs A2D are fixed in unary_bcast_init; pass logical operand ids through to LLK.
    // 32bit formats would require the A2D unpack-to-dest path (SrcB is only 19 bits wide), which is not
    // implemented for Quasar yet; only the B2D path is supported here.
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest =
        (dst_format == (std::uint32_t)DataFormat::Float32) || (dst_format == (std::uint32_t)DataFormat::Int32);
    LLK_ASSERT(!enable_unpack_to_dest, "32-bit unary broadcast (unpack-to-dest) not supported on Quasar");
    UNPACK((llk_unpack_A<bcast_type, false, EltwiseBinaryReuseDestType::NONE, false>(icb, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::B2D, false, bcast_type, false>(dst_tile_index, icb)));
#endif
#endif
}

template <BroadcastType bcast_type>
ALWI void unary_bcast_uninit(uint32_t icb) {
#ifndef ARCH_QUASAR
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);

    UNPACK((llk_unpack_A_uninit<bcast_type>()));

    if (enable_unpack_to_dest) {
        MATH((llk_math_eltwise_unary_datacopy_uninit<bcast_type, true>()));
    } else {
        MATH((llk_math_eltwise_unary_datacopy_uninit<bcast_type, false>()));
    }
#endif
#else
    UNPACK((llk_unpack_A_uninit<bcast_type>()));
    // Quasar's llk_math_eltwise_unary_datacopy_uninit is a no-op for both unpack_to_dest values,
    // so no runtime dispatch on dst format is needed here.
    MATH((llk_math_eltwise_unary_datacopy_uninit<bcast_type>()));
#endif
}

#ifndef ARCH_QUASAR
template <BroadcastType old_bcast_type, BroadcastType new_bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
[[deprecated(
    "Switch broadcast operands with the generic reconfig_data_format_srca / reconfig_data_format_srcb + "
    "pack_reconfig_data_format, then unary_bcast_init(new_icb). This will be removed after September 15th, "
    "2026.")]] void
reconfigure_unary_bcast(uint32_t old_icb, uint32_t new_icb, uint32_t old_ocb, uint32_t new_ocb) {
#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    // Pass through uses A2D and potentially direct unpack to dest.
    constexpr DataCopyType data_copy_type = unary_bcast_data_copy_type<new_bcast_type>;
    constexpr bool enable_unpack_to_dest = (data_copy_type == DataCopyType::A2D);
    const std::uint32_t new_operand_id = get_operand_id(new_icb);
    const std::uint32_t old_operand_id = get_operand_id(old_icb);
    bool unpacker_src_format_change = unpack_src_format[new_operand_id] != unpack_src_format[old_operand_id];
    bool unpacker_dst_format_change = unpack_dst_format[new_operand_id] != unpack_dst_format[old_operand_id];
    bool bcast_type_change = (old_bcast_type != new_bcast_type);

    if (unpacker_src_format_change || unpacker_dst_format_change) {
        // Will configure A & B in similar way
        UNPACK((llk_unpack_hw_configure<is_fp32_dest_acc_en>(new_icb)));
    }

    if (unpacker_src_format_change || unpacker_dst_format_change || bcast_type_change) {
        UNPACK((llk_unpack_A_init<new_bcast_type, false, EltwiseBinaryReuseDestType::NONE, enable_unpack_to_dest>(
            false, false /*transpose within 16x16 face*/, new_icb)));
    }

    if (unpacker_dst_format_change) {
        MATH((llk_math_hw_configure<is_fp32_dest_acc_en>(new_icb, new_icb)));
    }

    if (unpacker_dst_format_change || bcast_type_change) {
        MATH((llk_math_eltwise_unary_datacopy_init<data_copy_type, is_fp32_dest_acc_en, new_bcast_type>(new_icb)));
    }
#endif

    PACK((llk_pack_reconfig_data_format<is_fp32_dest_acc_en>(old_ocb, new_ocb)));
}
#endif

/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::COL,
          is_fp32_dest_acc_en,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}

/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::SCALAR,
          is_fp32_dest_acc_en,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::COL,
          is_fp32_dest_acc_en,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}

/**
 * Shorthand template instantiation of mul_tiles_bcast.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_tiles_bcast_rows(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
#ifdef ARCH_QUASAR
    LLK_ASSERT(bcast_row_idx == 0, "non-default bcast_row_idx not supported on Quasar");
#endif
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::ROW,
          is_fp32_dest_acc_en,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1, bcast_row_idx)));
}

/**
 * Please refer to documentation for add_tiles_bcast
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_tiles_bcast_rows(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
#ifdef ARCH_QUASAR
    LLK_ASSERT(bcast_row_idx == 0, "non-default bcast_row_idx not supported on Quasar");
#endif
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::ROW,
          is_fp32_dest_acc_en,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1, bcast_row_idx)));
}

/**
 * Shorthand template instantiation of sub_tiles_bcast.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_tiles_bcast_rows(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
#ifdef ARCH_QUASAR
    LLK_ASSERT(bcast_row_idx == 0, "non-default bcast_row_idx not supported on Quasar");
#endif
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::ROW,
          is_fp32_dest_acc_en,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(icb0, icb1, itile0, itile1, bcast_row_idx)));
}

/**
 * Please refer to documentation for add_tiles_bcast
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::COL,
          is_fp32_dest_acc_en,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}

/**
 * Please refer to documentation for add_tiles_bcast
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::SCALAR,
          is_fp32_dest_acc_en,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
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
 * | icb1           | The identifier of the circular buffer (CB) containing B       | uint32_t      | 0 to 31     | True     |
 * | ocb            | The identifier of the circular buffer (CB) containing output  | uint32_t      | 0 to 31     | False    |
 */
// clang-format on
template <EltwiseBinaryType tBcastOp, BroadcastType tBcastDim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
[[deprecated(
    "Use compute_kernel_hw_startup(icb0, icb1, ocb) once at kernel start, then "
    "bcast_init<tBcastOp, tBcastDim>(icb0, icb1). This will be removed after September 15th, 2026.")]] void
init_bcast(uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, ocb, call_line);
    MATH((llk_math_eltwise_binary_init<tBcastOp, tBcastDim, MATH_FIDELITY>(icb0, icb1)));
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_hw_configure<is_fp32_dest_acc_en>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<tBcastDim>(icb0, icb1)));

    PACK((llk_pack_hw_configure<is_fp32_dest_acc_en>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<is_fp32_dest_acc_en, PackMode::Default>(ocb)));

    MATH((llk_math_pack_sync_init<is_fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<is_fp32_dest_acc_en>(icb0, icb1)));
#else
    UNPACK((llk_unpack_hw_configure(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<tBcastDim>(icb0, icb1)));

    PACK((llk_pack_hw_configure<is_fp32_dest_acc_en>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));

    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<is_fp32_dest_acc_en>(icb0, icb1)));
#endif
}

/*
Internal helper function for all broadcast ops
*/
template <EltwiseBinaryType tBcastOp, BroadcastType tBcastDim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void any_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
#ifdef ARCH_QUASAR
    // bcast_row_idx is only consumed by the ROW broadcast path; it is ignored by the Quasar LLK.
    if constexpr (tBcastDim == BroadcastType::ROW) {
        LLK_ASSERT(bcast_row_idx == 0, "non-default bcast_row_idx not supported on Quasar");
    }
#endif
    MATH((llk_math_eltwise_binary<tBcastOp, tBcastDim, is_fp32_dest_acc_en, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
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
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B  | uint32_t      | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t      | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t      | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <BroadcastType tBcastDim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWADD, tBcastDim, is_fp32_dest_acc_en>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx);
}

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template <BroadcastType tBcastDim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sub_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWSUB, tBcastDim, is_fp32_dest_acc_en>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx);
}

/**
 * Please refer to documentation for *add_tiles_bcast*.
 */
template <BroadcastType tBcastDim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_tiles_bcast(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWMUL, tBcastDim, is_fp32_dest_acc_en>(
        icb0, icb1, itile0, itile1, idst, bcast_row_idx);
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_rows to be executed
 * correctly. Required to be called before add_tiles_bcast if using column as broadcast type
 */
ALWI void add_bcast_rows_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::ROW, MathFidelity::LoFi>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_tiles_bcast_rows to be
 * executed correctly.
 */
ALWI void sub_bcast_rows_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::ROW, MathFidelity::LoFi>(icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_cols to be executed
 * correctly. Required to be called before add_tiles_bcast if using column as broadcast type
 */
ALWI void add_bcast_cols_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::COL, MathFidelity::LoFi>(icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for add_bcast_scalar to be
 * executed correctly.
 */
ALWI void add_bcast_scalar_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::SCALAR, MathFidelity::LoFi>(
        icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed
 * correctly.
 */
ALWI void mul_bcast_scalar_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>(icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

/**
 * Performs a broadcast-multiply of a tile from icb0[itile0] with a scalar encoded as a tile from icb1[itile1].
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mul_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::SCALAR,
          is_fp32_dest_acc_en,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for mul_bcast_cols to be executed
 * correctly.
 */
ALWI void mul_bcast_cols_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1)));
}

/**
 * Performs a switch-from-another-op tile hw reconfiguration step needed for mul_bcast_rows to be executed correctly.
 */
ALWI void mul_bcast_rows_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, MATH_FIDELITY>(icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_bcast_cols to be executed
 * correctly.
 */
ALWI void sub_bcast_cols_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::COL, MathFidelity::LoFi>(icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1)));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for sub_tiles_bcast_scalar to be
 * executed correctly.
 */
ALWI void sub_bcast_scalar_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::SCALAR, MathFidelity::LoFi>(
        icb0, icb1)));
    // FIXME: API Update needed in compute kernel?
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}


// clang-format off
/**
 * Canonical broadcast binary init (generic op/dim). Configures the math + unpack pipeline for a
 * broadcast binary op. The one-time hardware configuration must already have been performed via
 * compute_kernel_hw_startup(icb0, icb1, ocb) at the start of MAIN (mirrors eltwise_binary.h). For the
 * fixed op/dim entry points with their historical fidelity, use the
 * {add,sub,mul}_bcast_{rows,cols,scalar}_init wrappers.
 *
 * | Param Type | Name      | Description                               | Type              | Valid Range | Required |
 * |------------|-----------|-------------------------------------------|-------------------|-------------|----------|
 * | Template   | tBcastOp  | The binary op (ELWADD / ELWSUB / ELWMUL)  | EltwiseBinaryType | N/A         | True     |
 * | Template   | tBcastDim | The broadcast dim (ROW / COL / SCALAR)    | BroadcastType     | N/A         | True     |
 * | Function   | icb0      | CB containing A                           | uint32_t          | 0 to 31     | True     |
 * | Function   | icb1      | CB containing B                           | uint32_t          | 0 to 31     | True     |
 */
// clang-format on
template <EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>
ALWI void bcast_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init<tBcastOp, tBcastDim, MATH_FIDELITY>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<tBcastDim>(icb0, icb1)));
}

// =====================================================================================================================
// Deprecated broadcast init API
// New model: compute_kernel_hw_startup(icb0, icb1, ocb) once at MAIN start, then the per-op broadcast
// init (add_bcast_rows_init / mul_bcast_cols_init / ... , or the generic bcast_init<OP, DIM>). The
// forwarders below preserve the old *_init_short names; init_bcast (above) is the deprecated full-config init.
// =====================================================================================================================
[[deprecated("Renamed to add_bcast_rows_init(). This will be removed after September 15th, 2026.")]] ALWI void add_bcast_rows_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    add_bcast_rows_init(icb0, icb1, call_line);
}

[[deprecated("Renamed to add_bcast_cols_init(). This will be removed after September 15th, 2026.")]] ALWI void add_bcast_cols_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    add_bcast_cols_init(icb0, icb1, call_line);
}

[[deprecated("Renamed to add_bcast_scalar_init(). This will be removed after September 15th, 2026.")]] ALWI void add_bcast_scalar_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    add_bcast_scalar_init(icb0, icb1, call_line);
}

[[deprecated("Renamed to sub_bcast_rows_init(). This will be removed after September 15th, 2026.")]] ALWI void sub_bcast_rows_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    sub_bcast_rows_init(icb0, icb1, call_line);
}

[[deprecated("Renamed to sub_bcast_cols_init(). This will be removed after September 15th, 2026.")]] ALWI void sub_bcast_cols_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    sub_bcast_cols_init(icb0, icb1, call_line);
}

[[deprecated("Renamed to sub_bcast_scalar_init(). This will be removed after September 15th, 2026.")]] ALWI void sub_tiles_bcast_scalar_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    sub_bcast_scalar_init(icb0, icb1, call_line);
}

[[deprecated("Renamed to mul_bcast_rows_init(). This will be removed after September 15th, 2026.")]] ALWI void mul_bcast_rows_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    mul_bcast_rows_init(icb0, icb1, call_line);
}

[[deprecated("Renamed to mul_bcast_cols_init(). This will be removed after September 15th, 2026.")]] ALWI void mul_bcast_cols_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    mul_bcast_cols_init(icb0, icb1, call_line);
}

[[deprecated("Renamed to mul_bcast_scalar_init(). This will be removed after September 15th, 2026.")]] ALWI void mul_tiles_bcast_scalar_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    mul_bcast_scalar_init(icb0, icb1, call_line);
}

}  // namespace ckernel
