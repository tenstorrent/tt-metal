// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"
#include "sanitizer/api.h"
#include "api/compute/src_order.h"

#ifdef TRISC_PACK
#include "llk_pack_common_api.h"
#include "llk_pack_tile_api.h"
#endif

namespace ckernel {

namespace detail {

// Shared implementation for the reconfig_data_format family. The public entry points pick is_tile_dim_reconfig_en
// (false for reconfig_data_format, true for reconfig_full_operand) and skip_int8 (false to re-derive the
// int8/unsigned state from the format, true for the _skip_int8 surface); every format-writing public function funnels
// through these helpers so the primary, deprecated, and _skip_int8 surfaces stay in lockstep.

// p_dim_stride_target is declared only on the unpack thread (via the unpack API header). This helper, and every other
// reference to that type, therefore lives under TRISC_UNPACK and is called only inside UNPACK((...)) -- which expands to
// nothing on the math/pack threads, so the type is never named there (naming it unconditionally breaks the math build).
#ifdef TRISC_UNPACK
constexpr p_dim_stride_target dim_stride_of(bool is_tile_dim_reconfig_en) {
    return is_tile_dim_reconfig_en ? p_dim_stride_target::FACE_ROW_MAJOR : p_dim_stride_target::IGNORE;
}
#endif

template <bool is_tile_dim_reconfig_en, bool skip_int8>
ALWI void reconfig_df_both(const uint32_t src_a_operand, const uint32_t src_b_operand) {
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE, dim_stride_of(is_tile_dim_reconfig_en), skip_int8>(
        src_a_operand, src_b_operand)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE, skip_int8>(src_a_operand, src_b_operand)));
}

template <bool is_tile_dim_reconfig_en, bool skip_int8>
ALWI void reconfig_df_both(
    const uint32_t src_a_old_operand,
    const uint32_t src_a_new_operand,
    const uint32_t src_b_old_operand,
    const uint32_t src_b_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE, dim_stride_of(is_tile_dim_reconfig_en), skip_int8>(
        src_a_old_operand, src_a_new_operand, src_b_old_operand, src_b_new_operand)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE, skip_int8>(
        src_a_old_operand, src_a_new_operand, src_b_old_operand, src_b_new_operand)));
}

template <bool is_tile_dim_reconfig_en, bool skip_int8>
ALWI void reconfig_df_srca(const uint32_t srca_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, dim_stride_of(is_tile_dim_reconfig_en), skip_int8>(
        srca_new_operand)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE, skip_int8>(srca_new_operand)));
}

template <bool is_tile_dim_reconfig_en, bool skip_int8>
ALWI void reconfig_df_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, dim_stride_of(is_tile_dim_reconfig_en), skip_int8>(
        srca_old_operand, srca_new_operand)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE, skip_int8>(srca_old_operand, srca_new_operand)));
}

template <bool is_tile_dim_reconfig_en, bool skip_int8>
ALWI void reconfig_df_srcb(const uint32_t srcb_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format_srcb<DST_ACCUM_MODE, dim_stride_of(is_tile_dim_reconfig_en), skip_int8>(
        srcb_new_operand)));
    MATH((llk_math_reconfig_data_format_srcb<DST_ACCUM_MODE, skip_int8>(srcb_new_operand)));
}

template <bool is_tile_dim_reconfig_en, bool skip_int8>
ALWI void reconfig_df_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    UNPACK((llk_unpack_reconfig_data_format_srcb<DST_ACCUM_MODE, dim_stride_of(is_tile_dim_reconfig_en), skip_int8>(
        srcb_old_operand, srcb_new_operand)));
    MATH((llk_math_reconfig_data_format_srcb<DST_ACCUM_MODE, skip_int8>(srcb_old_operand, srcb_new_operand)));
}

// Shape-only helpers (UNPACK-only; no MATH half -- the math-side reconfig has no tile/face geometry). Named
// reconfig_ts_* (not reconfig_tile_shape_*) so they do not collide with the public reconfig_tile_shape* functions,
// mirroring the reconfig_df_* naming. Defined only off Quasar, where tile geometry is programmed at op init, not by
// reconfig -- so the whole geometry surface (reconfig_full_operand / reconfig_tile_shape) is compiled out there.
#ifndef ARCH_QUASAR
ALWI void reconfig_ts_both(const uint32_t src_a_operand, const uint32_t src_b_operand) {
    UNPACK((llk_unpack_reconfig_tile_shape_srca(src_a_operand)));
    UNPACK((llk_unpack_reconfig_tile_shape_srcb(src_b_operand)));
}

ALWI void reconfig_ts_both(
    const uint32_t src_a_old_operand,
    const uint32_t src_a_new_operand,
    const uint32_t src_b_old_operand,
    const uint32_t src_b_new_operand) {
    UNPACK((llk_unpack_reconfig_tile_shape_srca(src_a_old_operand, src_a_new_operand)));
    UNPACK((llk_unpack_reconfig_tile_shape_srcb(src_b_old_operand, src_b_new_operand)));
}

ALWI void reconfig_ts_srca(const uint32_t srca_new_operand) {
    UNPACK((llk_unpack_reconfig_tile_shape_srca(srca_new_operand)));
}

ALWI void reconfig_ts_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    UNPACK((llk_unpack_reconfig_tile_shape_srca(srca_old_operand, srca_new_operand)));
}

ALWI void reconfig_ts_srcb(const uint32_t srcb_new_operand) {
    UNPACK((llk_unpack_reconfig_tile_shape_srcb(srcb_new_operand)));
}

ALWI void reconfig_ts_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    UNPACK((llk_unpack_reconfig_tile_shape_srcb(srcb_old_operand, srcb_new_operand)));
}
#endif

}  // namespace detail

// The reconfig_data_format family expresses *what* is being reconfigured through three intent-named entry points:
//
//   * reconfig_data_format  -- data format + tile size (tile/face geometry unchanged). The common, cheap case.
//   * reconfig_full_operand -- data format + tile size + tile/face geometry. Use when the new operand's tile or face
//                              shape differs from what is currently programmed.
//   * reconfig_tile_shape   -- tile size + tile/face geometry only, no format. Use when only the tile shape changes.
//
// The two format-writing entry points (reconfig_data_format, reconfig_full_operand) always re-derive the int8/unsigned
// state from the new format (Src{A,B}Unsigned on unpack, INT8_math_enabled on math), so a reconfig across an
// Int8/UInt8/Int32 boundary is handled automatically. reconfig_tile_shape writes neither format nor int8 state. The
// *_skip_int8 surface skips the int8 re-derivation for callers that know no int8 boundary is crossed and want to avoid
// the extra register write.
//
// The two-source overloads take operands in natural (icb0, icb1) order; src_order selects how they map onto SrcA/SrcB
// (SrcOrder::Reverse maps icb0 -> SrcB and icb1 -> SrcA, so matmul can pass its operands unswapped, matching
// compute_kernel_hw_startup). The srcA-only / srcB-only overloads reconfigure a single source and take no SrcOrder.
//
// NOTE(ARCH_QUASAR): On Quasar, buffer descriptors are programmed into the unpack MOP at op init. reconfig_data_format
// only reprograms THCON data formats (gasket), not the MOP. When operands or buffer descriptors change, call the op
// init again for the new operand pair before the next unpack operation. Because tile geometry lives in the MOP,
// reconfig_full_operand and reconfig_tile_shape do not exist on Quasar -- reprogram geometry by re-running the op init.

// ------------------------------------------------------------------------------------------------------------------
// reconfig_data_format -- data format + tile size, tile/face geometry unchanged.
// ------------------------------------------------------------------------------------------------------------------

/**
 * Reconfigures the srcA and srcB unpacker/math data formats for new operands, always re-deriving the int8/unsigned
 * state from the new formats. Operands are passed in natural (icb0, icb1) order; src_order selects how they map onto
 * SrcA/SrcB. Does not reprogram tile/face geometry -- use reconfig_full_operand when the tile shape also changes.
 */
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_data_format(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<false, false>(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Conditional variant of reconfig_data_format: reconfigures only the sources whose format differs between the old and
 * new operand. Operands are in natural (icb0, icb1) order and honor src_order as above.
 *
 * NOTE: like all conditional (old, new) reconfig overloads, this re-detects the operands' face geometry and updates it
 * when it differs (long-standing auto-detect behavior). The unconditional overload above never touches geometry.
 */
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_data_format(
    const uint32_t icb0_old_operand,
    const uint32_t icb0_new_operand,
    const uint32_t icb1_old_operand,
    const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<false, false>(
        reverse ? icb1_old_operand : icb0_old_operand,
        reverse ? icb1_new_operand : icb0_new_operand,
        reverse ? icb0_old_operand : icb1_old_operand,
        reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Reconfigures the srcA data format for a new operand, always re-deriving the int8/unsigned state from the new format.
 */
ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<false, false>(srca_new_operand);
}

/**
 * Reconfigures the srcA data format only if the new operand's format differs from the old one. See the conditional
 * geometry note on reconfig_data_format(old, new, ...).
 */
ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<false, false>(srca_old_operand, srca_new_operand);
}

/**
 * Reconfigures the srcB data format for a new operand, always re-deriving the int8/unsigned state from the new format.
 */
ALWI void reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<false, false>(srcb_new_operand);
}

/**
 * Reconfigures the srcB data format only if the new operand's format differs from the old one. See the conditional
 * geometry note on reconfig_data_format(old, new, ...).
 */
ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<false, false>(srcb_old_operand, srcb_new_operand);
}

#ifndef ARCH_QUASAR
// ------------------------------------------------------------------------------------------------------------------
// reconfig_full_operand -- data format + tile size + tile/face geometry. Wormhole/Blackhole only (see NOTE above).
// ------------------------------------------------------------------------------------------------------------------

/**
 * Reconfigures the srcA and srcB unpacker/math data formats AND tile/face geometry for new operands, always
 * re-deriving the int8/unsigned state from the new formats. Use when the new operands' tile or face shape differs from
 * what is currently programmed. Operands are in natural (icb0, icb1) order and honor src_order.
 */
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_full_operand(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<true, false>(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Conditional variant of reconfig_full_operand: reconfigures only the sources whose format differs between the old and
 * new operand, reprogramming their tile/face geometry. Operands are in natural (icb0, icb1) order.
 */
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_full_operand(
    const uint32_t icb0_old_operand,
    const uint32_t icb0_new_operand,
    const uint32_t icb1_old_operand,
    const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<true, false>(
        reverse ? icb1_old_operand : icb0_old_operand,
        reverse ? icb1_new_operand : icb0_new_operand,
        reverse ? icb0_old_operand : icb1_old_operand,
        reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Reconfigures the srcA data format and tile/face geometry for a new operand, re-deriving the int8/unsigned state.
 */
ALWI void reconfig_full_operand_srca(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<true, false>(srca_new_operand);
}

/**
 * Reconfigures the srcA data format and tile/face geometry only if the new operand's format differs from the old one.
 */
ALWI void reconfig_full_operand_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<true, false>(srca_old_operand, srca_new_operand);
}

/**
 * Reconfigures the srcB data format and tile/face geometry for a new operand, re-deriving the int8/unsigned state.
 */
ALWI void reconfig_full_operand_srcb(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<true, false>(srcb_new_operand);
}

/**
 * Reconfigures the srcB data format and tile/face geometry only if the new operand's format differs from the old one.
 */
ALWI void reconfig_full_operand_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<true, false>(srcb_old_operand, srcb_new_operand);
}

// ------------------------------------------------------------------------------------------------------------------
// reconfig_tile_shape -- tile size + tile/face geometry only, no format. UNPACK-only. Wormhole/Blackhole only.
// ------------------------------------------------------------------------------------------------------------------

/**
 * Reprograms only the srcA and srcB unpacker tile size and face geometry (x-dim, num_faces) for new operands, leaving
 * the data formats (and int8 state) untouched. Use when only the tile shape changes. Operands are in natural
 * (icb0, icb1) order and honor src_order.
 *
 * @note The format-derived ch1 strides are unchanged (a shape-only change does not alter them). A face_r_dim change
 *       additionally requires re-running the op init (it programs the unpacker ADC X-end); a num_faces-only change is
 *       fully handled here.
 */
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_tile_shape(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_ts_both(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Conditional variant of reconfig_tile_shape: reprograms srcA/srcB tile size and face geometry only for the sources
 * whose geometry (or CB) changed between the old and new operand. Operands are in natural (icb0, icb1) order.
 */
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_tile_shape(
    const uint32_t icb0_old_operand,
    const uint32_t icb0_new_operand,
    const uint32_t icb1_old_operand,
    const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_ts_both(
        reverse ? icb1_old_operand : icb0_old_operand,
        reverse ? icb1_new_operand : icb0_new_operand,
        reverse ? icb0_old_operand : icb1_old_operand,
        reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Reprograms only the srcA unpacker tile size and face geometry for a new operand, leaving the format untouched.
 */
ALWI void reconfig_tile_shape_srca(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_ts_srca(srca_new_operand);
}

/**
 * Reprograms the srcA unpacker tile size and face geometry only if the new operand's geometry (or CB) differs.
 */
ALWI void reconfig_tile_shape_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_ts_srca(srca_old_operand, srca_new_operand);
}

/**
 * Reprograms only the srcB unpacker tile size and face geometry for a new operand, leaving the format untouched.
 */
ALWI void reconfig_tile_shape_srcb(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_ts_srcb(srcb_new_operand);
}

/**
 * Reprograms the srcB unpacker tile size and face geometry only if the new operand's geometry (or CB) differs.
 */
ALWI void reconfig_tile_shape_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_ts_srcb(srcb_old_operand, srcb_new_operand);
}
#endif  // !ARCH_QUASAR

// ------------------------------------------------------------------------------------------------------------------
// reconfig_data_format_skip_int8 -- like reconfig_data_format, but skips re-deriving the int8/unsigned state. Use only
// when the caller knows the reconfig never crosses an Int8/UInt8/Int32 boundary and wants to avoid the extra register
// write. Tile/face geometry is left unchanged.
// ------------------------------------------------------------------------------------------------------------------

/**
 * Same as reconfig_data_format, but skips re-deriving the int8/unsigned state. Use only when the caller knows the
 * reconfig never crosses an Int8/UInt8/Int32 boundary and wants to avoid the extra register write.
 */
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_data_format_skip_int8(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<false, true>(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Conditional variant of reconfig_data_format_skip_int8: reconfigures only sources whose format differs, without
 * re-deriving the int8/unsigned state.
 */
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_data_format_skip_int8(
    const uint32_t icb0_old_operand,
    const uint32_t icb0_new_operand,
    const uint32_t icb1_old_operand,
    const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<false, true>(
        reverse ? icb1_old_operand : icb0_old_operand,
        reverse ? icb1_new_operand : icb0_new_operand,
        reverse ? icb0_old_operand : icb1_old_operand,
        reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * reconfig_data_format_srca without re-deriving the int8/unsigned state. See reconfig_data_format_skip_int8.
 */
ALWI void reconfig_data_format_srca_skip_int8(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<false, true>(srca_new_operand);
}

/**
 * Conditional srcA reconfig without re-deriving the int8/unsigned state. See reconfig_data_format_skip_int8.
 */
ALWI void reconfig_data_format_srca_skip_int8(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<false, true>(srca_old_operand, srca_new_operand);
}

/**
 * reconfig_data_format_srcb without re-deriving the int8/unsigned state. See reconfig_data_format_skip_int8.
 */
ALWI void reconfig_data_format_srcb_skip_int8(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<false, true>(srcb_new_operand);
}

/**
 * Conditional srcB reconfig without re-deriving the int8/unsigned state. See reconfig_data_format_skip_int8.
 */
ALWI void reconfig_data_format_srcb_skip_int8(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<false, true>(srcb_old_operand, srcb_new_operand);
}

// -------------------------------------------------------------------------------------------------------------------
// Deprecated (tt-metal#34499). The is_tile_dim_reconfig_en bool is replaced by intent-named entry points:
//   reconfig_data_format<SrcOrder, false> -> reconfig_data_format<SrcOrder>()
//   reconfig_data_format<SrcOrder, true>  -> reconfig_full_operand<SrcOrder>() (or reconfig_tile_shape() for geometry-only)
// (and the matching _srca / _srcb / _skip_int8 forms). These <SrcOrder, is_tile_dim> overloads preserve today's exact
// behavior and work until 2026-09-15. They coexist with the older <to_from_int8, is_tile_dim> bool overloads (further
// down), disambiguated by first template arg type: a SrcOrder first arg selects these; a bool first arg selects the
// older ones. The cleanup PR removes both.
// -------------------------------------------------------------------------------------------------------------------

/// \cond DEPRECATED_RECONFIG_DATA_FORMAT_TILE_DIM (excluded from published docs; overloads the current API by template only)
#define RECONFIG_DF_TILE_DIM_DEPRECATED(new_fn)                                                                       \
    [[deprecated("The is_tile_dim_reconfig_en bool on reconfig_data_format_* is replaced by intent-named APIs and "   \
                 "will be removed after September 15th 2026 (tt-metal#34499). Use " new_fn "() for geometry, or the " \
                 "plain reconfig_data_format*() when only the format changes.")]]

/**
 * @deprecated Use reconfig_full_operand<SrcOrder>() (geometry on) or reconfig_data_format<SrcOrder>() (format only).
 * Kept until 2026-09-15. See tt-metal#34499.
 */
template <SrcOrder src_order, bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_full_operand<SrcOrder>")
ALWI void reconfig_data_format(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<is_tile_dim_reconfig_en, false>(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * @deprecated Use reconfig_full_operand<SrcOrder>() or reconfig_data_format<SrcOrder>(). Kept until 2026-09-15.
 */
template <SrcOrder src_order, bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_full_operand<SrcOrder>")
ALWI void reconfig_data_format(
    const uint32_t icb0_old_operand,
    const uint32_t icb0_new_operand,
    const uint32_t icb1_old_operand,
    const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<is_tile_dim_reconfig_en, false>(
        reverse ? icb1_old_operand : icb0_old_operand,
        reverse ? icb1_new_operand : icb0_new_operand,
        reverse ? icb0_old_operand : icb1_old_operand,
        reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * @deprecated Use reconfig_full_operand_srca() or reconfig_data_format_srca(). Kept until 2026-09-15.
 */
template <bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_full_operand_srca")
ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, false>(srca_new_operand);
}

/**
 * @deprecated Use reconfig_full_operand_srca() or reconfig_data_format_srca(). Kept until 2026-09-15.
 */
template <bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_full_operand_srca")
ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, false>(srca_old_operand, srca_new_operand);
}

/**
 * @deprecated Use reconfig_full_operand_srcb() or reconfig_data_format_srcb(). Kept until 2026-09-15.
 */
template <bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_full_operand_srcb")
ALWI void reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, false>(srcb_new_operand);
}

/**
 * @deprecated Use reconfig_full_operand_srcb() or reconfig_data_format_srcb(). Kept until 2026-09-15.
 */
template <bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_full_operand_srcb")
ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, false>(srcb_old_operand, srcb_new_operand);
}

// _skip_int8 shims: same is_tile_dim removal. <..., true> preserves format + geometry + skip-int8 until removal.
/**
 * @deprecated The is_tile_dim bool was removed. Use reconfig_data_format_skip_int8<SrcOrder>() (or, for geometry,
 * reconfig_full_operand<SrcOrder>() which always derives int8). Kept until 2026-09-15. See tt-metal#34499.
 */
template <SrcOrder src_order, bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_data_format_skip_int8<SrcOrder>")
ALWI void reconfig_data_format_skip_int8(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<is_tile_dim_reconfig_en, true>(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * @deprecated See reconfig_data_format_skip_int8 above. Kept until 2026-09-15.
 */
template <SrcOrder src_order, bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_data_format_skip_int8<SrcOrder>")
ALWI void reconfig_data_format_skip_int8(
    const uint32_t icb0_old_operand,
    const uint32_t icb0_new_operand,
    const uint32_t icb1_old_operand,
    const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<is_tile_dim_reconfig_en, true>(
        reverse ? icb1_old_operand : icb0_old_operand,
        reverse ? icb1_new_operand : icb0_new_operand,
        reverse ? icb0_old_operand : icb1_old_operand,
        reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * @deprecated Use reconfig_data_format_srca_skip_int8(). Kept until 2026-09-15.
 */
template <bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_data_format_srca_skip_int8")
ALWI void reconfig_data_format_srca_skip_int8(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, true>(srca_new_operand);
}

/**
 * @deprecated Use reconfig_data_format_srca_skip_int8(). Kept until 2026-09-15.
 */
template <bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_data_format_srca_skip_int8")
ALWI void reconfig_data_format_srca_skip_int8(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, true>(srca_old_operand, srca_new_operand);
}

/**
 * @deprecated Use reconfig_data_format_srcb_skip_int8(). Kept until 2026-09-15.
 */
template <bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_data_format_srcb_skip_int8")
ALWI void reconfig_data_format_srcb_skip_int8(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, true>(srcb_new_operand);
}

/**
 * @deprecated Use reconfig_data_format_srcb_skip_int8(). Kept until 2026-09-15.
 */
template <bool is_tile_dim_reconfig_en>
RECONFIG_DF_TILE_DIM_DEPRECATED("reconfig_data_format_srcb_skip_int8")
ALWI void reconfig_data_format_srcb_skip_int8(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, true>(srcb_old_operand, srcb_new_operand);
}

#undef RECONFIG_DF_TILE_DIM_DEPRECATED
/// \endcond

// -------------------------------------------------------------------------------------------------------------------
// Deprecated (tt-metal#34499). These keep the old <to_from_int8, is_tile_dim_reconfig_en> signature and work until
// 2026-08-20. to_from_int8 is now ignored: the int8/unsigned state is always re-derived from the format, so callers
// get the fix for free. Move to reconfig_data_format<SrcOrder::Regular>() / reconfig_full_operand<SrcOrder::Regular>()
// (or reconfig_data_format_skip_int8() when you know no int8 boundary is crossed). The cleanup PR removes these
// overloads and the now-vestigial to_from_int8 flag.
// -------------------------------------------------------------------------------------------------------------------

/// \cond DEPRECATED_RECONFIG_DATA_FORMAT (excluded from published docs; overloads the current API by template only)
#define RECONFIG_DF_DEPRECATED(new_fn)                                                                              \
    [[deprecated("This call to reconfig_data_format_* will be removed after August 20th 2026 (tt-metal#34499). Use " \
                 new_fn "() or the *_skip_int8 variant; int8/unsigned state is now always derived from the format.")]]

/**
 * @deprecated Use reconfig_data_format<SrcOrder::Regular>() / reconfig_full_operand<SrcOrder::Regular>() (or
 * reconfig_data_format_skip_int8()). Kept until 2026-08-20; to_from_int8 is ignored and the int8/unsigned state is
 * always re-derived from the format. See tt-metal#34499.
 */
template <bool to_from_int8, bool is_tile_dim_reconfig_en>
RECONFIG_DF_DEPRECATED("reconfig_data_format<SrcOrder::Regular>")
ALWI void reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    // to_from_int8 is silently ignored on Quasar: the unpack LLK marks it [[maybe_unused]] and the
    // math reconfig is a no-op.
    static_assert(!to_from_int8, "non-default to_from_int8 not supported on Quasar");
#endif
    detail::reconfig_df_both<is_tile_dim_reconfig_en, false>(srca_new_operand, srcb_new_operand);
}

/**
 * @deprecated Use reconfig_data_format<SrcOrder::Regular>() / reconfig_full_operand<SrcOrder::Regular>() (or
 * reconfig_data_format_skip_int8()). Kept until 2026-08-20; to_from_int8 is ignored and the int8/unsigned state is
 * always re-derived from the format. See tt-metal#34499.
 */
template <bool to_from_int8, bool is_tile_dim_reconfig_en>
RECONFIG_DF_DEPRECATED("reconfig_data_format<SrcOrder::Regular>")
ALWI void reconfig_data_format(
    const uint32_t srca_old_operand,
    const uint32_t srca_new_operand,
    const uint32_t srcb_old_operand,
    const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    // to_from_int8 is silently ignored on Quasar: the unpack LLK marks it [[maybe_unused]] and the
    // math reconfig is a no-op.
    static_assert(!to_from_int8, "non-default to_from_int8 not supported on Quasar");
#endif
    detail::reconfig_df_both<is_tile_dim_reconfig_en, false>(
        srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand);
}

/**
 * @deprecated Use reconfig_data_format_srca() / reconfig_full_operand_srca() (or reconfig_data_format_srca_skip_int8()).
 * Kept until 2026-08-20; to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format.
 * See tt-metal#34499.
 */
template <bool to_from_int8, bool is_tile_dim_reconfig_en>
RECONFIG_DF_DEPRECATED("reconfig_data_format_srca")
ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    // to_from_int8 is silently ignored on Quasar: the unpack LLK marks it [[maybe_unused]] and the
    // math reconfig is a no-op.
    static_assert(!to_from_int8, "non-default to_from_int8 not supported on Quasar");
#endif
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, false>(srca_new_operand);
}

/**
 * @deprecated Use reconfig_data_format_srca() / reconfig_full_operand_srca() (or reconfig_data_format_srca_skip_int8()).
 * Kept until 2026-08-20; to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format.
 * See tt-metal#34499.
 */
template <bool to_from_int8, bool is_tile_dim_reconfig_en>
RECONFIG_DF_DEPRECATED("reconfig_data_format_srca")
ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    // to_from_int8 is silently ignored on Quasar: the unpack LLK marks it [[maybe_unused]] and the
    // math reconfig is a no-op.
    static_assert(!to_from_int8, "non-default to_from_int8 not supported on Quasar");
#endif
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, false>(srca_old_operand, srca_new_operand);
}

/**
 * @deprecated Use reconfig_data_format_srcb() / reconfig_full_operand_srcb() (or reconfig_data_format_srcb_skip_int8()).
 * Kept until 2026-08-20; to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format.
 * See tt-metal#34499.
 */
template <bool to_from_int8, bool is_tile_dim_reconfig_en>
RECONFIG_DF_DEPRECATED("reconfig_data_format_srcb")
ALWI void reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    // to_from_int8 is silently ignored on Quasar: the unpack LLK marks it [[maybe_unused]] and the
    // math reconfig is a no-op.
    static_assert(!to_from_int8, "non-default to_from_int8 not supported on Quasar");
#endif
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, false>(srcb_new_operand);
}

/**
 * @deprecated Use reconfig_data_format_srcb() / reconfig_full_operand_srcb() (or reconfig_data_format_srcb_skip_int8()).
 * Kept until 2026-08-20; to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format.
 * See tt-metal#34499.
 */
template <bool to_from_int8, bool is_tile_dim_reconfig_en>
RECONFIG_DF_DEPRECATED("reconfig_data_format_srcb")
ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    // to_from_int8 is silently ignored on Quasar: the unpack LLK marks it [[maybe_unused]] and the
    // math reconfig is a no-op.
    static_assert(!to_from_int8, "non-default to_from_int8 not supported on Quasar");
#endif
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, false>(srcb_old_operand, srcb_new_operand);
}

#undef RECONFIG_DF_DEPRECATED
/// \endcond

// clang-format off
/**
 * Reconfigures the packer output data format by specifying the CB ID of the new operand. This function
 * call will always perform the reconfiguration, regardless of the data format of the old operand.
 * If the new CB ID is the same as the current one, reconfiguration will still occur.
 *
 * NOTE(ARCH_QUASAR): On Quasar, buffer descriptors are programmed at op init. pack_reconfig_data_format
 * only reprograms THCON IN_DATA_FORMAT (gasket), not the MOP or buffer descriptors. When the pack output
 * operand changes, call pack_init(new_cb_id) before pack_tile.
 *
 * NOTE: Packer reconfiguration functions are used similarly to the initialization function, in a sense
 * that they are called before the call to the packer function that uses the new configuration. It is
 * recommended to call this function right after other op-specific initialization functions.
 *
 * Return value: None
 *
 * | Param Type | Name                    | Description                   | Type     | Valid Range | Required |
 * |------------|-------------------------|-------------------------------|----------|-------------|----------|
 * | Template   | is_tile_dim_reconfig_en | Toggle tile reconfiguration   | bool     | true/false  | False    |
 * | Function   | new_cb_id               | New data format operand value | uint32_t | Any         | True     |
 */
// clang-format on
template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const uint32_t new_cb_id) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    static_assert(
        !is_tile_dim_reconfig_en,
        "Quasar pack reconfig does not support tile-dimension changes; call pack_init instead");
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(new_cb_id)));
#else
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(new_cb_id)));
    if constexpr (is_tile_dim_reconfig_en) {
        PACK((llk_pack_init<PackMode::Default, false /* zero_output */, true /* skip_addrmod_config */>(new_cb_id)));
    }
#endif
}

// clang-format off
/**
 * Reconfigures the packer output data format by specifying the CB IDs of the old and new operands.
 * This function internally calls the reconfiguration function with the new CB ID, but before it does so,
 * it checks if the old and new data formats are different. If they are the same, it does not perform
 * the reconfiguration. This function is useful when you want to ensure that the packer only reconfigures
 * when different data format is wanted, avoiding unnecessary reconfiguration overhead.
 *
 * NOTE(ARCH_QUASAR): See pack_reconfig_data_format(new_cb_id). Call pack_init(new_cb_id) before pack_tile
 * when switching pack output operand.
 *
 * NOTE: Packer reconfiguration functions are used similarly to the initialization function, in a sense
 * that they are called before the call to the packer function that uses the new configuration. It is
 * recommended to call this function right after other op-specific initialization functions.
 *
 * Return value: None
 *
 * | Param Type | Name                    | Description                        | Type     | Valid Range | Required |
 * |------------|-------------------------|------------------------------------|----------|-------------|----------|
 * | Template   | is_tile_dim_reconfig_en | Toggle tile reconfiguration        | bool     | true/false  | False    |
 * | Function   | old_cb_id               | Previous data format operand value | uint32_t | Any         | True     |
 * | Function   | new_cb_id               | New data format operand value      | uint32_t | Any         | True     |
 */
// clang-format on
template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const uint32_t old_cb_id, const uint32_t new_cb_id) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    static_assert(
        !is_tile_dim_reconfig_en,
        "Quasar pack reconfig does not support tile-dimension changes; call pack_init instead");
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(old_cb_id, new_cb_id)));
#else
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(old_cb_id, new_cb_id)));
    if constexpr (is_tile_dim_reconfig_en) {
        PACK((llk_pack_init<PackMode::Default, false /* zero_output */, true /* skip_addrmod_config */>(new_cb_id)));
    }
#endif
}

}  // namespace ckernel
