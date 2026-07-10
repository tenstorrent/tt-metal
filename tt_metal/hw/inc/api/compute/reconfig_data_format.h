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
// (threaded through from the caller) and skip_int8 (hardcoded per public function: false to re-derive the int8/unsigned
// state from the format, true to skip it); every public function funnels through these helpers so the primary,
// deprecated, and _skip_int8 surfaces stay in lockstep.

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

}  // namespace detail

// The reconfig_data_format signature is changing (tt-metal#34499). Two things move at once:
//   1. int8 fix: the reconfig now always re-derives the int8/unsigned state (Src{A,B}Unsigned on unpack,
//      INT8_math_enabled on math) from the new format. The old to_from_int8 flag defaulted to false and was never
//      set, so those bits went stale when a reconfig crossed an int8 boundary and the math was wrong.
//   2. SrcOrder: the two-source reconfig now takes operands in natural (icb0, icb1) order and maps them onto
//      SrcA/SrcB from a SrcOrder tag, so matmul no longer swaps operands by hand (mirrors compute_kernel_hw_startup).
//
// SrcOrder is added as a same-name overload with no default on src_order, so a plain reconfig_data_format(a, b) call
// resolves to the functions below while an explicit reconfig_data_format<false, true>(...) still resolves to the
// deprecated bool overloads (further down). Those bool overloads carry the removed to_from_int8 flag, are deprecated,
// and work until 2026-08-15; the cleanup PR deletes them and their now-vestigial flag.
//
// NOTE(ARCH_QUASAR): On Quasar, buffer descriptors are programmed into the unpack MOP at op init. reconfig_data_format
// only reprograms THCON data formats (gasket), not the MOP. When operands or buffer descriptors change, call the op
// init again for the new operand pair before the next unpack operation.

/**
 * Reconfigures the srcA and srcB unpacker/math data formats for new operands, always re-deriving the int8/unsigned
 * state from the new formats. Operands are passed in natural (icb0, icb1) order; src_order selects how they map onto
 * SrcA/SrcB (SrcOrder::Reverse maps icb0 -> SrcB and icb1 -> SrcA, so matmul can pass its operands unswapped, matching
 * compute_kernel_hw_startup). Set is_tile_dim_reconfig_en when the new tile/face geometry differs from the current one.
 */
template <SrcOrder src_order = SrcOrder::Regular, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<is_tile_dim_reconfig_en, false>(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Conditional variant of reconfig_data_format: reconfigures only the sources whose format differs between the old
 * and new operand. Operands are in natural (icb0, icb1) order and honor src_order as above.
 */
template <SrcOrder src_order = SrcOrder::Regular, bool is_tile_dim_reconfig_en = false>
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
 * Reconfigures the srcA data format for a new operand, always re-deriving the int8/unsigned state from the new format.
 */
template <bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, false>(srca_new_operand);
}

/**
 * Reconfigures the srcA data format only if the new operand's format differs from the old one.
 */
template <bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, false>(srca_old_operand, srca_new_operand);
}

/**
 * Reconfigures the srcB data format for a new operand, always re-deriving the int8/unsigned state from the new format.
 */
template <bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, false>(srcb_new_operand);
}

/**
 * Reconfigures the srcB data format only if the new operand's format differs from the old one.
 */
template <bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, false>(srcb_old_operand, srcb_new_operand);
}

// Perf variant: skips re-deriving the int8/unsigned state (the old to_from_int8 == false behavior). Use it only when
// the caller knows the reconfig never crosses an Int8/UInt8/Int32 boundary and wants to avoid the extra register write.
// The two-source overloads take operands in natural (icb0, icb1) order and honor SrcOrder like reconfig_data_format.

/**
 * Same as reconfig_data_format, but skips re-deriving the int8/unsigned state (the old to_from_int8 == false
 * behavior). Use only when the caller knows the reconfig never crosses an Int8/UInt8/Int32 boundary and wants to
 * avoid the extra register write.
 */
template <SrcOrder src_order = SrcOrder::Regular, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_skip_int8(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<is_tile_dim_reconfig_en, true>(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}

/**
 * Conditional variant of reconfig_data_format_skip_int8: reconfigures only sources whose format differs, without
 * re-deriving the int8/unsigned state.
 */
template <SrcOrder src_order = SrcOrder::Regular, bool is_tile_dim_reconfig_en = false>
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
 * reconfig_data_format_srca without re-deriving the int8/unsigned state. See reconfig_data_format_skip_int8.
 */
template <bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srca_skip_int8(const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, true>(srca_new_operand);
}

/**
 * Conditional srcA reconfig without re-deriving the int8/unsigned state. See reconfig_data_format_skip_int8.
 */
template <bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srca_skip_int8(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<is_tile_dim_reconfig_en, true>(srca_old_operand, srca_new_operand);
}

/**
 * reconfig_data_format_srcb without re-deriving the int8/unsigned state. See reconfig_data_format_skip_int8.
 */
template <bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srcb_skip_int8(const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, true>(srcb_new_operand);
}

/**
 * Conditional srcB reconfig without re-deriving the int8/unsigned state. See reconfig_data_format_skip_int8.
 */
template <bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srcb_skip_int8(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srcb<is_tile_dim_reconfig_en, true>(srcb_old_operand, srcb_new_operand);
}

// -------------------------------------------------------------------------------------------------------------------
// Deprecated (tt-metal#34499). These keep the old <to_from_int8, is_tile_dim_reconfig_en> signature and work until
// 2026-08-15. to_from_int8 is now ignored: the int8/unsigned state is always re-derived from the format, so callers
// get the fix for free. Move to reconfig_data_format<SrcOrder::Regular>() (or reconfig_data_format_skip_int8() when you know
// no int8 boundary is crossed). The cleanup PR removes these overloads and the now-vestigial to_from_int8 flag.
//
// MIGRATION CAVEAT (srca/srcb): the new reconfig_data_format_srca/_srcb take ONE bool template arg
// (is_tile_dim_reconfig_en), the old ones took two (to_from_int8, is_tile_dim_reconfig_en). A one-arg call like
// reconfig_data_format_srca<true>(...) now binds true to is_tile_dim_reconfig_en (new), not to_from_int8 (old) --
// it compiles with no deprecation warning and reprograms tile/face geometry. The [[deprecated]] shim cannot catch
// this (the two-arg form still binds to the deprecated overload and warns). All in-repo call sites are migrated;
// external callers must drop the leading to_from_int8 arg (int8 state is derived regardless).
// -------------------------------------------------------------------------------------------------------------------

/// \cond DEPRECATED_RECONFIG_DATA_FORMAT (excluded from published docs; overloads the current API by template only)
#define RECONFIG_DF_DEPRECATED(new_fn)                                                                             \
    [[deprecated("reconfig_data_format signature is changing (SrcOrder support; int8 state always derived). Use " \
                 new_fn "() or the *_skip_int8 variant. This overload works until 2026-08-15. See tt-metal#34499.")]]

/**
 * @deprecated Use reconfig_data_format<SrcOrder::Regular>() (or reconfig_data_format_skip_int8()). Kept until 2026-08-15;
 * to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format. See tt-metal#34499.
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
 * @deprecated Use reconfig_data_format<SrcOrder::Regular>() (or reconfig_data_format_skip_int8()). Kept until 2026-08-15;
 * to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format. See tt-metal#34499.
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
 * @deprecated Use reconfig_data_format_srca() (or reconfig_data_format_srca_skip_int8()). Kept until 2026-08-15;
 * to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format. See tt-metal#34499.
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
 * @deprecated Use reconfig_data_format_srca() (or reconfig_data_format_srca_skip_int8()). Kept until 2026-08-15;
 * to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format. See tt-metal#34499.
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
 * @deprecated Use reconfig_data_format_srcb() (or reconfig_data_format_srcb_skip_int8()). Kept until 2026-08-15;
 * to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format. See tt-metal#34499.
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
 * @deprecated Use reconfig_data_format_srcb() (or reconfig_data_format_srcb_skip_int8()). Kept until 2026-08-15;
 * to_from_int8 is ignored and the int8/unsigned state is always re-derived from the format. See tt-metal#34499.
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
