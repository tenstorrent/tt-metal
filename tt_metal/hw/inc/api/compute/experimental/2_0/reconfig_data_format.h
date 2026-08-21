// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "data_format_derive.h"  // ckernel::infer_unpack_dst_format[_2op] -- register-format derivation (BH)

#ifdef TRISC_UNPACK
// p_dim_stride_target + the raw UNPACK cores _llk_unpack_reconfig_data_format_srca_impl_/_srcb_impl_ (via
// the transitively-included llk_unpack_common.h). No id-free (LLKMemDescriptor-NTTP) unpack reconfig LLK
// exists yet, so this file calls straight through to the raw cores with a compile-time-resolved register
// format, bypassing the CB-id operand-array lookups the legacy llk_unpack_reconfig_data_format_srca(operand)
// wrapper does.
#include "llk_unpack_common_api.h"
#endif

#ifdef TRISC_MATH
// The raw MATH cores _llk_math_reconfig_data_format_srca_/_srcb_/_ (via the transitively-included
// llk_math_common.h). Same rationale as TRISC_UNPACK above -- no id-free MATH reconfig LLK exists yet.
#include "llk_math_common_api.h"
#endif

#ifdef TRISC_PACK
// The id-free packer reconfig ALREADY EXISTS here: llk_pack_reconfig_data_format<LLKMemDescriptor, bool>().
// This file only wraps it; no new PACK-side LLK code is written.
#include "experimental/2_0/llk_pack_tile.h"
#endif

// =====================================================================================================
// Id-free (2.0) reconfig_data_format -- CORE families only. Ports the four CORE overloads of the legacy
// tt_metal/hw/inc/api/compute/reconfig_data_format.h to the id-free LLKOperand pattern:
//
//   1. reconfig_data_format_srca  -- 1-arg (always programs) + 2-arg (skip if format unchanged)
//   2. reconfig_data_format_srcb  -- 1-arg + 2-arg
//   3. reconfig_data_format       -- both srcA+srcB, 1-arg (new_a,new_b) + 2-arg (old_a,new_a,old_b,new_b)
//   4. pack_reconfig_data_format  -- 1-arg (new_out) + 2-arg (old_out,new_out); wraps the EXISTING id-free
//                                    llk_pack_reconfig_data_format<DESC,...>() from llk_pack_tile.h (2_0)
//
// Legacy's runtime "skip if old==new" optimization becomes a compile-time `if constexpr` on the operands'
// ::descriptor.format (or, equivalently here, the Format NTTP directly) -- the skip fires at COMPILE time,
// so a matched-format reconfig call compiles down to nothing extra, not just a cheap runtime branch.
//
// No id-free (LLKMemDescriptor-NTTP) UNPACK/MATH reconfig LLK exists yet (only the PACK side does, reused
// by pack_untilize). For srcA/srcB/both this file therefore derives the register format itself (via
// ckernel::infer_unpack_dst_format / infer_unpack_dst_format_2op, data_format_derive.h) and calls the RAW
// per-arch cores directly (_llk_unpack_reconfig_data_format_srca_impl_ / _srcb_impl_, and
// _llk_math_reconfig_data_format_srca_ / _srcb_ / the two-operand _llk_math_reconfig_data_format_),
// bypassing the CB-id operand-array lookups the legacy llk_*_api.h wrappers do. This mirrors exactly what
// the task calls "resolve RegFmt via infer_* and call the legacy entry/_impl with the resolved scalar
// format" -- no new llk_*_api.h file is created.
//
// DEFERRED (not ported -- see handoff for the full list): reconfig_full_operand[_srca/_srcb]
// (tile/face geometry), reconfig_tile_shape[_srca/_srcb] (shape-only, no format), the *_skip_int8 surface,
// SrcOrder (matmul operand-swap convenience), and all deprecated <bool,bool>/<to_from_int8,...> overloads.
// =====================================================================================================

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

namespace detail {

// clang-format off
/**
 * (detail) Unconditional SrcA format program: derives the SrcA register format from the L1 format
 * `Format` via infer_unpack_dst_format(Format, DST_ACCUM_MODE), then reprograms the UNPACK tile
 * descriptor/format registers (raw core, dim_stride_target = IGNORE -- geometry untouched, matching the
 * legacy reconfig_data_format family, not reconfig_full_operand) and the MATH INT8-enable bit. Shared core
 * for both the 1-arg and 2-arg (changed-only) public reconfig_data_format_srca surfaces.
 *
 * | Param Type | Name   | Description                          | Type        | Valid Range | Required |
 * |------------|--------|---------------------------------------|-------------|-------------|----------|
 * | Template   | Format | New SrcA buffer L1 data format        | DataFormat  | N/A         | True     |
 * | Template   | Shape  | New SrcA tile geometry                | TensorShape | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void reconfig_df_srca() {
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(ckernel::infer_unpack_dst_format(Format, DST_ACCUM_MODE));
    constexpr std::uint32_t tile_size = tile_stride_words(static_cast<std::uint8_t>(Format), Shape);
    UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE, false>(
        static_cast<std::uint32_t>(Format),
        static_cast<std::uint32_t>(RegFmt),
        tile_size,
        Shape.face_r_dim,
        Shape.total_num_faces())));
    MATH((_llk_math_reconfig_data_format_srca_<DST_ACCUM_MODE, false>(static_cast<std::uint32_t>(RegFmt))));
}

// clang-format off
/**
 * (detail) Unconditional SrcB format program. Mirrors reconfig_df_srca for SrcB.
 *
 * | Param Type | Name   | Description                          | Type        | Valid Range | Required |
 * |------------|--------|---------------------------------------|-------------|-------------|----------|
 * | Template   | Format | New SrcB buffer L1 data format        | DataFormat  | N/A         | True     |
 * | Template   | Shape  | New SrcB tile geometry                | TensorShape | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void reconfig_df_srcb() {
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(ckernel::infer_unpack_dst_format(Format, DST_ACCUM_MODE));
    constexpr std::uint32_t tile_size = tile_stride_words(static_cast<std::uint8_t>(Format), Shape);
    UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE, false>(
        static_cast<std::uint32_t>(Format),
        static_cast<std::uint32_t>(RegFmt),
        tile_size,
        Shape.face_r_dim,
        Shape.total_num_faces())));
    MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE, false>(static_cast<std::uint32_t>(RegFmt))));
}

// clang-format off
/**
 * (detail) Unconditional SrcA+SrcB format program. Register formats come from the TWO-OPERAND exponent-
 * width helper infer_unpack_dst_format_2op<Self,Other>(DST_ACCUM_MODE) (data_format_derive.h), which fires a
 * compile-time static_assert if the two L1 formats mix exponent-width families in a way the HW cannot
 * support (only Float32 rebiases to the other operand's family). UNPACK programs SrcA and SrcB
 * independently (each core only touches its own SrcXUnsigned bit); MATH uses the COMBINED two-operand core
 * so the INT8-math-enable bit is OR'd across both formats in one write, matching legacy's
 * detail::reconfig_df_both -> llk_math_reconfig_data_format(srca_new, srcb_new) path exactly (calling the
 * srca_/srcb_ MATH cores back-to-back would instead leave only the LAST call's format's bit set).
 *
 * | Param Type | Name           | Description                    | Type                   | Valid Range | Required |
 * |------------|----------------|---------------------------------|------------------------|-------------|----------|
 * | Template   | AFormat/AShape | New SrcA L1 format + geometry  | DataFormat/TensorShape | N/A         | True     |
 * | Template   | BFormat/BShape | New SrcB L1 format + geometry  | DataFormat/TensorShape | N/A         | True     |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void reconfig_df_both() {
    constexpr std::uint8_t RegA =
        static_cast<std::uint8_t>(ckernel::infer_unpack_dst_format_2op<AFormat, BFormat>(DST_ACCUM_MODE));
    constexpr std::uint8_t RegB =
        static_cast<std::uint8_t>(ckernel::infer_unpack_dst_format_2op<BFormat, AFormat>(DST_ACCUM_MODE));
    constexpr std::uint32_t tile_size_a = tile_stride_words(static_cast<std::uint8_t>(AFormat), AShape);
    constexpr std::uint32_t tile_size_b = tile_stride_words(static_cast<std::uint8_t>(BFormat), BShape);
    UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE, false>(
        static_cast<std::uint32_t>(AFormat),
        static_cast<std::uint32_t>(RegA),
        tile_size_a,
        AShape.face_r_dim,
        AShape.total_num_faces())));
    UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE, false>(
        static_cast<std::uint32_t>(BFormat),
        static_cast<std::uint32_t>(RegB),
        tile_size_b,
        BShape.face_r_dim,
        BShape.total_num_faces())));
    MATH((_llk_math_reconfig_data_format_<DST_ACCUM_MODE, false>(
        static_cast<std::uint32_t>(RegA), static_cast<std::uint32_t>(RegB))));
}

}  // namespace detail

// ---------------------------------------------------------------------------------------------------------
// (1) reconfig_data_format_srca
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reconfigures the SrcA unpacker/math data format for a new operand, always re-deriving the register format
 * from the new L1 format (id-free equivalent of legacy reconfig_data_format_srca(new_operand)). Does not
 * reprogram tile/face geometry (id-free reconfig_full_operand_srca is out of scope for this port).
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Function   | new_a | New SrcA operand (format+shape are NTTPs)     | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void reconfig_data_format_srca(LLKOperand<Format, Shape> /*new_a*/) {
    static_assert(is_legal_tile_shape(Shape), "reconfig_data_format_srca: illegal tile shape.");
    detail::reconfig_df_srca<Format, Shape>();
}

// clang-format off
/**
 * Conditional variant: reconfigures SrcA only if the new operand's L1 format differs from the old one's --
 * a COMPILE-TIME analog of legacy's runtime should_reconfigure_cbs skip (the skip fires on format equality
 * alone; unlike legacy it does not also auto-detect and reprogram on a geometry-only change with a matching
 * format -- see handoff RISKS).
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Function   | old_a | Currently-configured SrcA operand              | LLKOperand | N/A         | True     |
 * | Function   | new_a | New SrcA operand to switch to                  | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat OldFormat, TensorShape OldShape, DataFormat Format, TensorShape Shape>
ALWI void reconfig_data_format_srca(LLKOperand<OldFormat, OldShape> /*old_a*/, LLKOperand<Format, Shape> new_a) {
    static_assert(is_legal_tile_shape(Shape), "reconfig_data_format_srca: illegal tile shape.");
    if constexpr (OldFormat == Format) {
        // no-op: SrcA's format is unchanged.
    } else {
        reconfig_data_format_srca(new_a);
    }
}

// ---------------------------------------------------------------------------------------------------------
// (2) reconfig_data_format_srcb
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reconfigures the SrcB unpacker/math data format for a new operand, always re-deriving the register format
 * from the new L1 format (id-free equivalent of legacy reconfig_data_format_srcb(new_operand)). Does not
 * reprogram tile/face geometry.
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Function   | new_b | New SrcB operand (format+shape are NTTPs)     | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void reconfig_data_format_srcb(LLKOperand<Format, Shape> /*new_b*/) {
    static_assert(is_legal_tile_shape(Shape), "reconfig_data_format_srcb: illegal tile shape.");
    detail::reconfig_df_srcb<Format, Shape>();
}

// clang-format off
/**
 * Conditional variant: reconfigures SrcB only if the new operand's L1 format differs from the old one's.
 * See reconfig_data_format_srca(old,new) for the compile-time-skip vs legacy-geometry-auto-detect note.
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Function   | old_b | Currently-configured SrcB operand              | LLKOperand | N/A         | True     |
 * | Function   | new_b | New SrcB operand to switch to                  | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat OldFormat, TensorShape OldShape, DataFormat Format, TensorShape Shape>
ALWI void reconfig_data_format_srcb(LLKOperand<OldFormat, OldShape> /*old_b*/, LLKOperand<Format, Shape> new_b) {
    static_assert(is_legal_tile_shape(Shape), "reconfig_data_format_srcb: illegal tile shape.");
    if constexpr (OldFormat == Format) {
        // no-op: SrcB's format is unchanged.
    } else {
        reconfig_data_format_srcb(new_b);
    }
}

// ---------------------------------------------------------------------------------------------------------
// (3) reconfig_data_format (SrcA + SrcB)
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reconfigures the SrcA and SrcB unpacker/math data formats for new operands, always re-deriving both
 * register formats via the two-operand exponent-width helper (infer_unpack_dst_format_2op), which fires a
 * compile-time assertion if the two formats mix exponent-width families incompatibly (only Float32 rebiases
 * to the other operand's family). Id-free equivalent of legacy reconfig_data_format(icb0_new, icb1_new);
 * SrcOrder is out of scope for this port -- operands map 1:1 onto SrcA/SrcB in argument order.
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Function   | new_a | New SrcA operand (format+shape are NTTPs)     | LLKOperand | N/A         | True     |
 * | Function   | new_b | New SrcB operand (format+shape are NTTPs)     | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void reconfig_data_format(LLKOperand<AFormat, AShape> /*new_a*/, LLKOperand<BFormat, BShape> /*new_b*/) {
    static_assert(is_legal_tile_shape(AShape), "reconfig_data_format: illegal tile shape for operand A.");
    static_assert(is_legal_tile_shape(BShape), "reconfig_data_format: illegal tile shape for operand B.");
    detail::reconfig_df_both<AFormat, AShape, BFormat, BShape>();
}

// clang-format off
/**
 * Conditional variant: reconfigures SrcA and/or SrcB only for the source(s) whose format changed, choosing
 * the combined two-operand core when BOTH changed (so the INT8-math-enable bit stays correctly OR'd across
 * A and B) and the single-operand path when only one changed -- the compile-time analog of legacy
 * llk_math_reconfig_data_format(old_a,new_a,old_b,new_b)'s three-way runtime branch. See
 * reconfig_data_format_srca(old,new) for the compile-time-skip vs legacy-geometry-auto-detect note.
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Function   | old_a | Currently-configured SrcA operand              | LLKOperand | N/A         | True     |
 * | Function   | new_a | New SrcA operand                               | LLKOperand | N/A         | True     |
 * | Function   | old_b | Currently-configured SrcB operand              | LLKOperand | N/A         | True     |
 * | Function   | new_b | New SrcB operand                               | LLKOperand | N/A         | True     |
 */
// clang-format on
template <
    DataFormat OAFormat,
    TensorShape OAShape,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat OBFormat,
    TensorShape OBShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void reconfig_data_format(
    LLKOperand<OAFormat, OAShape> /*old_a*/,
    LLKOperand<AFormat, AShape> new_a,
    LLKOperand<OBFormat, OBShape> /*old_b*/,
    LLKOperand<BFormat, BShape> new_b) {
    static_assert(is_legal_tile_shape(AShape), "reconfig_data_format: illegal tile shape for operand A.");
    static_assert(is_legal_tile_shape(BShape), "reconfig_data_format: illegal tile shape for operand B.");
    if constexpr (OAFormat == AFormat && OBFormat == BFormat) {
        // no-op: neither source's format changed.
    } else if constexpr (OAFormat != AFormat && OBFormat != BFormat) {
        // both changed: combined core, correctly OR's the INT8-math-enable bit across A and B.
        reconfig_data_format(new_a, new_b);
    } else if constexpr (OAFormat != AFormat) {
        reconfig_data_format_srca(new_a);
    } else {
        reconfig_data_format_srcb(new_b);
    }
}

// ---------------------------------------------------------------------------------------------------------
// (4) pack_reconfig_data_format -- thin wrapper over the EXISTING id-free LLK
// (llk_pack_reconfig_data_format<LLKMemDescriptor, bool> in llk_pack_tile.h, already used by pack_untilize).
// No new PACK-side LLK code; this is purely the compute-API-layer surface.
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reconfigures the packer output data format for a new output operand. Always performs the reconfiguration
 * (id-free equivalent of legacy pack_reconfig_data_format(new_cb_id)); reprograms only src/dst format
 * registers + tile geometry, not tile-dimension addrmod state (is_tile_dim_reconfig_en is not ported --
 * deferred, see handoff).
 *
 * | Param Type | Name    | Description                                    | Type       | Valid Range | Required |
 * |------------|---------|-------------------------------------------------|------------|-------------|----------|
 * | Function   | new_out | New output operand (format+shape are NTTPs)   | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_reconfig_data_format(LLKOperand<Format, Shape> /*new_out*/) {
    static_assert(is_legal_tile_shape(Shape), "pack_reconfig_data_format: illegal output tile shape.");
    PACK((llk_pack_reconfig_data_format<LLKOperand<Format, Shape>::descriptor, DST_ACCUM_MODE>()));
}

// clang-format off
/**
 * Conditional variant: reconfigures the packer output format only if the new operand's L1 format differs
 * from the old one's (id-free equivalent of legacy pack_reconfig_data_format(old_cb_id, new_cb_id); the
 * legacy Invalid-format guard has no id-free analog since an LLKOperand's Format is always a concrete
 * compile-time DataFormat, never Invalid).
 *
 * | Param Type | Name    | Description                                    | Type       | Valid Range | Required |
 * |------------|---------|-------------------------------------------------|------------|-------------|----------|
 * | Function   | old_out | Currently-configured output operand            | LLKOperand | N/A         | True     |
 * | Function   | new_out | New output operand to switch to                | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat OldFormat, TensorShape OldShape, DataFormat Format, TensorShape Shape>
ALWI void pack_reconfig_data_format(LLKOperand<OldFormat, OldShape> /*old_out*/, LLKOperand<Format, Shape> new_out) {
    static_assert(is_legal_tile_shape(Shape), "pack_reconfig_data_format: illegal output tile shape.");
    if constexpr (OldFormat == Format) {
        // no-op: output format is unchanged.
    } else {
        pack_reconfig_data_format(new_out);
    }
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
