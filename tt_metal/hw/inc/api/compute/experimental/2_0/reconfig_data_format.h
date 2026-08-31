// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "data_format_derive.h"  // ckernel::infer_unpack_dst_format -- register-format derivation (BH)

#ifdef TRISC_UNPACK
// p_dim_stride_target + the raw UNPACK cores _llk_unpack_reconfig_data_format_srca_impl_/_srcb_impl_ (via
// the transitively-included llk_unpack_common.h). No id-free (LLKMemDescriptor-NTTP) unpack reconfig LLK
// exists yet, so this file calls straight through to the raw cores with a compile-time-resolved register
// format, bypassing the CB-id operand-array lookups the legacy llk_unpack_reconfig_data_format_srca(operand)
// wrapper does.
#include "llk_unpack_common_api.h"
#endif

#ifdef TRISC_MATH
// The raw MATH cores _llk_math_reconfig_data_format_srca_/_srcb_ (via the transitively-included
// llk_math_common.h). Same rationale as TRISC_UNPACK above -- no id-free MATH reconfig LLK exists yet.
#include "llk_math_common_api.h"
#endif

#ifdef TRISC_PACK
// The id-free packer reconfig ALREADY EXISTS here: llk_pack_reconfig_data_format<LLKMemDescriptor, bool>().
// This file only wraps it; no new PACK-side LLK code is written.
#include "experimental/2_0/llk_pack_tile.h"
#endif

// =====================================================================================================
// Id-free (2.0) reconfig_data_format -- SINGLE-OPERAND CORE surfaces only. Ports the three 1-operand CORE
// overloads of the legacy tt_metal/hw/inc/api/compute/reconfig_data_format.h to the id-free LLKOperand
// pattern:
//
//   1. reconfig_data_format_srca(new_a)   -- always re-derives + programs the SrcA format
//   2. reconfig_data_format_srcb(new_b)   -- always re-derives + programs the SrcB format
//   3. pack_reconfig_data_format(new_out) -- wraps the EXISTING id-free
//                                            llk_pack_reconfig_data_format<DESC,...>() from llk_pack_tile.h
//
// Each call unconditionally reprograms its operand's format -- the id-free surface deliberately keeps ONLY
// the single-`LLKOperand` forms. The legacy 2-arg (old,new) "skip if format unchanged" variants and the
// both-src reconfig_data_format(new_a,new_b)/(old_a,new_a,old_b,new_b) forms are intentionally NOT ported:
// with compile-time NTTP formats a caller that would benefit from the skip simply omits the call, so the
// old/new pairs and the two-operand exponent-width reconciliation add no value on the id-free surface.
//
// No id-free (LLKMemDescriptor-NTTP) UNPACK/MATH reconfig LLK exists yet (only the PACK side does, reused
// by pack_untilize). For srcA/srcB this file therefore derives the register format itself (via
// ckernel::infer_unpack_dst_format, data_format_derive.h) and calls the RAW per-arch cores directly
// (_llk_unpack_reconfig_data_format_srca_impl_ / _srcb_impl_, and _llk_math_reconfig_data_format_srca_ /
// _srcb_), bypassing the CB-id operand-array lookups the legacy llk_*_api.h wrappers do. No new llk_*_api.h
// file is created.
//
// DEFERRED (not ported -- see handoff for the full list): the 2-arg (old,new) skip variants; both-src
// reconfig_data_format; reconfig_full_operand[_srca/_srcb] (tile/face geometry); reconfig_tile_shape
// [_srca/_srcb] (shape-only, no format); the *_skip_int8 surface; SrcOrder (matmul operand-swap
// convenience); and all deprecated <bool,bool>/<to_from_int8,...> overloads.
// =====================================================================================================

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// ---------------------------------------------------------------------------------------------------------
// (1) reconfig_data_format_srca
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reconfigures the SrcA unpacker/math data format for a new operand, always re-deriving the register format
 * from the new L1 format (id-free equivalent of legacy reconfig_data_format_srca(new_operand)). Derives the
 * SrcA register format from `Format` via infer_unpack_dst_format(Format, DST_ACCUM_MODE), then reprograms the
 * UNPACK tile descriptor/format registers (raw core, dim_stride_target = IGNORE -- geometry untouched,
 * matching the legacy reconfig_data_format family, not reconfig_full_operand) and the MATH INT8-enable bit.
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Function   | new_a | New SrcA operand (format+shape are NTTPs)     | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void reconfig_data_format_srca(LLKOperand<Format, Shape> /*new_a*/) {
    static_assert(is_legal_tile_shape(Shape), "reconfig_data_format_srca: illegal tile shape.");
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(ckernel::infer_unpack_dst_format(Format, DST_ACCUM_MODE));
    constexpr std::uint32_t tile_size = tile_stride_words(Format, Shape);
    UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE, false>(
        static_cast<std::uint32_t>(Format),
        static_cast<std::uint32_t>(RegFmt),
        tile_size,
        Shape.face_r_dim,
        Shape.total_num_faces())));
    MATH((_llk_math_reconfig_data_format_srca_<DST_ACCUM_MODE, false>(static_cast<std::uint32_t>(RegFmt))));
}

// ---------------------------------------------------------------------------------------------------------
// (2) reconfig_data_format_srcb
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reconfigures the SrcB unpacker/math data format for a new operand, always re-deriving the register format
 * from the new L1 format (id-free equivalent of legacy reconfig_data_format_srcb(new_operand)). Mirrors
 * reconfig_data_format_srca for SrcB. Does not reprogram tile/face geometry.
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Function   | new_b | New SrcB operand (format+shape are NTTPs)     | LLKOperand | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void reconfig_data_format_srcb(LLKOperand<Format, Shape> /*new_b*/) {
    static_assert(is_legal_tile_shape(Shape), "reconfig_data_format_srcb: illegal tile shape.");
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(ckernel::infer_unpack_dst_format(Format, DST_ACCUM_MODE));
    constexpr std::uint32_t tile_size = tile_stride_words(Format, Shape);
    UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE, false>(
        static_cast<std::uint32_t>(Format),
        static_cast<std::uint32_t>(RegFmt),
        tile_size,
        Shape.face_r_dim,
        Shape.total_num_faces())));
    MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE, false>(static_cast<std::uint32_t>(RegFmt))));
}

// ---------------------------------------------------------------------------------------------------------
// (3) pack_reconfig_data_format -- thin wrapper over the EXISTING id-free LLK
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

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
