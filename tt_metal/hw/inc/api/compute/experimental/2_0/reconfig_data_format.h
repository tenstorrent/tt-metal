// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "data_format_derive.h"  // ckernel::infer_unpack_dst_format -- register-format derivation (BH)

#ifdef TRISC_UNPACK
// Raw UNPACK cores (_llk_unpack_reconfig_data_format_srca_impl_/_srcb_impl_); no id-free unpack reconfig LLK
// exists yet, so this file calls through directly with a compile-time-resolved register format.
#include "llk_unpack_common_api.h"
#endif

#ifdef TRISC_MATH
// Raw MATH cores (_llk_math_reconfig_data_format_srca_/_srcb_); no id-free MATH reconfig LLK exists yet either.
#include "llk_math_common_api.h"
#endif

#ifdef TRISC_PACK
// Id-free packer reconfig already exists (llk_pack_reconfig_data_format<LLKMemDescriptor, bool>()); this
// file just wraps it.
#include "experimental/2_0/llk_pack_tile.h"
#endif

// =====================================================================================================
// Id-free (2.0) reconfig_data_format -- single-operand surfaces only. Reprograms the srca / srcb / pack
// register data format from a new operand's descriptor NTTP:
//
//   1. reconfig_data_format_srca(new_a)   -- reprograms the SrcA format
//   2. reconfig_data_format_srcb(new_b)   -- reprograms the SrcB format
//   3. pack_reconfig_data_format(new_out) -- wraps the existing id-free llk_pack_reconfig_data_format<...>()
//
// Each call unconditionally reprograms its operand's format (format only; tile/face geometry is untouched).
// Since formats are compile-time NTTPs, a caller that wants to skip a call simply omits it: the legacy 2-arg
// (old,new) "skip if unchanged" variants and the both-src forms are not ported, nor are the geometry-aware
// forms (reconfig_full_operand, reconfig_tile_shape) or the SrcOrder / deprecated overloads. Blackhole only.
// =====================================================================================================

namespace ckernel {
namespace experimental {

// ---------------------------------------------------------------------------------------------------------
// (1) reconfig_data_format_srca
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reprograms the SrcA unpacker/math data format for a new operand. Derives the SrcA register format from
 * `Format` via infer_unpack_dst_format, then reprograms the UNPACK format registers and the MATH INT8-enable
 * bit. Tile/face geometry is not touched.
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode               | bool       |             | False    |
 * | Function   | new_a               | New SrcA operand (format+shape are NTTPs) | LLKOperand | N/A         | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void reconfig_data_format_srca(LLKOperand<Format, Shape> /*new_a*/) {
    static_assert(is_legal_tile_shape(Shape), "reconfig_data_format_srca: illegal tile shape.");
    constexpr std::uint8_t RegFmt =
        static_cast<std::uint8_t>(ckernel::infer_unpack_dst_format(Format, is_fp32_dest_acc_en));
    constexpr std::uint32_t tile_size = tile_stride_words(Format, Shape);
    UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
        static_cast<std::uint32_t>(Format),
        static_cast<std::uint32_t>(RegFmt),
        tile_size,
        Shape.face_r_dim,
        Shape.total_num_faces())));
    MATH((_llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(static_cast<std::uint32_t>(RegFmt))));
}

// ---------------------------------------------------------------------------------------------------------
// (2) reconfig_data_format_srcb
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reprograms the SrcB unpacker/math data format for a new operand. Mirrors reconfig_data_format_srca for
 * SrcB; tile/face geometry is not touched.
 *
 * | Param Type | Name  | Description                                   | Type       | Valid Range | Required |
 * |------------|-------|------------------------------------------------|------------|-------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode               | bool       |             | False    |
 * | Function   | new_b               | New SrcB operand (format+shape are NTTPs) | LLKOperand | N/A         | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void reconfig_data_format_srcb(LLKOperand<Format, Shape> /*new_b*/) {
    static_assert(is_legal_tile_shape(Shape), "reconfig_data_format_srcb: illegal tile shape.");
    constexpr std::uint8_t RegFmt =
        static_cast<std::uint8_t>(ckernel::infer_unpack_dst_format(Format, is_fp32_dest_acc_en));
    constexpr std::uint32_t tile_size = tile_stride_words(Format, Shape);
    UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
        static_cast<std::uint32_t>(Format),
        static_cast<std::uint32_t>(RegFmt),
        tile_size,
        Shape.face_r_dim,
        Shape.total_num_faces())));
    MATH((_llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, false>(static_cast<std::uint32_t>(RegFmt))));
}

// ---------------------------------------------------------------------------------------------------------
// (3) pack_reconfig_data_format -- thin wrapper over the EXISTING id-free LLK
// (llk_pack_reconfig_data_format<LLKMemDescriptor, bool> in llk_pack_tile.h, already used by pack_untilize).
// No new PACK-side LLK code; this is purely the compute-API-layer surface.
// ---------------------------------------------------------------------------------------------------------

// clang-format off
/**
 * Reprograms the packer output data format for a new output operand. Updates src/dst format registers and
 * tile geometry; does not reconfigure tile-dimension addrmod state (is_tile_dim_reconfig_en is not ported).
 *
 * | Param Type | Name    | Description                                    | Type       | Valid Range | Required |
 * |------------|---------|-------------------------------------------------|------------|-------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                    | bool       |             | False    |
 * | Function   | new_out             | New output operand (format+shape are NTTPs)   | LLKOperand | N/A         | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void pack_reconfig_data_format(LLKOperand<Format, Shape> /*new_out*/) {
    static_assert(is_legal_tile_shape(Shape), "pack_reconfig_data_format: illegal output tile shape.");
    PACK((llk_pack_reconfig_data_format<LLKOperand<Format, Shape>::descriptor, is_fp32_dest_acc_en>()));
}

}  // namespace experimental
}  // namespace ckernel
