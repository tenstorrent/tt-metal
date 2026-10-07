// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "data_format_derive.h"
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

// Id-free init helpers. Each 2.0 op init calls the helpers for the threads it uses, then its own
// MOP / addrmod setup. Unconditional writes; no format cache (add one only if BH .text regresses).

#ifdef TRISC_UNPACK
#include "llk_unpack_tilize_api.h"  // _llk_unpack_tilize_uninit_ + unpack reconfig

// SrcA/SrcB L1 and register formats via the 2-operand rule, geometry, then clear tilize bits.
// Register formats use infer_unpack_dst_format_2op (Float32 rebias / mixed-width compile error).
// FACE_ROW_MAJOR refreshes Tile_x_dim; the 2.0 reconfig path passes IGNORE and never does.
// Tilize teardown runs first so its canonical X/Z write does not clobber the geometry reconfig,
// and so tileize_mode/haloize/shift_amount are cleared. The following field RMWs leave those bits 0.
template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor DESC_A,
    ckernel::experimental::LLKMemDescriptor DESC_B>
inline void llk_unpack_config() {
    constexpr DataFormat A = DESC_A.format;
    constexpr DataFormat B = DESC_B.format;
    constexpr DataFormat reg_a = ckernel::infer_unpack_dst_format_2op<A, B>(is_fp32_dest_acc_en);
    constexpr DataFormat reg_b = ckernel::infer_unpack_dst_format_2op<B, A>(is_fp32_dest_acc_en);
    constexpr std::uint32_t tile_size_a = ckernel::experimental::tile_stride_words(A, DESC_A.shape);
    constexpr std::uint32_t tile_size_b = ckernel::experimental::tile_stride_words(B, DESC_B.shape);

    _llk_unpack_tilize_uninit_(static_cast<std::uint32_t>(reg_a), DESC_A.shape);
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, false>(
        static_cast<std::uint32_t>(A),
        static_cast<std::uint32_t>(reg_a),
        tile_size_a,
        DESC_A.shape.face_r_dim,
        DESC_A.shape.total_num_faces());
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, false>(
        static_cast<std::uint32_t>(B),
        static_cast<std::uint32_t>(reg_b),
        tile_size_b,
        DESC_B.shape.face_r_dim,
        DESC_B.shape.total_num_faces());
}
#endif

#ifdef TRISC_MATH
#include "llk_math_common_api.h"

// INT8 enable and both zero-flag format caches, via the combined math reconfig (the srca-then-srcb
// pair leaves INT8 = is_int(srcB)). That reconfig also applies the compute zero flag. Datacopy ops
// pass datacopy_zero_flag so the copy rule wins: preserve bf16 -0 / 16b ints, flush fp8.
template <
    bool is_fp32_dest_acc_en,
    ckernel::experimental::LLKMemDescriptor DESC_A,
    ckernel::experimental::LLKMemDescriptor DESC_B,
    bool datacopy_zero_flag = false>
inline void llk_math_config() {
    constexpr DataFormat reg_a = ckernel::infer_unpack_dst_format_2op<DESC_A.format, DESC_B.format>(is_fp32_dest_acc_en);
    constexpr DataFormat reg_b = ckernel::infer_unpack_dst_format_2op<DESC_B.format, DESC_A.format>(is_fp32_dest_acc_en);
    _llk_math_reconfig_data_format_<is_fp32_dest_acc_en, false>(
        static_cast<std::uint32_t>(reg_a), static_cast<std::uint32_t>(reg_b));
    if constexpr (datacopy_zero_flag) {
        ckernel::math::_configure_copy_zero_flag_state_(static_cast<std::uint32_t>(reg_a));
    }
}
#endif

#ifdef TRISC_PACK
#include "llk_pack_reduce_api.h"          // _llk_pack_reduce_mask_clear_
#include "experimental/2_0/llk_pack_tile.h"  // llk_pack_reconfig_data_format / llk_pack_init

// Clear the reduce edge mask, then program Default-mode formats, strides, and MOP.
// reduce_init sets the mask again after this so it survives. Every other init leaves it clear.
template <bool is_fp32_dest_acc_en, ckernel::experimental::LLKMemDescriptor OUT_DESC>
inline void llk_pack_config() {
    _llk_pack_reduce_mask_clear_();
    llk_pack_reconfig_data_format<OUT_DESC, is_fp32_dest_acc_en>();
    llk_pack_init<OUT_DESC, is_fp32_dest_acc_en>();
}
#endif
