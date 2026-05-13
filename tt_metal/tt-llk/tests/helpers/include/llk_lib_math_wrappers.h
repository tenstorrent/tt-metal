// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// These wrappers are intended exclusively for LLK tests and are only available
// when the LLK infrastructure is enabled.
#ifdef ENV_LLK_INFRA

#include <cstdint>

#include "experimental/llk_math_reduce_custom.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_transpose_dest.h"

#ifdef ARCH_WORMHOLE

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    [[maybe_unused]] bool tilize   = false,
    bool is_int_fpu_en             = false>
inline void _llk_math_eltwise_unary_datacopy_init_wrapper_(
    const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255, [[maybe_unused]] const bool skip_bh_tilize_workaround = false)
{
    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en>(num_faces, dst_format);
}

template <DataCopyType type, DstSync Dst, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_wrapper_(
    const std::uint32_t dst_index, const std::uint32_t src_format, const std::uint32_t dst_format, [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    _llk_math_eltwise_unary_datacopy_<type, Dst, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(dst_index, src_format, dst_format);
}

template <[[maybe_unused]] bool is_fp32_dest_acc_en, bool transpose_of_faces = true, bool is_32bit = false>
inline void _llk_math_transpose_dest_wrapper_(const std::uint32_t dst_index)
{
    _llk_math_transpose_dest_<transpose_of_faces, is_32bit>(dst_index);
}

inline void _llk_math_reconfig_remap_wrapper_([[maybe_unused]] const bool remap_enable)
{
}

inline bool _llk_math_skip_bh_tilize_workaround_wrapper_([[maybe_unused]] const std::uint32_t unpack_src_format)
{
    return false;
}

template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_reinit_wrapper_()
{
    _llk_math_reduce_block_max_row_init_<block_ct_dim, is_fp32_dest_acc_en>();
}

#elif defined(ARCH_BLACKHOLE)

template <DataCopyType type, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool tilize = false, bool is_int_fpu_en = false>
inline void _llk_math_eltwise_unary_datacopy_init_wrapper_(
    const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255, const bool skip_bh_tilize_workaround = false)
{
    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, tilize, is_int_fpu_en>(
        num_faces, dst_format, skip_bh_tilize_workaround);
}

template <DataCopyType type, DstSync Dst, bool is_fp32_dest_acc_en, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_datacopy_wrapper_(
    const std::uint32_t dst_index, const std::uint32_t src_format, const std::uint32_t dst_format, const std::uint32_t num_faces = 4)
{
    _llk_math_eltwise_unary_datacopy_<type, Dst, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(dst_index, src_format, dst_format, num_faces);
}

template <bool is_fp32_dest_acc_en, bool transpose_of_faces = true, bool is_32bit = false>
inline void _llk_math_transpose_dest_wrapper_(const std::uint32_t dst_index)
{
    _llk_math_transpose_dest_<is_fp32_dest_acc_en, transpose_of_faces, is_32bit>(dst_index);
}

inline void _llk_math_reconfig_remap_wrapper_(const bool remap_enable)
{
    _llk_math_reconfig_remap_(remap_enable);
}

inline bool _llk_math_skip_bh_tilize_workaround_wrapper_(const std::uint32_t unpack_src_format)
{
    return IS_8BIT_FORMAT(unpack_src_format);
}

template <[[maybe_unused]] std::uint32_t block_ct_dim, [[maybe_unused]] bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_reinit_wrapper_()
{
    reduce_max_row_configure_addrmod_reinit_minimal();
}

#else
#error "Unsupported architecture"
#endif

#endif // ENV_LLK_INFRA
