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
#include "llk_math_reduce.h"
#include "llk_math_transpose_dest.h"
#include "tensor_shape.h"

using ckernel::PackMode;

template <bool untilize, bool tilize>
inline constexpr PackMode llk_test_pack_mode_v = untilize ? PackMode::Untilize
                                                 : tilize ? PackMode::Tilize
                                                          : PackMode::Default;

#ifdef ARCH_WORMHOLE

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type      = BroadcastType::NONE,
    bool is_int_fpu_en                  = false,
    [[maybe_unused]] PackMode pack_mode = PackMode::Default>
inline void _llk_math_eltwise_unary_datacopy_init_wrapper_(const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize || pack_mode == PackMode::Tilize,
        "Wormhole B0 LLK tests: math datacopy init wrapper accepts PackMode::Default, PackMode::Untilize, or PackMode::Tilize (tilize is ignored on WH)");
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

template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_reinit_wrapper_(const ckernel::TensorShape& tensor_shape)
{
    _llk_math_reduce_block_max_row_init_<block_ct_dim, is_fp32_dest_acc_en>(tensor_shape);
}

#elif defined(ARCH_BLACKHOLE)

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en             = false,
    PackMode pack_mode             = PackMode::Default>
inline void _llk_math_eltwise_unary_datacopy_init_wrapper_(const std::uint32_t num_faces = 4, const std::uint32_t dst_format = 255)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Tilize,
        "Blackhole LLK tests: math datacopy init wrapper supports PackMode::Default or PackMode::Tilize");
    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en, pack_mode>(num_faces, dst_format);
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
    _llk_math_transpose_dest_<transpose_of_faces, is_32bit>(dst_index);
}

inline void _llk_math_reconfig_remap_wrapper_(const bool remap_enable)
{
    _llk_math_reconfig_remap_(remap_enable);
}

template <[[maybe_unused]] std::uint32_t block_ct_dim, [[maybe_unused]] bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_reinit_wrapper_([[maybe_unused]] const ckernel::TensorShape& tensor_shape)
{
    reduce_max_row_configure_addrmod_reinit_minimal();
}

#else
#error "Unsupported architecture"
#endif

#endif // ENV_LLK_INFRA
