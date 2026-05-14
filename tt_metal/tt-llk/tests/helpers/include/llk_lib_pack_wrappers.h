// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// These wrappers are intended exclusively for LLK tests and are only available
// when the LLK infrastructure is enabled.
#ifdef ENV_LLK_INFRA

#include <cstdint>

#include "llk_pack.h"

using ckernel::FACE_R_DIM;
using ckernel::TILE_C_DIM;

#ifdef ARCH_WORMHOLE

inline bool _llk_pack_skip_bh_tilize_workaround_wrapper_([[maybe_unused]] const std::uint32_t pack_src_format)
{
    // Wormhole does not need the Blackhole-specific tilize workaround, so the
    // source format does not affect pack configuration in these LLK tests.
    return false;
}

template <bool is_fp32_dest_acc_en, bool untilize = false, [[maybe_unused]] bool tilize = false>
inline void _llk_pack_hw_configure_wrapper_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const bool narrow_tile                          = false,
    const std::uint32_t relu_config                 = 0)
{
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, untilize>(
        pack_src_format, pack_dst_format, tile_size, face_r_dim, num_faces, partial_face, narrow_tile, relu_config);
}

template <bool is_fp32_dest_acc_en, [[maybe_unused]] bool is_tile_dim_reconfig_en = false>
inline void _llk_pack_reconfig_data_format_wrapper_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const bool narrow_tile                          = false,
    [[maybe_unused]] const std::uint32_t num_tiles  = 1)
{
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, face_r_dim, num_faces, partial_face, narrow_tile);
}

template <bool untilize = false, bool zero_output = false, [[maybe_unused]] bool tilize = false>
inline void _llk_pack_init_wrapper_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const bool narrow_tile                          = false,
    const std::uint32_t num_tiles                   = 1)
{
    _llk_pack_init_<untilize, zero_output>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile, num_tiles);
}

template <bool untilize = false, bool zero_output = false, [[maybe_unused]] bool tilize = false>
inline void _llk_pack_init_with_src_wrapper_(
    [[maybe_unused]] const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim                        = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim       = TILE_C_DIM,
    const std::uint32_t num_faces                         = 4,
    const bool partial_face                               = false,
    const bool narrow_tile                                = false,
    const std::uint32_t num_tiles                         = 1,
    [[maybe_unused]] const bool skip_bh_tilize_workaround = false)
{
    _llk_pack_init_<untilize, zero_output>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile, num_tiles);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_dest_init_wrapper_(const std::uint32_t face_r_dim = FACE_R_DIM, const bool narrow_tile = false)
{
    _llk_pack_dest_init_<Dst, is_fp32_dest_acc_en, untilize>(face_r_dim, narrow_tile);
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    [[maybe_unused]] bool dense  = false>
inline void _llk_pack_untilize_init_wrapper_(
    [[maybe_unused]] const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4)
{
    _llk_pack_untilize_init_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(pack_dst_format, face_r_dim, num_faces);
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim        = block_ct_dim,
    bool diagonal                    = false,
    bool narrow_row                  = false,
    std::uint32_t row_num_datums     = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    [[maybe_unused]] bool dense      = false>
inline void _llk_pack_untilize_wrapper_(
    const std::uint32_t address,
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim                 = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t num_faces = 4,
    const std::uint32_t tile_dst_rt_offset         = 0)
{
    _llk_pack_untilize_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums, tile_dst_ct_offset>(
        address, pack_dst_format, face_r_dim, tile_dst_rt_offset);
}

inline void _llk_pack_untilize_uninit_wrapper_([[maybe_unused]] const std::uint32_t pack_src_format, const std::uint32_t face_r_dim = FACE_R_DIM)
{
    _llk_pack_untilize_uninit_(face_r_dim);
}

#elif defined(ARCH_BLACKHOLE)

inline bool _llk_pack_skip_bh_tilize_workaround_wrapper_(const std::uint32_t pack_src_format)
{
    // Blackhole requires the tilize workaround for 8-bit source formats to
    // keep pack behavior aligned with the unpack tilize path used by LLK tests.
    return IS_8BIT_FORMAT(pack_src_format);
}

template <bool is_fp32_dest_acc_en, bool untilize = false, bool tilize = false>
inline void _llk_pack_hw_configure_wrapper_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim          = FACE_R_DIM,
    const std::uint32_t tile_c_dim          = TILE_C_DIM,
    const std::uint32_t num_faces           = 4,
    const bool partial_face                 = false,
    [[maybe_unused]] const bool narrow_tile = false,
    const std::uint32_t relu_config         = 0)
{
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, untilize, tilize>(
        pack_src_format, pack_dst_format, tile_size, face_r_dim, tile_c_dim, num_faces, partial_face, relu_config);
}

template <bool is_fp32_dest_acc_en, [[maybe_unused]] bool is_tile_dim_reconfig_en = false>
inline void _llk_pack_reconfig_data_format_wrapper_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim                 = FACE_R_DIM,
    const std::uint32_t tile_c_dim                 = TILE_C_DIM,
    const std::uint32_t num_faces                  = 4,
    const bool partial_face                        = false,
    [[maybe_unused]] const bool narrow_tile        = false,
    [[maybe_unused]] const std::uint32_t num_tiles = 1)
{
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, face_r_dim, tile_c_dim, num_faces, partial_face);
}

template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_init_wrapper_(
    [[maybe_unused]] const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim           = FACE_R_DIM,
    const std::uint32_t tile_c_dim           = TILE_C_DIM,
    const std::uint32_t num_faces            = 4,
    [[maybe_unused]] const bool partial_face = false,
    [[maybe_unused]] const bool narrow_tile  = false,
    const std::uint32_t num_tiles            = 1)
{
    _llk_pack_init_<untilize, zero_output, tilize>(face_r_dim, tile_c_dim, num_faces, num_tiles);
}

template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_init_with_src_wrapper_(
    const std::uint32_t pack_src_format,
    [[maybe_unused]] const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim           = FACE_R_DIM,
    const std::uint32_t tile_c_dim           = TILE_C_DIM,
    const std::uint32_t num_faces            = 4,
    [[maybe_unused]] const bool partial_face = false,
    [[maybe_unused]] const bool narrow_tile  = false,
    const std::uint32_t num_tiles            = 1,
    const bool skip_bh_tilize_workaround     = false)
{
    _llk_pack_init_<untilize, zero_output, tilize>(pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles, skip_bh_tilize_workaround);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_dest_init_wrapper_([[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const bool narrow_tile = false)
{
    _llk_pack_dest_init_<Dst, is_fp32_dest_acc_en>();
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool diagonal                = false,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense                   = false>
inline void _llk_pack_untilize_init_wrapper_(
    const std::uint32_t pack_src_format,
    [[maybe_unused]] const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4)
{
    static_assert(!diagonal, "Blackhole pack untilize does not support diagonal mode");
    _llk_pack_untilize_init_<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(pack_src_format, pack_dst_format, face_r_dim, num_faces);
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim                     = block_ct_dim,
    bool diagonal                                 = false,
    bool narrow_row                               = false,
    [[maybe_unused]] std::uint32_t row_num_datums = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset              = 0,
    bool dense                                    = false>
inline void _llk_pack_untilize_wrapper_(
    const std::uint32_t address,
    [[maybe_unused]] const std::uint32_t pack_dst_format,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces                   = 4,
    const std::uint32_t tile_dst_rt_offset          = 0)
{
    static_assert(!diagonal, "Blackhole pack untilize does not support diagonal mode");
    _llk_pack_untilize_<block_ct_dim, full_ct_dim, narrow_row, tile_dst_ct_offset, dense>(address, num_faces, tile_dst_rt_offset);
}

inline void _llk_pack_untilize_uninit_wrapper_(const std::uint32_t pack_src_format, [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM)
{
    _llk_pack_untilize_uninit_(pack_src_format);
}

#else
#error "Unsupported architecture"
#endif

#endif // ENV_LLK_INFRA
