// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_pack.h"

using ckernel::FACE_R_DIM;
using ckernel::TILE_C_DIM;

#ifdef ARCH_WORMHOLE

template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_init_wrapper_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false,
    const std::uint32_t num_tiles  = 1)
{
    (void)tile_c_dim;
    (void)tilize;
    _llk_pack_init_<untilize, zero_output>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile, num_tiles);
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim        = block_ct_dim,
    bool diagonal                    = false,
    bool narrow_row                  = false,
    std::uint32_t row_num_datums     = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense                       = false>
inline void _llk_pack_untilize_wrapper_(
    const std::uint32_t address,
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim         = FACE_R_DIM,
    const std::uint32_t num_faces          = 4,
    const std::uint32_t tile_dst_rt_offset = 0)
{
    (void)num_faces;
    (void)dense;
    _llk_pack_untilize_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums, tile_dst_ct_offset>(
        address, pack_dst_format, face_r_dim, tile_dst_rt_offset);
}

#else // ARCH_BLACKHOLE version of the wrappers

template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_init_wrapper_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false,
    const std::uint32_t num_tiles  = 1)
{
    (void)pack_dst_format;
    (void)partial_face;
    (void)narrow_tile;
    _llk_pack_init_<untilize, zero_output, tilize>(face_r_dim, tile_c_dim, num_faces, num_tiles);
}

template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim        = block_ct_dim,
    bool diagonal                    = false,
    bool narrow_row                  = false,
    std::uint32_t row_num_datums     = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense                       = false>
inline void _llk_pack_untilize_wrapper_(
    const std::uint32_t address,
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim         = FACE_R_DIM,
    const std::uint32_t num_faces          = 4,
    const std::uint32_t tile_dst_rt_offset = 0)
{
    static_assert(!diagonal, "Blackhole pack untilize does not support diagonal mode");
    (void)pack_dst_format;
    (void)face_r_dim;
    (void)row_num_datums;
    _llk_pack_untilize_<block_ct_dim, full_ct_dim, narrow_row, tile_dst_ct_offset, dense>(address, num_faces, tile_dst_rt_offset);
}

#endif
