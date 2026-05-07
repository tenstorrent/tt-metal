// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_tilize.h"

using ckernel::FACE_R_DIM;

#ifdef ARCH_WORMHOLE

inline void _llk_unpack_tilize_init_wrapper_(
    const std::uint32_t unpack_src_format = 0 /* unpack_src_format */,
    const std::uint32_t unpack_dst_format = 0 /* unpack_dst_format */,
    const std::uint32_t ct_dim            = 0 /* ct_dim */,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const bool narrow_tile                = false /* narrow_tile */,
    const std::uint32_t num_faces         = 4 /* num_faces */)
{
    (void)num_faces;
    _llk_unpack_tilize_init_(unpack_src_format, unpack_dst_format, ct_dim, face_r_dim, narrow_tile);
}

inline void _llk_unpack_tilize_wrapper_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format = 0 /* unpack_src_format */,
    const std::uint32_t unpack_dst_format = 0 /* unpack_dst_format */,
    const std::uint32_t block_ct_dim      = 0 /* block_ct_dim */,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const std::uint32_t num_faces         = 4 /* num_faces */,
    const bool narrow_tile                = false /* narrow_tile */)
{
    _llk_unpack_tilize_(base_address, tile_index, unpack_src_format, unpack_dst_format, block_ct_dim, face_r_dim, num_faces, narrow_tile);
}

#else // ARCH_BLACKHOLE version of the wrappers

inline void _llk_unpack_tilize_init_wrapper_(
    const std::uint32_t unpack_src_format = 0 /* unpack_src_format */,
    const std::uint32_t unpack_dst_format = 0 /* unpack_dst_format */,
    const std::uint32_t ct_dim            = 0 /* ct_dim */,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const bool narrow_tile                = false /* narrow_tile */,
    const std::uint32_t num_faces         = 4 /* num_faces */)
{
    _llk_unpack_tilize_init_(unpack_src_format, unpack_dst_format, ct_dim, face_r_dim, narrow_tile, num_faces);
}

inline void _llk_unpack_tilize_wrapper_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format = 0 /* unpack_src_format */,
    const std::uint32_t unpack_dst_format = 0 /* unpack_dst_format */,
    const std::uint32_t block_ct_dim      = 0 /* block_ct_dim */,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const std::uint32_t num_faces         = 4 /* num_faces */,
    const bool narrow_tile                = false /* narrow_tile */)
{
    (void)block_ct_dim;
    _llk_unpack_tilize_(base_address, tile_index, unpack_src_format, unpack_dst_format, face_r_dim, num_faces, narrow_tile);
}

#endif
