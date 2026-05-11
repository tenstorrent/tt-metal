// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_tilize.h"

#ifdef ARCH_WORMHOLE

inline void _llk_unpack_tilize_init_wrapper_(
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t ct_dim            = 0,
    const std::uint32_t face_r_dim        = ckernel::FACE_R_DIM,
    const bool narrow_tile                = false,
    const std::uint32_t num_faces         = 4)
{
    (void)num_faces;
    _llk_unpack_tilize_init_(unpack_src_format, unpack_dst_format, ct_dim, face_r_dim, narrow_tile);
}

inline void _llk_unpack_tilize_wrapper_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t block_ct_dim      = 0,
    const std::uint32_t face_r_dim        = ckernel::FACE_R_DIM,
    const std::uint32_t num_faces         = 4,
    const bool narrow_tile                = false)
{
    _llk_unpack_tilize_(base_address, tile_index, unpack_src_format, unpack_dst_format, block_ct_dim, face_r_dim, num_faces, narrow_tile);
}

#elif defined(ARCH_BLACKHOLE)

inline void _llk_unpack_tilize_init_wrapper_(
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t ct_dim            = 0,
    const std::uint32_t face_r_dim        = ckernel::FACE_R_DIM,
    const bool narrow_tile                = false,
    const std::uint32_t num_faces         = 4)
{
    _llk_unpack_tilize_init_(unpack_src_format, unpack_dst_format, ct_dim, face_r_dim, narrow_tile, num_faces);
}

inline void _llk_unpack_tilize_wrapper_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t block_ct_dim      = 0,
    const std::uint32_t face_r_dim        = ckernel::FACE_R_DIM,
    const std::uint32_t num_faces         = 4,
    const bool narrow_tile                = false)
{
    (void)block_ct_dim;
    _llk_unpack_tilize_(base_address, tile_index, unpack_src_format, unpack_dst_format, face_r_dim, num_faces, narrow_tile);
}

#else
#error "Unsupported architecture for LLK unpack tilize wrappers"

#endif
