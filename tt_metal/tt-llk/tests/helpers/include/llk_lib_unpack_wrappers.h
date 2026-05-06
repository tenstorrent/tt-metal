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

inline std::uint32_t _llk_unpack_tilize_block_ct_dim_wrapper_(const std::uint32_t block_ct_dim)
{
    return block_ct_dim;
}

inline std::uint32_t _llk_unpack_tilize_num_faces_wrapper_(const std::uint32_t num_faces)
{
    return num_faces;
}

inline std::uint32_t _llk_unpack_tilize_num_dvalids_wrapper_(const std::uint32_t tile_count, const std::uint32_t tile_num_faces)
{
    return tile_count * tile_num_faces;
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

inline void _llk_unpack_tilize_uninit_wrapper_(
    const std::uint32_t unpack_dst_format, const std::uint32_t num_faces = 4, const std::uint32_t face_r_dim = ckernel::FACE_R_DIM)
{
    (void)num_faces;
    _llk_unpack_tilize_uninit_(unpack_dst_format, face_r_dim);
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

inline std::uint32_t _llk_unpack_tilize_block_ct_dim_wrapper_(const std::uint32_t block_ct_dim)
{
    (void)block_ct_dim;
    return 0;
}

inline std::uint32_t _llk_unpack_tilize_num_faces_wrapper_(const std::uint32_t num_faces)
{
    (void)num_faces;
    return 4;
}

inline std::uint32_t _llk_unpack_tilize_num_dvalids_wrapper_(const std::uint32_t tile_count, const std::uint32_t tile_num_faces)
{
    (void)tile_num_faces;
    return tile_count;
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

inline void _llk_unpack_tilize_uninit_wrapper_(
    const std::uint32_t unpack_dst_format, const std::uint32_t num_faces = 4, const std::uint32_t face_r_dim = ckernel::FACE_R_DIM)
{
    _llk_unpack_tilize_uninit_(unpack_dst_format, num_faces, face_r_dim);
}

#else
#error "Unsupported architecture for LLK unpack tilize wrappers"
#endif
