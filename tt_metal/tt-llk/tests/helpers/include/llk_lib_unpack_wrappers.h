// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// These wrappers are intended exclusively for LLK tests and are only available
// when the LLK infrastructure is enabled.
#ifdef ENV_LLK_INFRA

#include <cstdint>

#include "llk_unpack_tilize.h"

#ifdef ARCH_WORMHOLE // ARCH_WORMHOLE version of the wrappers

inline void _llk_unpack_tilize_init_wrapper_(
    const std::uint32_t unpack_src_format          = 0,
    const std::uint32_t unpack_dst_format          = 0,
    const std::uint32_t ct_dim                     = 0,
    const std::uint32_t face_r_dim                 = ckernel::FACE_R_DIM,
    const bool narrow_tile                         = false,
    [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    _llk_unpack_tilize_init_(unpack_src_format, unpack_dst_format, ct_dim, face_r_dim, narrow_tile);
}

inline std::uint32_t _llk_unpack_tilize_block_ct_dim_wrapper_(const std::uint32_t block_ct_dim)
{
    // Wormhole uses block_ct_dim to compute tilize source offsets.
    return block_ct_dim;
}

inline std::uint32_t _llk_unpack_tilize_num_faces_wrapper_(const std::uint32_t num_faces)
{
    // Wormhole uses num_faces to select the tilize loop count.
    return num_faces;
}

inline std::uint32_t _llk_unpack_tilize_num_dvalids_wrapper_(const std::uint32_t tile_count, const std::uint32_t tile_num_faces)
{
    // Wormhole tracks dvalids per tile face.
    return tile_count * tile_num_faces;
}

inline void _llk_unpack_tilize_wrapper_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format = 0,
    const std::uint32_t unpack_dst_format = 0,
    const std::uint32_t block_ct_dim      = 0,
    const std::uint32_t face_r_dim        = FACE_R_DIM,
    const std::uint32_t num_faces         = 4,
    const bool narrow_tile                = false)
{
    _llk_unpack_tilize_(base_address, tile_index, unpack_src_format, unpack_dst_format, block_ct_dim, face_r_dim, num_faces, narrow_tile);
}

inline void _llk_unpack_tilize_uninit_wrapper_(
    const std::uint32_t unpack_dst_format, [[maybe_unused]] const std::uint32_t num_faces = 4, const std::uint32_t face_r_dim = FACE_R_DIM)
{
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

inline std::uint32_t _llk_unpack_tilize_block_ct_dim_wrapper_([[maybe_unused]] const std::uint32_t block_ct_dim)
{
    // Blackhole unpack_tilize does not use block_ct_dim and expects zero.
    return 0;
}

inline std::uint32_t _llk_unpack_tilize_num_faces_wrapper_([[maybe_unused]] const std::uint32_t num_faces)
{
    // Blackhole tests keep unpack_tilize on the default 4-face path.
    return 4;
}

inline std::uint32_t _llk_unpack_tilize_num_dvalids_wrapper_(const std::uint32_t tile_count, [[maybe_unused]] const std::uint32_t tile_num_faces)
{
    // Blackhole tracks one dvalid per tile for unpack_tilize.
    return tile_count;
}

inline void _llk_unpack_tilize_wrapper_(
    const std::uint32_t base_address,
    const std::uint32_t tile_index,
    const std::uint32_t unpack_src_format             = 0,
    const std::uint32_t unpack_dst_format             = 0,
    [[maybe_unused]] const std::uint32_t block_ct_dim = 0,
    const std::uint32_t face_r_dim                    = ckernel::FACE_R_DIM,
    const std::uint32_t num_faces                     = 4,
    const bool narrow_tile                            = false)
{
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

#endif // ENV_LLK_INFRA
