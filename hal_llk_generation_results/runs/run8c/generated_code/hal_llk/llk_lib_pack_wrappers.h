// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test wrappers (same interface as tests/helpers/include/llk_lib_pack_wrappers.h, Blackhole branch)
// on top of the HAL-based pack implementation.

#pragma once

#include <cstdint>

#include "llk_pack.h"

using ckernel::FACE_R_DIM;
using ckernel::PackMode;
using ckernel::TILE_C_DIM;

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
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
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, pack_mode>(
        pack_src_format, pack_dst_format, tile_size, face_r_dim, tile_c_dim, num_faces, partial_face, relu_config);
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_init_wrapper_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim           = FACE_R_DIM,
    const std::uint32_t tile_c_dim           = TILE_C_DIM,
    const std::uint32_t num_faces            = 4,
    [[maybe_unused]] const bool partial_face = false,
    [[maybe_unused]] const bool narrow_tile  = false,
    const std::uint32_t num_tiles            = 1)
{
    // Strides are owned by the preceding hw-configure, so the source format is not needed here.
    llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, false /* skip_addrmod_config */, true /* skip_packer_strides */, false /* mutex_ADC */>(
        pack_dst_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_dest_init_wrapper_([[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const bool narrow_tile = false)
{
    _llk_pack_dest_init_<Dst, is_fp32_dest_acc_en>();
}
