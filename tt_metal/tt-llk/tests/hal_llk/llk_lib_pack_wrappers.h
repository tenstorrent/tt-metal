// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Reduced copy of tests/helpers/include/llk_lib_pack_wrappers.h for the HAL port (Blackhole only): only the wrappers
// the eltwise binary test uses. The original includes production llk_pack_untilize.h, which would pull the production
// llk_pack.h in next to the HAL port.

// These wrappers are intended exclusively for LLK tests and are only available
// when the LLK infrastructure is enabled.
#ifdef ENV_LLK_INFRA

#include <cstdint>

#include "llk_pack.h"

using ckernel::FACE_R_DIM;
using ckernel::PackMode;
using ckernel::TILE_C_DIM;

#ifdef ARCH_BLACKHOLE

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
    // No-src wrapper: the packer strides are owned by the preceding hw-configure/reconfig, so we skip
    // them here. Because strides are the only consumer of pack_src_format, we call the internal init
    // helper directly instead of the public _llk_pack_init_: that avoids fabricating a pack_src_format
    // (the src format is never read once strides are skipped) and bypasses the public entry point's
    // format-based num_tiles assert, which would otherwise validate against a placeholder value.
    llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, false /* skip_addrmod_config */, true /* skip_packer_strides */, false /* mutex_ADC */>(
        pack_dst_format /* pack_src_format: ignored when strides are skipped */, face_r_dim, tile_c_dim, num_faces, num_tiles);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode /*maybe_unused*/ = PackMode::Default>
inline void _llk_pack_dest_init_wrapper_([[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM, [[maybe_unused]] const bool narrow_tile = false)
{
    _llk_pack_dest_init_<Dst, is_fp32_dest_acc_en>();
}

#else
#error "Unsupported architecture"
#endif

#endif // ENV_LLK_INFRA
