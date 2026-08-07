// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// These wrappers are intended exclusively for LLK tests and are only available
// when the LLK infrastructure is enabled.
#ifdef ENV_LLK_INFRA

#include <cstdint>

#include "llk_pack.h"
#include "llk_pack_untilize.h"

using ckernel::FACE_R_DIM;
using ckernel::PackMode;
using ckernel::TILE_C_DIM;

/// Maps legacy LLK test (untilize, tilize) constexpr flags to a single \ref PackMode for wrapper templates.
template <bool untilize, bool tilize>
inline constexpr PackMode llk_test_pack_mode_v = untilize ? PackMode::Untilize
                                                 : tilize ? PackMode::Tilize
                                                          : PackMode::Default;

/// Legacy ``untilize`` flag for `_llk_pack_` (Default vs Untilize only).
template <bool untilize>
inline constexpr PackMode pack_exec_mode_v = untilize ? PackMode::Untilize : PackMode::Default;

#ifdef ARCH_WORMHOLE

inline bool _llk_pack_skip_bh_tilize_workaround_wrapper_([[maybe_unused]] const std::uint32_t pack_src_format)
{
    // Wormhole does not need the Blackhole-specific tilize workaround, so the
    // source format does not affect pack configuration in these LLK tests.
    return false;
}

/// Pack configure/init \ref PackMode for unpack-tilize sweep-style tests. Wormhole B0 pack does not support
/// \c PackMode::Tilize in \c configure_pack / \c _llk_pack_init_; Blackhole uses \ref llk_test_pack_mode_v.
template <[[maybe_unused]] bool untilize, [[maybe_unused]] bool tilize>
inline constexpr PackMode llk_unpack_tilize_sweep_pack_cfg_mode_v = PackMode::Default;

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
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
    static_assert(pack_mode != PackMode::Tilize, "Wormhole B0 LLK tests: pack hw configure supports PackMode::Default or PackMode::Untilize only");
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, pack_mode>(
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

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_init_wrapper_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const bool narrow_tile                          = false,
    const std::uint32_t num_tiles                   = 1)
{
    static_assert(pack_mode != PackMode::Tilize, "Wormhole B0 LLK tests: pack init supports PackMode::Default or PackMode::Untilize only");
    _llk_pack_init_<pack_mode, zero_output>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile, num_tiles);
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
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
    static_assert(pack_mode != PackMode::Tilize, "Wormhole B0 LLK tests: pack init supports PackMode::Default or PackMode::Untilize only");
    _llk_pack_init_<pack_mode, zero_output>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile, num_tiles);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_dest_init_wrapper_(const std::uint32_t face_r_dim = FACE_R_DIM, const bool narrow_tile = false)
{
    static_assert(pack_mode != PackMode::Tilize, "Wormhole B0 LLK tests: pack dest init supports PackMode::Default or PackMode::Untilize only");
    _llk_pack_dest_init_<Dst, is_fp32_dest_acc_en, pack_mode>(face_r_dim, narrow_tile);
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

inline void _llk_pack_untilize_uninit_wrapper_(
    [[maybe_unused]] const std::uint32_t pack_src_format, [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM)
{
    _llk_pack_untilize_uninit_();
}

#elif defined(ARCH_BLACKHOLE)

inline bool _llk_pack_skip_bh_tilize_workaround_wrapper_(const std::uint32_t pack_src_format)
{
    // Blackhole requires the tilize workaround for 8-bit source formats to
    // keep pack behavior aligned with the unpack tilize path used by LLK tests.
    return IS_8BIT_FORMAT(pack_src_format);
}

/// Pack configure/init \ref PackMode for unpack-tilize sweep-style tests (maps legacy untilize/tilize flags).
template <bool untilize, bool tilize>
inline constexpr PackMode llk_unpack_tilize_sweep_pack_cfg_mode_v = llk_test_pack_mode_v<untilize, tilize>;

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

template <bool is_fp32_dest_acc_en, [[maybe_unused]] bool is_tile_dim_reconfig_en = false>
inline void _llk_pack_reconfig_data_format_wrapper_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim                  = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    [[maybe_unused]] const bool narrow_tile         = false,
    [[maybe_unused]] const std::uint32_t num_tiles  = 1)
{
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, tile_c_dim, num_faces, partial_face);
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
    llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        pack_dst_format /* pack_src_format: ignored when strides are skipped */, face_r_dim, tile_c_dim, num_faces, num_tiles);
}

template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
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
    _llk_pack_init_<pack_mode, zero_output>(pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles, skip_bh_tilize_workaround);
}

template <DstSync Dst, bool is_fp32_dest_acc_en, [[maybe_unused]] PackMode pack_mode = PackMode::Default>
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
