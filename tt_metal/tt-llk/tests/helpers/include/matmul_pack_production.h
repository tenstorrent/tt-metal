// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// The block pack of a matmul output as llk_pack_block (hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h) runs it with the W offset on, written with the
// same _llk_ calls in the same order, because a test kernel cannot include the metal API: the first block of an output programs the output (one tile at a time for
// a 32-bit output of up to 8 tiles and for a 4 tile block-float block, otherwise the plain or the closed block); the second block programs the output base and the
// tile stride once (_llk_pack_block_w_init_, llk_pack_w_take); a block that continues the last one only moves the base with the packer W counter. MATMUL_PACK_FORM 1
// is the plain block, 2 the closed block. Keep it equal to llk_pack_block and llk_pack_w_take.

#if defined(ARCH_BLACKHOLE)
#include "llk_pack.h"

namespace matmul_pack_production
{
constexpr std::uint32_t W_MAX_TILES = 127; // LLK_PACK_W_MAX_TILES

struct State
{
    bool seen            = false; // pack_w_last == the output
    bool valid           = false;
    std::uint32_t page   = 0;
    std::uint32_t next   = 0;
    std::uint32_t tiles  = 0;
};

inline State state;

constexpr bool wide_format(const DataFormat f)
{
    return f == DataFormat::Float32 || f == DataFormat::Int32 || f == DataFormat::UInt32 || f == DataFormat::Tf32;
}

constexpr bool bfp_format(const DataFormat f)
{
    return f == DataFormat::Bfp8 || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4 || f == DataFormat::Bfp4_b || f == DataFormat::Bfp2 || f == DataFormat::Bfp2_b;
}

// address in 16 byte words, tile_stride in bytes (the page size times 16)
template <DstSync sync, bool fp32, int form, DataFormat out_format>
inline void block(std::uint32_t tile_index, std::uint32_t address, std::uint32_t num_tiles, std::uint32_t tile_stride, std::uint32_t num_faces = 4)
{
    const std::uint32_t page = tile_stride >> 4;
    if (!state.seen)
    {
        state.seen = true;
        if ((num_tiles <= 8 && wide_format(out_format)) || (num_tiles == 4 && bfp_format(out_format)))
        {
            for (std::uint32_t i = 0; i < num_tiles; i++)
            {
                _llk_pack_<sync, fp32, ckernel::PackMode::Default>(tile_index + i, address + i * page);
            }
        }
        else if constexpr (form == 1)
        {
            _llk_pack_block_<sync, fp32, ckernel::PackMode::Default>(tile_index, address, num_tiles);
        }
        else
        {
            _llk_pack_block_closed_<sync, fp32>(tile_index, address, num_tiles, tile_stride, num_faces);
        }
        return;
    }
    const bool cont = state.valid && state.page == page && state.next == address && state.tiles + num_tiles <= W_MAX_TILES;
    if (!cont)
    {
        _llk_pack_block_w_init_(address, tile_stride);
        state.valid = true;
        state.page  = page;
        state.next  = address;
        state.tiles = 0;
    }
    const std::uint32_t out_tile = state.tiles;
    state.tiles += num_tiles;
    state.next += page * num_tiles;
    if constexpr (form == 1)
    {
        _llk_pack_block_w_<sync, fp32>(tile_index, out_tile, num_tiles);
    }
    else
    {
        _llk_pack_block_closed_w_mop_<sync, fp32>(tile_index, out_tile, num_tiles, num_faces);
    }
}
} // namespace matmul_pack_production
#endif
