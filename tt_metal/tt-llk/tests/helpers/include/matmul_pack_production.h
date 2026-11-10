// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// The block pack of a matmul output as llk_pack_block (hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h) runs it with the W offset on, written with the
// same _llk_ calls in the same order, because a test kernel cannot include the metal API. The form of a block (T per tile, B the block that programs the output, W the
// block with the W offset) is llk_pack_block_form with the matmul_kt overlay, generated into matmul_pack_tables.h from the API (gen_matmul_pack_tables.py, checked by
// hardware_free/kernels/test_matmul_pack_tables.py); the first block of the run takes the init table, the next ones the next table. MATMUL_PACK_FORM 1 is the plain
// B block, 2 the closed one (the keyed matmul form picks the block shape). Keep the rest equal to llk_pack_block and llk_pack_w_take. MATMUL_PACK_STATE makes every block the first of its output: 2 the init table, 1 the reconfigure table.

#include <cstdint>
#if defined(ARCH_BLACKHOLE)
#include "llk_pack.h"
#include "matmul_pack_tables.h"
#ifndef MATMUL_PACK_STATE
#define MATMUL_PACK_STATE 0
#endif

namespace matmul_pack_production
{
using namespace ckernel;

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

constexpr bool src32_format(const DataFormat f)
{
    return f == DataFormat::Float32 || f == DataFormat::Int32 || f == DataFormat::UInt32;
}

constexpr bool bfp_format(const DataFormat f)
{
    return f == DataFormat::Bfp8 || f == DataFormat::Bfp8_b || f == DataFormat::Bfp4 || f == DataFormat::Bfp4_b || f == DataFormat::Bfp2 || f == DataFormat::Bfp2_b;
}

#if defined(MATMUL_PACK_KEYS)
constexpr std::uint32_t key_nt  = MATMUL_PK_RT * MATMUL_PK_CT;
constexpr std::uint32_t key_kt  = MATMUL_PK_KT;
constexpr std::uint32_t key_r0  = MATMUL_PK_R0;
constexpr std::uint32_t key_c1  = MATMUL_PK_C1;
constexpr bool key_src32        = src32_format(MATMUL_PK_IN_A) || src32_format(MATMUL_PK_IN_B) || src32_format(MATMUL_PK_OUT);
#else
constexpr std::uint32_t key_nt = 0;
constexpr std::uint32_t key_kt = 0;
constexpr std::uint32_t key_r0 = 32;
constexpr std::uint32_t key_c1 = 32;
constexpr bool key_src32       = false;
#endif

// address in 16 byte words, tile_stride in bytes (the page size times 16)
template <DstSync sync, bool fp32, int form, DataFormat out_format>
inline void block(std::uint32_t tile_index, std::uint32_t address, std::uint32_t num_tiles, std::uint32_t tile_stride, std::uint32_t num_faces = 4)
{
    const std::uint32_t page = tile_stride >> 4;
    constexpr std::uint32_t face_rows = key_r0 < 16 ? key_r0 : 16;
    constexpr std::uint32_t faces     = (key_r0 > 16 ? 2 : 1) * (key_c1 > 16 ? 2 : 1);
    constexpr std::uint32_t row       = matmul_pack_tables::llk_pack_block_class_k(
        face_rows,
        faces,
        out_format == DataFormat::Bfp4 || out_format == DataFormat::Bfp2,
        bfp_format(out_format),
        out_format == DataFormat::Bfp8_b || out_format == DataFormat::Bfp8,
        wide_format(out_format));
    constexpr bool int8     = out_format == DataFormat::Int8 || out_format == DataFormat::UInt8;
    constexpr bool src32    = fp32 && key_src32;
    const bool first        = MATMUL_PACK_STATE != 0 || !state.seen;
    const char f = matmul_pack_tables::llk_pack_block_form_k<fp32, key_kt>(
        row,
        key_nt != 0 ? key_nt : num_tiles,
        first ? (MATMUL_PACK_STATE == 1 ? (src32 ? matmul_pack_tables::llk_pack_reconf_form : matmul_pack_tables::llk_pack_reconf_form_dc)
                                        : (src32 ? matmul_pack_tables::llk_pack_init_form : matmul_pack_tables::llk_pack_init_form_dc))
              : (src32 ? matmul_pack_tables::llk_pack_next_form : matmul_pack_tables::llk_pack_next_form_dc),
        !first,
        sync == DstSync::SyncFull,
        src32,
        int8);
    state.seen = true;
    if (first)
    {
        _llk_pack_close_fast_untilize_<sync>();
    }
    if (f == 'T')
    {
        state.valid = false;
        for (std::uint32_t i = 0; i < num_tiles; i++)
        {
            _llk_pack_<sync, fp32, ckernel::PackMode::Default>(tile_index + i, address + i * page);
        }
        return;
    }
    if (f != 'W')
    {
        state.valid = false;
        _llk_pack_close_fast_untilize_<sync>();
        if constexpr (form == 1)
        {
            _llk_pack_block_<sync, fp32, ckernel::PackMode::Default>(tile_index, address, num_tiles);
        }
        else
        {
            _llk_pack_block_closed_<sync, fp32>(tile_index, address, num_tiles, tile_stride, num_faces);
        }
        return;
    }
    constexpr std::uint32_t max_tiles = wide_format(out_format) ? 64 : 127; // LLK_PACK_W_MAX_TILES_W
    const bool cont = state.valid && state.page == page && state.next == address && state.tiles + num_tiles <= max_tiles;
    if (!cont)
    {
        // llk_pack_w_take: the output base is programmed again, the stride only when it changed
        _llk_pack_close_fast_untilize_<sync>();
        program_packer_destination<false>(address);
        if (tile_stride != pack_ch1_w_stride)
        {
            cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Wstride_RMW>(tile_stride);
            pack_ch1_w_stride = tile_stride;
        }
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

// Every block of the output takes the per-tile form (the first, after a reconfigure, the next ones): the kernel packs per tile, as llk_matmul_block_pack decides
// (llk_matmul_pack_always_per_tile). Needs the block keys.
template <DstSync sync, bool fp32, DataFormat out_format>
constexpr bool always_per_tile()
{
    if constexpr (key_nt == 0)
    {
        return false;
    }
    else
    {
        using namespace matmul_pack_tables;
        constexpr std::uint32_t face_rows = key_r0 < 16 ? key_r0 : 16;
        constexpr std::uint32_t faces     = (key_r0 > 16 ? 2 : 1) * (key_c1 > 16 ? 2 : 1);
        constexpr std::uint32_t row       = llk_pack_block_class_k(
            face_rows,
            faces,
            out_format == DataFormat::Bfp4 || out_format == DataFormat::Bfp2,
            bfp_format(out_format),
            out_format == DataFormat::Bfp8_b || out_format == DataFormat::Bfp8,
            wide_format(out_format));
        constexpr bool int8  = out_format == DataFormat::Int8 || out_format == DataFormat::UInt8;
        constexpr bool src32 = fp32 && key_src32;
        constexpr bool full  = sync == DstSync::SyncFull;
        return llk_pack_block_form_k<fp32, key_kt>(row, key_nt, src32 ? llk_pack_init_form : llk_pack_init_form_dc, false, full, src32, int8) == 'T' &&
               llk_pack_block_form_k<fp32, key_kt>(row, key_nt, src32 ? llk_pack_reconf_form : llk_pack_reconf_form_dc, false, full, src32, int8) == 'T' &&
               llk_pack_block_form_k<fp32, key_kt>(row, key_nt, src32 ? llk_pack_next_form : llk_pack_next_form_dc, true, full, src32, int8) == 'T';
    }
}
} // namespace matmul_pack_production
#endif
