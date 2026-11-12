// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// The block pack of a matmul output as llk_pack_block (hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h) runs it with the W offset on, written with the
// same _llk_ calls in the same order, because a test kernel cannot include the metal API. The form of a block (T per tile, B the block that programs the output, W the
// block with the W offset) is llk_pack_block_form with the matmul_kt overlay, generated into matmul_pack_tables.h from the API (gen_matmul_pack_tables.py, checked by
// hardware_free/kernels/test_matmul_pack_tables.py); the first block of the run takes the init table, the next ones the next table. MATMUL_PACK_FORM 1 is the plain
// B block, 2 the closed one (the keyed matmul form picks the block shape). Keep the rest equal to llk_pack_block and llk_pack_w_take. MATMUL_PACK_STATE makes every block the first of its output: 2 the init table, 1 the reconfigure table. MATMUL_PACK_NOK packs without the K (pack_block's default), the agreement keeps the K.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#if defined(ARCH_BLACKHOLE)
#include "llk_pack.h"
#include "matmul_pack_tables.h"
#ifndef MATMUL_PACK_STATE
#define MATMUL_PACK_STATE 0
#endif
#ifndef MATMUL_PACK_NOK
#define MATMUL_PACK_NOK 0
#endif

namespace matmul_pack_production
{
using namespace ckernel;

struct State
{
    std::uint32_t last   = 0; // the state half of llk_pack_block: 0 none, 1 a block of the output, 2 a W offset block (LLK_PACK_STATE_W); words: a byte store may alias anything and force reloads (dest_offset_id)
    std::uint32_t valid  = 0;
    std::uint32_t page   = 0;
    std::uint32_t next   = 0;
    std::uint32_t tiles  = 0;
};

// internal: the no-LTO test build keeps a write-only global with external linkage, production drops write-only pack state
static State state;

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

// llk_pack_block_forms of one context: the forms a block takes from the init, reconfigure and next states, as a mask (1 per tile, 2 block, 4 W offset)
template <bool fp32, std::uint32_t kt>
constexpr std::uint32_t forms_ctx(const std::uint32_t row, const std::uint32_t nt, const bool t32, const bool full, const bool src32, const bool int8, const std::uint32_t ctx)
{
    using namespace matmul_pack_tables;
    const char a = llk_pack_block_form_k<fp32, kt>(row, nt, t32 ? llk_pack_init_form : llk_pack_init_form_dc, 0, full, src32, int8, ctx);
    const char b = llk_pack_block_form_k<fp32, kt>(row, nt, t32 ? llk_pack_reconf_form : llk_pack_reconf_form_dc, 1, full, src32, int8, ctx);
    const char c = llk_pack_block_form_k<fp32, kt>(row, nt, t32 ? llk_pack_next_form : llk_pack_next_form_dc, 2, full, src32, int8, ctx);
    return (a == 'T' || b == 'T' || c == 'T' ? 1 : 0) | (a == 'B' || b == 'B' || c == 'B' ? 2 : 0) | (a == 'W' || b == 'W' || c == 'W' ? 4 : 0);
}

// llk_pack_block's fold over the contexts a production matmul kernel reaches: 0 (forms without the K), 1, and 2 (the bmm kernels include the
// pack-thread SFPU headers); a block per tile in one is per tile in all
template <bool fp32, std::uint32_t kt>
constexpr std::uint32_t forms_k(const std::uint32_t row, const std::uint32_t nt, const bool t32, const bool full, const bool src32, const bool int8)
{
    const std::uint32_t any = forms_ctx<fp32, 0>(row, nt, t32, full, src32, int8, 0u) | forms_ctx<fp32, kt>(row, nt, t32, full, src32, int8, 1u) |
                              forms_ctx<fp32, kt>(row, nt, t32, full, src32, int8, 2u);
    return (any & 1) != 0 ? 1 : any;
}

// forms_k of tile counts 1 to 16 at compile time: a constant count reads a constant before the IPA passes, as llk_pack_form_words
template <bool fp32, std::uint32_t kt, std::uint32_t row, bool t32, bool full, bool src32, bool int8>
struct forms_table_t
{
    std::uint8_t f[17];
    constexpr forms_table_t() : f()
    {
        for (std::uint32_t n = 1; n <= 16; n++)
        {
            f[n] = static_cast<std::uint8_t>(forms_k<fp32, kt>(row, n, t32, full, src32, int8));
        }
    }
};
template <bool fp32, std::uint32_t kt, std::uint32_t row, bool t32, bool full, bool src32, bool int8>
inline constexpr forms_table_t<fp32, kt, row, t32, full, src32, int8> forms_table {};

// llk_pack_form_words: the form of every tile count of a row from each state (0 init, 1 reconfigure, 2 next), two bits per count (T 0, B 1,
// W 2), in context 1 (a matmul init, no pack-thread SFPU); a variable block reads one word instead of the form tables and masks
template <bool fp32, std::uint32_t kt, std::uint32_t row, bool t32, bool full, bool src32, bool int8>
struct form_words_t
{
    std::uint32_t w[3];
    constexpr form_words_t() : w()
    {
        using namespace matmul_pack_tables;
        for (std::uint32_t st = 0; st < 3; st++)
        {
            for (std::uint32_t n = 1; n <= 16; n++)
            {
                const char f = llk_pack_block_form_k<fp32, kt>(
                    row,
                    n,
                    st == 0 ? (t32 ? llk_pack_init_form : llk_pack_init_form_dc) : (st == 1 ? (t32 ? llk_pack_reconf_form : llk_pack_reconf_form_dc) : (t32 ? llk_pack_next_form : llk_pack_next_form_dc)),
                    st,
                    full,
                    src32,
                    int8,
                    1u);
                w[st] |= static_cast<std::uint32_t>(f == 'T' ? 0 : (f == 'B' ? 1 : 2)) << (2 * (n - 1));
            }
        }
    }
};
template <bool fp32, std::uint32_t kt, std::uint32_t row, bool t32, bool full, bool src32, bool int8>
inline constexpr form_words_t<fp32, kt, row, t32, full, src32, int8> form_words {};

// tile i of a per-tile pack: the caller's address of it (the main arm's), else address + i * page
template <typename TileAddr>
__attribute__((always_inline)) inline std::uint32_t tile_address(const TileAddr& tile_addr, const std::uint32_t address, const std::uint32_t page, const std::uint32_t i)
{
    if constexpr (std::is_same_v<TileAddr, std::nullptr_t>)
    {
        return address + i * page;
    }
    else
    {
        return tile_addr(i);
    }
}

// llk_pack_block_forms of a tile count: 1 is a block that packs per tile from every state
template <DstSync sync, bool fp32, DataFormat out_format>
constexpr std::uint32_t forms_const(const std::uint32_t num_tiles)
{
    constexpr std::uint32_t face_rows = key_r0 < 16 ? key_r0 : 16;
    constexpr std::uint32_t faces     = (key_r0 > 16 ? 2 : 1) * (key_c1 > 16 ? 2 : 1);
    constexpr std::uint32_t row       = matmul_pack_tables::llk_pack_block_class_k(face_rows, faces, static_cast<std::uint32_t>(out_format));
    constexpr bool int8     = out_format == DataFormat::Int8 || out_format == DataFormat::UInt8;
    constexpr bool src32    = fp32 && key_src32;
    constexpr bool tables32 = src32 && unpack_to_dest;
    return num_tiles <= 16 ? forms_table<fp32, MATMUL_PACK_NOK ? 0 : key_kt, row, tables32, sync == DstSync::SyncFull, src32, int8>.f[num_tiles]
                           : forms_k<fp32, MATMUL_PACK_NOK ? 0 : key_kt>(row, num_tiles, tables32, sync == DstSync::SyncFull, src32, int8);
}

// the forms of a constant tile count, 0 for a variable one
template <DstSync sync, bool fp32, DataFormat out_format>
__attribute__((always_inline)) inline std::uint32_t folded_forms(const std::uint32_t num_tiles)
{
    return __builtin_constant_p(num_tiles) ? forms_const<sync, fp32, out_format>(num_tiles) : 0;
}

// address in 16 byte words, tile_stride in bytes (the page size times 16)
template <DstSync sync, bool fp32, int form, DataFormat out_format, typename TileAddr = std::nullptr_t>
__attribute__((always_inline)) inline void block(
    std::uint32_t tile_index, std::uint32_t address, std::uint32_t num_tiles, std::uint32_t tile_stride, std::uint32_t num_faces = 4, TileAddr tile_addr = nullptr)
{
    const std::uint32_t page = tile_stride >> 4;
    constexpr std::uint32_t face_rows = key_r0 < 16 ? key_r0 : 16;
    constexpr std::uint32_t faces     = (key_r0 > 16 ? 2 : 1) * (key_c1 > 16 ? 2 : 1);
    constexpr std::uint32_t row       = matmul_pack_tables::llk_pack_block_class_k(face_rows, faces, static_cast<std::uint32_t>(out_format));
    constexpr bool int8     = out_format == DataFormat::Int8 || out_format == DataFormat::UInt8;
    constexpr bool src32    = fp32 && key_src32;
    constexpr bool tables32 = src32 && unpack_to_dest; // llk_pack_form_table: the 32 bit input tables only when a CB unpacks to DEST
    // a count passed as a function argument (pack_block(dst, cb, n)) is main's llk_matmul_pack loop: per tile, its packs invalidate the W base
    if (!__builtin_constant_p(num_tiles))
    {
        state.valid = 0;
        for (std::uint32_t i = 0; i < num_tiles; i++)
        {
            _llk_pack_<sync, fp32, ckernel::PackMode::Default>(tile_index + i, tile_address(tile_addr, address, page, i));
        }
        return;
    }
    // pack_block<ntiles>: the forms of the three states fold at compile time; a block that is T from all of them is main's loop (no close, no guards)
    const std::uint32_t forms = folded_forms<sync, fp32, out_format>(num_tiles);
    if (forms == 1)
    {
        state.last = 1;
        for (std::uint32_t i = 0; i < num_tiles; i++)
        {
            _llk_pack_<sync, fp32, ckernel::PackMode::Default>(tile_index + i, tile_address(tile_addr, address, page, i));
        }
        return;
    }
    // a block of one folded form reads no state, so it closes no fast untilize either
    const bool one_form = forms != 0 && (forms & (forms - 1)) == 0;
    // per tile after another pack program init, and while the packer accumulates in L1 under half sync
    if (LLK_PACK_OTHER_PROGRAM_ACTIVE() || (sync == DstSync::SyncHalf && LLK_PACK_L1_ACC_ACTIVE()))
    {
        if (!one_form)
        {
            _llk_pack_close_fast_untilize_<sync>();
        }
        state.valid = 0;
        for (std::uint32_t i = 0; i < num_tiles; i++)
        {
            _llk_pack_<sync, fp32, ckernel::PackMode::Default>(tile_index + i, tile_address(tile_addr, address, page, i));
        }
        return;
    }
    const bool first        = MATMUL_PACK_STATE != 0 || state.last == 0;
    // as llk_pack_block_form: one word of the state, the count's two bits
    const std::uint32_t nt   = key_nt != 0 ? key_nt : num_tiles;
    const std::uint32_t word = form_words<fp32, MATMUL_PACK_NOK ? 0 : key_kt, row, tables32, sync == DstSync::SyncFull, src32, int8>.w[first ? (MATMUL_PACK_STATE == 1 ? 1 : 0) : 2];
    const std::uint32_t code = (word >> (2 * ((nt < 16 ? nt : 16) - 1))) & 3;
    const char f             = forms == 2 ? 'B' : (code == 0 ? 'T' : (code == 1 ? 'B' : 'W'));
    // a fast untilize left open marks the last output: only a first block closes it
    if (!one_form && first)
    {
        _llk_pack_close_fast_untilize_<sync>();
    }
    // a W offset block stores the state after it reads it
    const std::uint32_t last = state.last;
    if (f != 'W')
    {
        state.last = 1;
    }
    if (f == 'T')
    {
        for (std::uint32_t i = 0; i < num_tiles; i++)
        {
            _llk_pack_<sync, fp32, ckernel::PackMode::Default>(tile_index + i, tile_address(tile_addr, address, page, i));
        }
        return;
    }
    if (f != 'W')
    {
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
    // continues only right behind a W offset block of the output, with no init or reconfigure in between
    const bool cont = MATMUL_PACK_STATE == 0 && last == 2 && state.valid != 0 && state.page == page && state.next == address && state.tiles + num_tiles <= max_tiles;
    if (!cont)
    {
        // llk_pack_w_take: the output base is programmed again, the stride only when it changed
        program_packer_destination<false>(address);
        if (tile_stride != pack_ch1_w_stride())
        {
            cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Wstride_RMW>(tile_stride);
            pack_ch1_w_stride_set(tile_stride);
        }
        state.valid = 1;
        state.page  = page;
        state.next  = address;
        state.tiles = 0;
    }
    state.last                   = 2;
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

// Every block of the output takes the per-tile form (the first, after a reconfigure, the next ones): the kernel packs per tile, as llk_pack_block<n>
// folds such a block into main's loop. Needs the block keys.
template <DstSync sync, bool fp32, DataFormat out_format>
constexpr bool always_per_tile()
{
    if constexpr (key_nt == 0)
    {
        return false;
    }
    else
    {
        constexpr std::uint32_t face_rows = key_r0 < 16 ? key_r0 : 16;
        constexpr std::uint32_t faces     = (key_r0 > 16 ? 2 : 1) * (key_c1 > 16 ? 2 : 1);
        constexpr std::uint32_t row       = matmul_pack_tables::llk_pack_block_class_k(face_rows, faces, static_cast<std::uint32_t>(out_format));
        constexpr bool int8  = out_format == DataFormat::Int8 || out_format == DataFormat::UInt8;
        constexpr bool src32 = fp32 && key_src32;
        constexpr bool t32   = src32 && unpack_to_dest;
        return forms_k<fp32, key_kt>(row, key_nt, t32, sync == DstSync::SyncFull, src32, int8) == 1;
    }
}

// MATMUL_PACK_PROD: the block pack of llk_pack_block for this output, the plain block where one run writes it back to back (plain format, full tiles), else the
// closed block (3, not the prepared one)
template <DataFormat out_format>
constexpr int prod_form()
{
    return (bfp_format(out_format) || key_r0 < 32 || key_c1 < 32) ? 3 : 1;
}
} // namespace matmul_pack_production
#endif
