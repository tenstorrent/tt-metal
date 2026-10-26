// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Route table, pass 2 (tt-blaze G53F-0195): the regions MoE's routing tables from pass 1's per-block local counts,
// with the values the ttnn composite of tt/moe.py ``_forward_regions`` computes (all integers below 2^24, so its fp32
// arithmetic is exact and equal to this integer arithmetic):
//   counts[e]  = picks of local expert e (0 elsewhere);  aligned = counts rounded up to 32
//   offsets[e] = exclusive cumsum of aligned over the global expert order
//   pos[t, j]  = offsets[e] + (picks of e by tokens before t) for a local pick, the dummy slot (max_buf - 1) otherwise
//   wl[t, j]   = w[t, j] at a local pick, +0 otherwise (= w x loc for the router's positive weights)
// Every core reads all block rows (prefix + totals), then walks its own blocks in token order. pos / wl are written
// as fp32 TILEs (the layout the region scatter and the fused combine read); core 0 also writes counts / offsets
// ([1, E] uint32 rows, the expert kernel's).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t K = get_compile_time_arg_val(0);
constexpr uint32_t E = get_compile_time_arg_val(1);
constexpr uint32_t E_LOC = get_compile_time_arg_val(2);
constexpr uint32_t NBLOCKS = get_compile_time_arg_val(3);
constexpr uint32_t dummy_slot = get_compile_time_arg_val(4);
constexpr uint32_t cb_scr_id = get_compile_time_arg_val(5);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
constexpr uint32_t table_bytes = get_compile_time_arg_val(7);
constexpr uint32_t row_bytes = get_compile_time_arg_val(8);
constexpr auto idx_args = TensorAccessorArgs<9>();
constexpr auto w_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
constexpr auto table_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto bc_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
constexpr auto pos_args = TensorAccessorArgs<bc_args.next_compile_time_args_offset()>();
constexpr auto wl_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
constexpr auto counts_args = TensorAccessorArgs<wl_args.next_compile_time_args_offset()>();
constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t TILE_ELEMS = 1024;
constexpr uint32_t ROW_WORDS = row_bytes / 4;  // a block row's stride in L1 (>= E_LOC)

FORCE_INLINE uint32_t f32_to_u32(uint32_t bits) {
    const uint32_t e = (bits >> 23) & 0xFFu;
    if (e < 127u) {
        return 0u;
    }
    const uint32_t m = (bits & 0x7FFFFFu) | 0x800000u;
    return e >= 150u ? (m << (e - 150u)) : (m >> (150u - e));
}

// an integer below 2^24 -> its fp32 bit pattern (exact)
FORCE_INLINE uint32_t u32_to_f32(uint32_t v) {
    if (v == 0u) {
        return 0u;
    }
    uint32_t n = 23u;
    while (!(v >> n)) {
        --n;
    }
    return ((n + 127u) << 23) | ((v << (23u - n)) & 0x7FFFFFu);
}

FORCE_INLINE uint32_t tile_off(uint32_t r, uint32_t j) { return (r < 16u ? 0u : 512u) + (r & 15u) * 16u + j; }

void kernel_main() {
    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t bc_addr = get_arg_val<uint32_t>(3);
    const uint32_t pos_addr = get_arg_val<uint32_t>(4);
    const uint32_t wl_addr = get_arg_val<uint32_t>(5);
    const uint32_t counts_addr = get_arg_val<uint32_t>(6);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(7);
    const uint32_t block0 = get_arg_val<uint32_t>(8);
    const uint32_t nblocks = get_arg_val<uint32_t>(9);
    const uint32_t write_tables = get_arg_val<uint32_t>(10);

    const auto s_idx = TensorAccessor(idx_args, idx_addr, tile_bytes);
    const auto s_w = TensorAccessor(w_args, w_addr, tile_bytes);
    const auto s_table = TensorAccessor(table_args, table_addr, table_bytes);
    const auto s_bc = TensorAccessor(bc_args, bc_addr, row_bytes);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, tile_bytes);
    const auto s_wl = TensorAccessor(wl_args, wl_addr, tile_bytes);
    const auto s_counts = TensorAccessor(counts_args, counts_addr, table_bytes);
    const auto s_offsets = TensorAccessor(offsets_args, offsets_addr, table_bytes);

    Noc noc;
    CircularBuffer cb_scr(cb_scr_id);
    cb_scr.reserve_back(1);  // private scratch, reserved once and never pushed
    const uint32_t base = get_write_ptr(cb_scr_id);
    // L1: idx | w | pos | wl tiles, the table, counts / offsets rows (E words each), prefix / local offsets, block rows
    const uint32_t idx_l1 = base, w_l1 = idx_l1 + tile_bytes, pos_l1 = w_l1 + tile_bytes, wl_l1 = pos_l1 + tile_bytes;
    const uint32_t table_l1 = wl_l1 + tile_bytes, counts_l1 = table_l1 + table_bytes;
    const uint32_t offsets_l1 = counts_l1 + table_bytes, pre_l1 = offsets_l1 + table_bytes;
    const uint32_t off_l1 = pre_l1 + row_bytes, bc_l1 = off_l1 + row_bytes;
    const CoreLocalMem<volatile uint32_t> idx_t(idx_l1), w_t(w_l1), pos_t(pos_l1), wl_t(wl_l1);
    const CoreLocalMem<volatile int32_t> table(table_l1);
    const CoreLocalMem<volatile uint32_t> counts(counts_l1), offsets(offsets_l1), pre(pre_l1), off(off_l1), bc(bc_l1);

    noc.async_read(s_table, CoreLocalMem<uint32_t>(table_l1), table_bytes, {.page_id = 0}, {});
    for (uint32_t b = 0; b < NBLOCKS; ++b) {
        noc.async_read(s_bc, CoreLocalMem<uint32_t>(bc_l1), row_bytes, {.page_id = b}, {.offset_bytes = b * row_bytes});
    }
    noc.async_read_barrier();
    uint32_t lo = 0;
    while (lo < E && table[lo] < 0) {
        ++lo;
    }
    // prefix over the blocks before this core's first block, totals over all blocks
    for (uint32_t l = 0; l < E_LOC; ++l) {
        uint32_t p = 0, tot = 0;
        for (uint32_t b = 0; b < NBLOCKS; ++b) {
            const uint32_t c = bc[b * ROW_WORDS + l];
            p += b < block0 ? c : 0u;
            tot += c;
        }
        pre[l] = p;
        counts[l] = tot;  // staged: the global row is built below
    }
    // local offsets (exclusive cumsum of the aligned counts) and the global rows
    uint32_t run = 0;
    for (uint32_t l = 0; l < E_LOC; ++l) {
        off[l] = run;
        run += (counts[l] + 31u) & ~31u;
    }
    if (write_tables) {
        // the global [E] rows over the staged local counts, top down: counts[l] is read before it is overwritten
        for (uint32_t e = E; e-- > 0;) {
            const uint32_t l = e - lo;
            const bool local = l < E_LOC;
            offsets[e] = local ? off[l] : (e < lo ? 0u : run);
            counts[e] = local ? counts[l] : 0u;
        }
        noc.async_write(CoreLocalMem<uint32_t>(counts_l1), s_counts, table_bytes, {}, {.page_id = 0});
        noc.async_write(CoreLocalMem<uint32_t>(offsets_l1), s_offsets, table_bytes, {}, {.page_id = 0});
    }

    for (uint32_t b = block0; b < block0 + nblocks; ++b) {
        noc.async_read(s_idx, CoreLocalMem<uint32_t>(idx_l1), tile_bytes, {.page_id = b}, {});
        noc.async_read(s_w, CoreLocalMem<uint32_t>(w_l1), tile_bytes, {.page_id = b}, {});
        noc.async_write_barrier();  // the previous block's tiles have left L1
        for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
            pos_t[i] = 0u;
            wl_t[i] = 0u;
        }
        noc.async_read_barrier();
        for (uint32_t r = 0; r < TILE_ROWS; ++r) {
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t o = tile_off(r, j);
                const uint32_t wb = w_t[o];
                const uint32_t l = f32_to_u32(idx_t[o]) - lo;
                if (((wb >> 23) & 0xFFu) != 0u && l < E_LOC) {
                    pos_t[o] = u32_to_f32(off[l] + pre[l]);
                    pre[l] = pre[l] + 1u;
                    wl_t[o] = wb;
                } else {
                    pos_t[o] = u32_to_f32(dummy_slot);
                }
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(pos_l1), s_pos, tile_bytes, {}, {.page_id = b});
        noc.async_write(CoreLocalMem<uint32_t>(wl_l1), s_wl, tile_bytes, {}, {.page_id = b});
    }
    noc.async_write_barrier();
}
