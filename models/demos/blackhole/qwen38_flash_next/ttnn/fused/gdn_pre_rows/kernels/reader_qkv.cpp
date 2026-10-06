// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group A reader.  Once per core: the nine 0/1 selection tiles (CB_SEL), the reduce scaler, the epsilon tile and the
// bf16 scale tile (CB_SCALER / CB_EPS / CB_SCALE), all host-built and never popped.  Per (column group, tile row)
// unit: the four conv taps' four column tiles of the group (CB_TAP, index 4 d + t, re-read only when the column
// group changes) and one whole CB_IN window -- the PREVIOUS tile row's four column tiles (the history tile's when
// the tile row is 0) at 0..3, then this tile row's four at 4..7.
//
// Column tiles of group g: q key head g at 4 g, k key head g - 4 at 16 + 4 (g - 4), v value head g - 8 at
// 32 + 4 (g - 8).  Projection page of tile row c = c * 130 + column tile; the history and the taps are one tile row
// (page = column tile).
//
// A v group also gets the chain's row-mask tile for its tile row (CB_MASK): 1.0 in column 0 of the rows below
// `rows`, which the compute broadcasts across the columns and multiplies in, exactly as
// ttnn.multiply(v_slice, row_mask_bf16_col) does.
//
// Compile-time args: TensorAccessorArgs of projected, history, tap0..tap3, selects, scalars (no chunk count:
// a projection page is c * 130 + column tile, whose stride does not depend on NC).
// Runtime args: 0 projected, 1 history, 2-5 tap, 6 selects, 7 scalars addresses, 8 rows, 9 units, then
// (group, tile row) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_IN = 0, CB_SEL = 1, CB_TAP = 2, CB_SCALER = 8, CB_EPS = 9, CB_SCALE = 10, CB_MASK = 15;
constexpr uint32_t HEAD_TILES = 4, CONV_KERNEL = 4, QK_HEADS = 4, QK_GROUPS = 8;
constexpr uint32_t WINDOW = 2 * HEAD_TILES;  // the previous tile row's four column tiles, then this one's
constexpr uint32_t K_TILE0 = 16, V_TILE0 = 32, PROJECTION_TILES = 130;
constexpr uint32_t SELECT_TILES = 9, BF16_TILE = 2048;
constexpr uint32_t TILE_ROWS = 32, FACE_ELEMS = 256, FACE_COLS = 16;
constexpr uint16_t BF16_ONE = 0x3F80;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * FACE_ELEMS + (row & 15) * FACE_COLS + (col & 15);
}
constexpr uint32_t SCALAR_REDUCE = 0, SCALAR_EPS = 1, SCALAR_SCALE = 2;

constexpr uint32_t group_first_tile(uint32_t group) {
    return group < QK_HEADS      ? group * HEAD_TILES
           : (group < QK_GROUPS) ? K_TILE0 + (group - QK_HEADS) * HEAD_TILES
                                 : V_TILE0 + (group - QK_GROUPS) * HEAD_TILES;
}

// The chain's row_mask_bf16_col tile for one tile row: 1.0 in column 0 of the rows below `rows`, zero elsewhere
// (the tile's other columns are the tensor's own padding and the column broadcast never reads them).
void build_row_mask(uint32_t cb, uint32_t keep) {
    cb_reserve_back(cb, 1);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0; k < BF16_TILE / 4; ++k) {
        words[k] = 0;
    }
    volatile tt_l1_ptr uint16_t* elements = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1);
    for (uint32_t row = 0; row < keep; ++row) {
        elements[face_element(row, 0)] = BF16_ONE;
    }
    cb_push_back(cb, 1);
}

template <typename Acc>
void read_one(uint32_t cb, const Acc& acc, uint32_t page) {
    cb_reserve_back(cb, 1);
    noc_async_read_page(page, acc, get_write_ptr(cb));
    noc_async_read_barrier();
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto h_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto t0_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
    constexpr auto t2_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
    constexpr auto t3_args = TensorAccessorArgs<t2_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<t3_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0_addr = get_arg_val<uint32_t>(2);
    const uint32_t t1_addr = get_arg_val<uint32_t>(3);
    const uint32_t t2_addr = get_arg_val<uint32_t>(4);
    const uint32_t t3_addr = get_arg_val<uint32_t>(5);
    const uint32_t s_addr = get_arg_val<uint32_t>(6);
    const uint32_t c_addr = get_arg_val<uint32_t>(7);
    const uint32_t rows = get_arg_val<uint32_t>(8);
    const uint32_t units = get_arg_val<uint32_t>(9);
    uint32_t arg = 10;
    const uint32_t last_chunk = (rows - 1) / TILE_ROWS;
    const uint32_t partial = rows % TILE_ROWS;

    const auto p = TensorAccessor(p_args, p_addr);
    const auto h = TensorAccessor(h_args, h_addr);
    const auto t0 = TensorAccessor(t0_args, t0_addr);
    const auto t1 = TensorAccessor(t1_args, t1_addr);
    const auto t2 = TensorAccessor(t2_args, t2_addr);
    const auto t3 = TensorAccessor(t3_args, t3_addr);
    const auto s = TensorAccessor(s_args, s_addr);
    const auto k = TensorAccessor(c_args, c_addr);

    {  // the constants, once: nine selection tiles then the three small tiles, each into its own CB
        FUSED_ZONE("fz_gpr_rq_setup");
        cb_reserve_back(CB_SEL, SELECT_TILES);
        const uint32_t l1 = get_write_ptr(CB_SEL);
        for (uint32_t t = 0; t < SELECT_TILES; ++t) {
            noc_async_read_page(t, s, l1 + t * BF16_TILE);
        }
        noc_async_read_barrier();
        cb_push_back(CB_SEL, SELECT_TILES);
        read_one(CB_SCALER, k, SCALAR_REDUCE);
        read_one(CB_EPS, k, SCALAR_EPS);
        read_one(CB_SCALE, k, SCALAR_SCALE);
    }

    uint32_t previous_group = 0xFFFFFFFFu;
    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t group = get_arg_val<uint32_t>(arg++);
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);
        const uint32_t first = group_first_tile(group);
        const bool fresh = group != previous_group;
        previous_group = group;

        if (fresh) {
            FUSED_ZONE("fz_gpr_rq_taps");
            // the group's tap weights: tap t's column tile d at CB_TAP index 4 d + t
            cb_reserve_back(CB_TAP, CONV_KERNEL * HEAD_TILES);
            const uint32_t tl1 = get_write_ptr(CB_TAP);
            for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                noc_async_read_page(first + d, t0, tl1 + (HEAD_TILES * d + 0) * BF16_TILE);
                noc_async_read_page(first + d, t1, tl1 + (HEAD_TILES * d + 1) * BF16_TILE);
                noc_async_read_page(first + d, t2, tl1 + (HEAD_TILES * d + 2) * BF16_TILE);
                noc_async_read_page(first + d, t3, tl1 + (HEAD_TILES * d + 3) * BF16_TILE);
            }
            noc_async_read_barrier();
            cb_push_back(CB_TAP, CONV_KERNEL * HEAD_TILES);
        }

        // One whole window per unit: the previous tile row (the history tile at tile row 0) then this one.  The
        // window is pushed and popped as one block of WINDOW tiles on a CB of 2 * WINDOW pages, so the compute's
        // WINDOW-tile read never straddles the buffer's end and every cycle of pushes and pops sums to the CB size
        // (cb_api.h's contract).  The previous tile row is re-read from DRAM rather than carried across the pop:
        // one extra read per unit, and the only form in which the window is a single contiguous block.
        {
            FUSED_ZONE("fz_gpr_rq_window");
            cb_reserve_back(CB_IN, WINDOW);
            const uint32_t l1 = get_write_ptr(CB_IN);
            for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                if (chunk == 0) {
                    noc_async_read_page(first + d, h, l1 + d * BF16_TILE);
                } else {
                    noc_async_read_page((chunk - 1) * PROJECTION_TILES + first + d, p, l1 + d * BF16_TILE);
                }
                noc_async_read_page(chunk * PROJECTION_TILES + first + d, p, l1 + (HEAD_TILES + d) * BF16_TILE);
            }
            noc_async_read_barrier();
            cb_push_back(CB_IN, WINDOW);
        }

        if (group >= QK_GROUPS) {  // a v group takes the chain's row-mask multiply, which is also its rows mask
            FUSED_ZONE("fz_gpr_rq_mask");
            const uint32_t keep = chunk > last_chunk ? 0 : (chunk == last_chunk && partial != 0 ? partial : TILE_ROWS);
            build_row_mask(CB_MASK, keep);
        }
    }
}
