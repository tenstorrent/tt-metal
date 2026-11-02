// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group A reader. Once per core: the nine 0/1 selection tiles, the reduce scaler, the epsilon tile and the bf16
// scale tile, host-built and never popped. Per (column group, tile row) unit: the group's four taps (cb_tap index
// 4 d + t, re-read only when the column group changes) and one cb_in window: the previous tile row's four column tiles
// (the history tile's at tile row 0) at 0..3, then this tile row's at 4..7. A v group also gets the chain's row-mask
// tile (cb_mask).
// Column tiles of group g: q key head g at 4 g, k key head g - 4 at 16 + 4 (g - 4), v value head g - 8 at
// 32 + 4 (g - 8). Projection page of tile row c = c * 130 + column tile; the history and the taps are one tile row.
//
// Compile-time args: TensorAccessorArgs of projected, history, tap0..tap3, selects, scalars (no chunk count:
// a projection page is c * 130 + column tile, whose stride does not depend on NC).
// Runtime args: 0 projected, 1 history, 2-5 tap, 6 selects, 7 scalars addresses, 8 rows, 9 units, then
// (group, tile row) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_in = 0, cb_sel = 1, cb_tap = 2, cb_scaler = 8, cb_eps = 9, cb_scale = 10, cb_mask = 15;
constexpr uint32_t head_tiles = 4, conv_taps = 4, qk_heads = 4, qk_groups = 8;
constexpr uint32_t window_tiles = 2 * head_tiles;  // the previous tile row's four column tiles, then this one's
constexpr uint32_t k_tile0 = 16, v_tile0 = 32, projection_tiles = 130;
constexpr uint32_t select_tiles = 9, bf16_tile_bytes = 2048;
constexpr uint32_t tile_rows = 32, face_elements = 256, face_columns = 16;
constexpr uint16_t bf16_one_bits = 0x3F80;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * face_elements + (row & 15) * face_columns + (col & 15);
}
constexpr uint32_t reduce_scaler_page = 0, eps_page = 1, scale_page = 2;

constexpr uint32_t group_first_tile(uint32_t group) {
    return group < qk_heads      ? group * head_tiles
           : (group < qk_groups) ? k_tile0 + (group - qk_heads) * head_tiles
                                 : v_tile0 + (group - qk_groups) * head_tiles;
}

// The chain's row_mask_bf16_col tile for one tile row: 1.0 in column 0 of its first `keep` rows (those below `rows`),
// zero elsewhere (the tile's other columns are the tensor's own padding and the column broadcast never reads them).
void build_row_mask(uint32_t cb_id, uint32_t keep) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    const uint32_t mask_l1 = cb.get_write_ptr();
    const CoreLocalMem<volatile uint32_t> words(mask_l1);
    for (uint32_t k = 0; k < bf16_tile_bytes / 4; ++k) {
        words[k] = 0;
    }
    const CoreLocalMem<volatile uint16_t> elements(mask_l1);
    for (uint32_t row = 0; row < keep; ++row) {
        elements[face_element(row, 0)] = bf16_one_bits;
    }
    cb.push_back(1);
}

// One page of `tensor` into one reserved page of `cb`, landed and pushed.
template <typename Accessor>
void read_one(uint32_t cb_id, const Accessor& tensor, uint32_t page) {
    const Noc noc;
    CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    noc.async_read(tensor, cb, tensor.get_aligned_page_size(), {.page_id = page}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb.push_back(1);
}
}  // namespace

void kernel_main() {
    constexpr auto projected_args = TensorAccessorArgs<0>();
    constexpr auto history_args = TensorAccessorArgs<projected_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<history_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto selects_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto scalars_args = TensorAccessorArgs<selects_args.next_compile_time_args_offset()>();

    const uint32_t projected_addr = get_arg_val<uint32_t>(0);
    const uint32_t history_addr = get_arg_val<uint32_t>(1);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(2);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(3);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(4);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(5);
    const uint32_t selects_addr = get_arg_val<uint32_t>(6);
    const uint32_t scalars_addr = get_arg_val<uint32_t>(7);
    const uint32_t rows = get_arg_val<uint32_t>(8);
    const uint32_t units = get_arg_val<uint32_t>(9);
    uint32_t arg = 10;
    const uint32_t last_chunk = (rows - 1) / tile_rows;
    const uint32_t partial = rows % tile_rows;

    const auto projected = TensorAccessor(projected_args, projected_addr);
    const auto history = TensorAccessor(history_args, history_addr);
    const auto tap0 = TensorAccessor(tap0_args, tap0_addr);
    const auto tap1 = TensorAccessor(tap1_args, tap1_addr);
    const auto tap2 = TensorAccessor(tap2_args, tap2_addr);
    const auto tap3 = TensorAccessor(tap3_args, tap3_addr);
    const auto selects = TensorAccessor(selects_args, selects_addr);
    const auto scalars = TensorAccessor(scalars_args, scalars_addr);
    const Noc noc;

    {  // the constants, once: nine selection tiles then the three small tiles, each into its own CB
        CircularBuffer(cb_sel).reserve_back(select_tiles);
        for (uint32_t t = 0; t < select_tiles; ++t) {
            noc.async_read(
                selects,
                CircularBuffer(cb_sel),
                selects.get_aligned_page_size(),
                {.page_id = t},
                {.offset_bytes = t * bf16_tile_bytes});
        }
        noc.async_read_barrier();
        CircularBuffer(cb_sel).push_back(select_tiles);
        read_one(cb_scaler, scalars, reduce_scaler_page);
        read_one(cb_eps, scalars, eps_page);
        read_one(cb_scale, scalars, scale_page);
    }

    uint32_t previous_group = 0xFFFFFFFFu;
    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t group = get_arg_val<uint32_t>(arg++);
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);
        const uint32_t first_tile = group_first_tile(group);
        const bool fresh = group != previous_group;
        previous_group = group;

        if (fresh) {
            // the group's tap weights: tap t's column tile d at cb_tap index 4 d + t
            CircularBuffer(cb_tap).reserve_back(conv_taps * head_tiles);
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc.async_read(
                    tap0,
                    CircularBuffer(cb_tap),
                    tap0.get_aligned_page_size(),
                    {.page_id = first_tile + d},
                    {.offset_bytes = (head_tiles * d + 0) * bf16_tile_bytes});
                noc.async_read(
                    tap1,
                    CircularBuffer(cb_tap),
                    tap1.get_aligned_page_size(),
                    {.page_id = first_tile + d},
                    {.offset_bytes = (head_tiles * d + 1) * bf16_tile_bytes});
                noc.async_read(
                    tap2,
                    CircularBuffer(cb_tap),
                    tap2.get_aligned_page_size(),
                    {.page_id = first_tile + d},
                    {.offset_bytes = (head_tiles * d + 2) * bf16_tile_bytes});
                noc.async_read(
                    tap3,
                    CircularBuffer(cb_tap),
                    tap3.get_aligned_page_size(),
                    {.page_id = first_tile + d},
                    {.offset_bytes = (head_tiles * d + 3) * bf16_tile_bytes});
            }
            noc.async_read_barrier();
            CircularBuffer(cb_tap).push_back(conv_taps * head_tiles);
        }

        // One whole window per unit, pushed and popped as one block of window_tiles tiles on a CB of twice that many
        // pages, so the compute's read never straddles the buffer's end (cb_api.h's contract). The previous tile row is
        // re-read from DRAM rather than carried across the pop, the only way the window stays one contiguous block.
        {
            CircularBuffer(cb_in).reserve_back(window_tiles);
            for (uint32_t d = 0; d < head_tiles; ++d) {
                if (chunk == 0) {
                    noc.async_read(
                        history,
                        CircularBuffer(cb_in),
                        history.get_aligned_page_size(),
                        {.page_id = first_tile + d},
                        {.offset_bytes = d * bf16_tile_bytes});
                } else {
                    noc.async_read(
                        projected,
                        CircularBuffer(cb_in),
                        projected.get_aligned_page_size(),
                        {.page_id = (chunk - 1) * projection_tiles + first_tile + d},
                        {.offset_bytes = d * bf16_tile_bytes});
                }
                noc.async_read(
                    projected,
                    CircularBuffer(cb_in),
                    projected.get_aligned_page_size(),
                    {.page_id = chunk * projection_tiles + first_tile + d},
                    {.offset_bytes = (head_tiles + d) * bf16_tile_bytes});
            }
            noc.async_read_barrier();
            CircularBuffer(cb_in).push_back(window_tiles);
        }

        if (group >= qk_groups) {  // a v group takes the chain's row-mask multiply, which is also its rows mask
            const uint32_t keep = chunk > last_chunk ? 0 : (chunk == last_chunk && partial != 0 ? partial : tile_rows);
            build_row_mask(cb_mask, keep);
        }
    }
}
