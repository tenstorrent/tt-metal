// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

constexpr uint32_t tap_count = 4;
constexpr uint32_t halo_rows = tap_count - 1;
constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

// Per work item (channel_block, split): weights, then for every 32-row step a 35-row input window (3 rows of history or
// previous rows + 32 new rows), 4 shifted views of it, and the matching grad_q/k/v tiles.
template <
    uint32_t channel_tiles_per_block,
    uint32_t tile_rows,
    uint32_t row_splits,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t v_tiles>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto grad_q = TensorAccessor(tensor::grad_q);
    const auto grad_k = TensorAccessor(tensor::grad_k);
    const auto grad_v = TensorAccessor(tensor::grad_v);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer activation(dfb::act_rm);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer grad_out(dfb::grad_out);
    Scratchpad<uint16_t> window(scratch::act_window);
    Noc noc;

    constexpr uint32_t row_bytes = channel_tiles_per_block * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    constexpr uint32_t view_bytes = tile_height * row_bytes;
    static_assert(row_bytes % L1_ALIGNMENT == 0 && row_bytes % DRAM_ALIGNMENT == 0);
    const uint32_t tile_bytes = weights.get_entry_size();
    const uint32_t window_base = window.get_base_address();
    UnicastEndpoint self;
    const uint32_t self_x = my_x[noc.get_noc_id()];
    const uint32_t self_y = my_y[noc.get_noc_id()];

    DataflowBuffer ones(dfb::ones);
    ones.reserve_back(1);
    auto* ones_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones.get_write_ptr());
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
        ones_ptr[i] = 0x3F803F80;
    }
    ones.push_back(1);

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t channel_block = work / row_splits;
        const uint32_t split = work % row_splits;
        const uint32_t first_tile_column = channel_block * channel_tiles_per_block;
        const uint32_t col_bytes = first_tile_column * tt::constants::TILE_WIDTH * sizeof(uint16_t);
        const uint32_t first_tile_row = split * tile_rows / row_splits;
        const uint32_t end_tile_row = (split + 1) * tile_rows / row_splits;
        const uint32_t step_count = end_tile_row - first_tile_row + (end_tile_row < tile_rows ? 1 : 0);

        weights.reserve_back(tap_count * channel_tiles_per_block);
        for (uint32_t tile_column = 0; tile_column < channel_tiles_per_block; ++tile_column) {
            const uint32_t page = first_tile_column + tile_column;
            noc.async_read(tap0, weights, tile_bytes, {.page_id = page}, {.offset_bytes = tile_column * tile_bytes});
            noc.async_read(
                tap1,
                weights,
                tile_bytes,
                {.page_id = page},
                {.offset_bytes = (channel_tiles_per_block + tile_column) * tile_bytes});
            noc.async_read(
                tap2,
                weights,
                tile_bytes,
                {.page_id = page},
                {.offset_bytes = (2 * channel_tiles_per_block + tile_column) * tile_bytes});
            noc.async_read(
                tap3,
                weights,
                tile_bytes,
                {.page_id = page},
                {.offset_bytes = (3 * channel_tiles_per_block + tile_column) * tile_bytes});
        }
        noc.async_read_barrier();
        weights.push_back(tap_count * channel_tiles_per_block);

        for (uint32_t step = 0; step < step_count; ++step) {
            const uint32_t tile_row = first_tile_row + step;
            uint32_t first_new_row = 0;
            if (step == 0) {
                for (uint32_t row = 0; row < halo_rows; ++row) {
                    const int32_t source_row = static_cast<int32_t>(tile_row * tile_height + row) - halo_rows;
                    if (source_row < 0) {
                        noc.async_read(
                            history,
                            window,
                            row_bytes,
                            {.page_id = static_cast<uint32_t>(source_row + halo_rows), .offset_bytes = col_bytes},
                            {.offset_bytes = row * row_bytes});
                    } else {
                        noc.async_read(
                            input,
                            window,
                            row_bytes,
                            {.page_id = static_cast<uint32_t>(source_row), .offset_bytes = col_bytes},
                            {.offset_bytes = row * row_bytes});
                    }
                }
            } else {
                noc.async_read(
                    self,
                    window,
                    halo_rows * row_bytes,
                    {.noc_x = self_x, .noc_y = self_y, .addr = window_base + tile_height * row_bytes},
                    {});
                noc.async_read_barrier();
            }
            for (uint32_t row = 0; row < tile_height; ++row) {
                noc.async_read(
                    input,
                    window,
                    row_bytes,
                    {.page_id = tile_row * tile_height + row, .offset_bytes = col_bytes},
                    {.offset_bytes = (halo_rows + row) * row_bytes});
            }
            noc.async_read_barrier();

            activation.reserve_back(tap_count * channel_tiles_per_block);
            for (uint32_t tap = 0; tap < tap_count; ++tap) {
                noc.async_read(
                    self,
                    activation,
                    view_bytes,
                    {.noc_x = self_x, .noc_y = self_y, .addr = window_base + tap * row_bytes},
                    {.offset_bytes = tap * view_bytes});
            }
            grad_out.reserve_back(channel_tiles_per_block);
            for (uint32_t tile_column_in_block = 0; tile_column_in_block < channel_tiles_per_block;
                 ++tile_column_in_block) {
                const uint32_t tile_column = first_tile_column + tile_column_in_block;
                if (tile_column < q_tiles) {
                    noc.async_read(
                        grad_q,
                        grad_out,
                        tile_bytes,
                        {.page_id = tile_row * q_tiles + tile_column},
                        {.offset_bytes = tile_column_in_block * tile_bytes});
                } else if (tile_column < q_tiles + k_tiles) {
                    noc.async_read(
                        grad_k,
                        grad_out,
                        tile_bytes,
                        {.page_id = tile_row * k_tiles + (tile_column - q_tiles)},
                        {.offset_bytes = tile_column_in_block * tile_bytes});
                } else {
                    noc.async_read(
                        grad_v,
                        grad_out,
                        tile_bytes,
                        {.page_id = tile_row * v_tiles + (tile_column - q_tiles - k_tiles)},
                        {.offset_bytes = tile_column_in_block * tile_bytes});
                }
            }
            noc.async_read_barrier();
            activation.push_back(tap_count * channel_tiles_per_block);
            grad_out.push_back(channel_tiles_per_block);
        }
    }
}
