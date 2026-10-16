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

// Keeps the previous dc tile as rows in a 35-row window. When the next dc tile arrives, its first 3 rows complete
// the window, 4 shifted views are pushed for the grad_input conv of the previous tile, then grad_input is written.
template <uint32_t channel_tiles_per_block, uint32_t tile_rows, uint32_t row_splits, uint32_t channel_tiles>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto grad_input = TensorAccessor(tensor::grad_input);
    const auto grad_tap_partial = TensorAccessor(tensor::grad_tap_partial);
    DataflowBuffer dc_rm(dfb::dc_rm);
    DataflowBuffer dc_views(dfb::dcv_rm);
    DataflowBuffer output(dfb::output);
    DataflowBuffer grad_tap(dfb::grad_tap);
    Scratchpad<uint16_t> window(scratch::dc_window);
    Noc noc;

    constexpr uint32_t row_bytes = channel_tiles_per_block * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    constexpr uint32_t view_bytes = tile_height * row_bytes;
    const uint32_t tile_bytes = output.get_entry_size();
    const uint32_t grad_tap_tile_bytes = grad_tap.get_entry_size();  // fp32 tiles
    const uint32_t window_base = window.get_base_address();
    UnicastEndpoint self;
    const uint32_t self_x = my_x[noc.get_noc_id()];
    const uint32_t self_y = my_y[noc.get_noc_id()];

    const auto push_views = [&]() {
        dc_views.reserve_back(tap_count * channel_tiles_per_block);
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            noc.async_read(
                self,
                dc_views,
                view_bytes,
                {.noc_x = self_x, .noc_y = self_y, .addr = window_base + tap * row_bytes},
                {.offset_bytes = tap * view_bytes});
        }
        noc.async_read_barrier();
        dc_views.push_back(tap_count * channel_tiles_per_block);
    };
    const auto write_output = [&](uint32_t tile_row, uint32_t first_tile_column) {
        output.wait_front(channel_tiles_per_block);
        for (uint32_t tile_column = 0; tile_column < channel_tiles_per_block; ++tile_column) {
            noc.async_write(
                output,
                grad_input,
                tile_bytes,
                {.offset_bytes = tile_column * tile_bytes},
                {.page_id = tile_row * channel_tiles + first_tile_column + tile_column});
        }
        noc.async_write_barrier();
        output.pop_front(channel_tiles_per_block);
    };

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t channel_block = work / row_splits;
        const uint32_t split = work % row_splits;
        const uint32_t first_tile_column = channel_block * channel_tiles_per_block;
        const uint32_t first_tile_row = split * tile_rows / row_splits;
        const uint32_t end_tile_row = (split + 1) * tile_rows / row_splits;
        const bool has_halo = end_tile_row < tile_rows;
        const uint32_t step_count = end_tile_row - first_tile_row + (has_halo ? 1 : 0);

        for (uint32_t step = 0; step < step_count; ++step) {
            dc_rm.wait_front(channel_tiles_per_block);
            const uint32_t dc_addr = dc_rm.get_read_ptr();
            if (step > 0) {
                noc.async_read(
                    self,
                    window,
                    halo_rows * row_bytes,
                    {.noc_x = self_x, .noc_y = self_y, .addr = dc_addr},
                    {.offset_bytes = tile_height * row_bytes});
                noc.async_read_barrier();
                push_views();
            }
            noc.async_read(self, window, view_bytes, {.noc_x = self_x, .noc_y = self_y, .addr = dc_addr}, {});
            noc.async_read_barrier();
            dc_rm.pop_front(channel_tiles_per_block);
            if (step > 0) {
                write_output(first_tile_row + step - 1, first_tile_column);
            }
        }
        if (!has_halo) {
            auto* tail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window_base + tile_height * row_bytes);
            for (uint32_t i = 0; i < halo_rows * row_bytes / sizeof(uint32_t); ++i) {
                tail[i] = 0;
            }
            push_views();
            write_output(end_tile_row - 1, first_tile_column);
        }

        grad_tap.wait_front(tap_count * channel_tiles_per_block);

        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            for (uint32_t tile_column = 0; tile_column < channel_tiles_per_block; ++tile_column) {
                noc.async_write(
                    grad_tap,
                    grad_tap_partial,
                    grad_tap_tile_bytes,
                    {.offset_bytes = (tap * channel_tiles_per_block + tile_column) * grad_tap_tile_bytes},
                    {.page_id = (tap * row_splits + split) * channel_tiles + first_tile_column + tile_column});
            }
        }
        noc.async_write_barrier();
        grad_tap.pop_front(tap_count * channel_tiles_per_block);
    }
}
