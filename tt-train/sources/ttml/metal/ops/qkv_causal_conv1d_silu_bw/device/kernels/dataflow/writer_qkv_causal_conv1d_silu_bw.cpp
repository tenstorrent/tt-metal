// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "tt-train/sources/ttml/metal/ops/qkv_causal_conv1d_silu_bw/device/hostdevcommon/qkv_causal_conv1d_silu_bw_layout.hpp"

using qkv_conv_bw::halo_rows;
using qkv_conv_bw::tap_count;
using qkv_conv_bw::tile_height;

// Keeps the previous grad_preact tile as rows in a 35-row window. When the next grad_preact tile arrives, its first 3
// rows complete the window, 4 shifted copies are pushed for the grad_input conv of the previous tile, then grad_input
// is written.
template <uint32_t channel_tiles_per_block, uint32_t tile_rows, uint32_t row_splits, uint32_t channel_tiles>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto grad_input = TensorAccessor(tensor::grad_input);
#if COMPUTE_TAP_GRADS
    const auto grad_tap_partial = TensorAccessor(tensor::grad_tap_partial);
    DataflowBuffer partial(dfb::partial);  // the finished partial, accumulated in place by the packer
    const uint32_t partial_tile_bytes = partial.get_entry_size();  // fp32 tiles
#endif
    DataflowBuffer grad_preact_rm(dfb::grad_preact_rm);
    DataflowBuffer grad_preact_shifted(dfb::grad_preact_shifted_rm);
    DataflowBuffer output(dfb::output);
    Scratchpad<uint16_t> window(scratch::grad_preact_window);
    Noc noc;

    constexpr uint32_t row_bytes = channel_tiles_per_block * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    constexpr uint32_t view_bytes = tile_height * row_bytes;
    const uint32_t tile_bytes = output.get_entry_size();

    const auto push_grad_preact_shifted = [&]() {
        grad_preact_shifted.reserve_back(tap_count * channel_tiles_per_block);
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            noc.async_read(
                window,
                grad_preact_shifted,
                view_bytes,
                {.offset_bytes = tap * row_bytes},
                {.offset_bytes = tap * view_bytes});
        }
        noc.async_read_barrier();
        grad_preact_shifted.push_back(tap_count * channel_tiles_per_block);
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
            grad_preact_rm.wait_front(channel_tiles_per_block);
            if (step > 0) {
                noc.async_write(
                    grad_preact_rm, window, halo_rows * row_bytes, {}, {.offset_bytes = tile_height * row_bytes});
                noc.async_write_barrier();
                push_grad_preact_shifted();
            }
            noc.async_write(grad_preact_rm, window, view_bytes, {}, {});
            noc.async_write_barrier();
            grad_preact_rm.pop_front(channel_tiles_per_block);
            // Compute runs grad_input one step behind the shifted copies (see the compute kernel), so row step - 2.
            if (step > 1) {
                write_output(first_tile_row + step - 2, first_tile_column);
            }
        }
        if (!has_halo) {
            noc.async_write_zeros(window, halo_rows * row_bytes, {.offset_bytes = tile_height * row_bytes});
            noc.write_zeros_l1_barrier();
            push_grad_preact_shifted();  // before draining, so compute never waits for the last row's copies
        }
        if (step_count > 1) {
            write_output(first_tile_row + step_count - 2, first_tile_column);
        }
        if (!has_halo) {
            write_output(end_tile_row - 1, first_tile_column);
        }

#if COMPUTE_TAP_GRADS
        partial.wait_front(tap_count * channel_tiles_per_block);

        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            for (uint32_t tile_column = 0; tile_column < channel_tiles_per_block; ++tile_column) {
                noc.async_write(
                    partial,
                    grad_tap_partial,
                    partial_tile_bytes,
                    {.offset_bytes = (tap * channel_tiles_per_block + tile_column) * partial_tile_bytes},
                    // tap outermost, then split: the host sums over rows in one fast reduction. With split
                    // outermost that sum runs over an outer axis in fp32 and takes 1.6 ms instead of 0.07 ms.
                    {.page_id = (tap * row_splits + split) * channel_tiles + first_tile_column + tile_column});
            }
        }
        noc.async_write_barrier();
        partial.pop_front(tap_count * channel_tiles_per_block);
#endif
    }
}
