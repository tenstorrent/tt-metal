// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/debug/dprint.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

constexpr uint32_t tap_count = 4;
constexpr uint32_t halo_rows = tap_count - 1;
constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

// Keeps the previous dc tile as rows in a 35-row window. When the next dc tile arrives, its first 3 rows complete
// the window, 4 shifted views are pushed for the grad_input conv of the previous tile, then grad_input is written.
template <uint32_t block_ct, uint32_t Mt, uint32_t splits, uint32_t Ct>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto grad_input = TensorAccessor(tensor::grad_input);
    const auto grad_tap_partial = TensorAccessor(tensor::grad_tap_partial);
    DataflowBuffer dc_rm(dfb::dc_rm);
    DataflowBuffer dc_views(dfb::dcv_rm);
    DataflowBuffer output(dfb::output);
    DataflowBuffer grad_tap(dfb::grad_tap);
    Scratchpad<uint16_t> window(scratch::dc_window);
    Noc noc;

    constexpr uint32_t row_bytes = block_ct * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    constexpr uint32_t view_bytes = tile_height * row_bytes;
    const uint32_t tile_bytes = output.get_entry_size();
    const uint32_t window_base = window.get_base_address();
    UnicastEndpoint self;
    const uint32_t self_x = my_x[noc.get_noc_id()];
    const uint32_t self_y = my_y[noc.get_noc_id()];

    const auto push_views = [&]() {
        dc_views.reserve_back(tap_count * block_ct);
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            noc.async_read(
                self,
                dc_views,
                view_bytes,
                {.noc_x = self_x, .noc_y = self_y, .addr = window_base + tap * row_bytes},
                {.offset_bytes = tap * view_bytes});
        }
        noc.async_read_barrier();
        dc_views.push_back(tap_count * block_ct);
    };
    const auto write_output = [&](uint32_t mt, uint32_t ct_start) {
        output.wait_front(block_ct);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            noc.async_write(
                output,
                grad_input,
                tile_bytes,
                {.offset_bytes = ct * tile_bytes},
                {.page_id = mt * Ct + ct_start + ct});
        }
        noc.async_write_barrier();
        output.pop_front(block_ct);
        DPRINT("writer: mt {} grad_input tiles written\n", mt);
    };

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / splits;
        const uint32_t split = work % splits;
        const uint32_t ct_start = block * block_ct;
        const uint32_t m0 = split * Mt / splits;
        const uint32_t m1 = (split + 1) * Mt / splits;
        const bool has_halo = m1 < Mt;
        const uint32_t dc_tiles = m1 - m0 + (has_halo ? 1 : 0);

        for (uint32_t k = 0; k < dc_tiles; ++k) {
            dc_rm.wait_front(block_ct);
            const uint32_t dc_addr = dc_rm.get_read_ptr();
            if (k > 0) {
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
            dc_rm.pop_front(block_ct);
            if (k > 0) {
                write_output(m0 + k - 1, ct_start);
            }
        }
        if (!has_halo) {
            auto* tail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window_base + tile_height * row_bytes);
            for (uint32_t i = 0; i < halo_rows * row_bytes / sizeof(uint32_t); ++i) {
                tail[i] = 0;
            }
            push_views();
            write_output(m1 - 1, ct_start);
        }

        grad_tap.wait_front(tap_count * block_ct);

        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                noc.async_write(
                    grad_tap,
                    grad_tap_partial,
                    tile_bytes,
                    {.offset_bytes = (tap * block_ct + ct) * tile_bytes},
                    {.page_id = (split * tap_count + tap) * Ct + ct_start + ct});
            }
        }
        noc.async_write_barrier();
        grad_tap.pop_front(tap_count * block_ct);
        DPRINT("writer: work {} (block {} split {}) grad_tap_partial written\n", work, block, split);
    }
}
