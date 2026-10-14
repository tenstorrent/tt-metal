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

// Per work item (block, split): weights, then for every dc tile a 35-row input window (3 rows of history or
// previous rows + 32 new rows), 4 shifted views of it, and the matching grad_q/k/v tiles.
template <uint32_t block_ct, uint32_t Mt, uint32_t splits, uint32_t Qt, uint32_t Kt, uint32_t Vt>
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

    constexpr uint32_t row_bytes = block_ct * tt::constants::TILE_WIDTH * sizeof(uint16_t);
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
        const uint32_t block = work / splits;
        const uint32_t split = work % splits;
        const uint32_t ct_start = block * block_ct;
        const uint32_t col_bytes = ct_start * tt::constants::TILE_WIDTH * sizeof(uint16_t);
        const uint32_t m0 = split * Mt / splits;
        const uint32_t m1 = (split + 1) * Mt / splits;
        const uint32_t dc_tiles = m1 - m0 + (m1 < Mt ? 1 : 0);

        weights.reserve_back(tap_count * block_ct);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            const uint32_t page = ct_start + ct;
            noc.async_read(tap0, weights, tile_bytes, {.page_id = page}, {.offset_bytes = ct * tile_bytes});
            noc.async_read(
                tap1, weights, tile_bytes, {.page_id = page}, {.offset_bytes = (block_ct + ct) * tile_bytes});
            noc.async_read(
                tap2, weights, tile_bytes, {.page_id = page}, {.offset_bytes = (2 * block_ct + ct) * tile_bytes});
            noc.async_read(
                tap3, weights, tile_bytes, {.page_id = page}, {.offset_bytes = (3 * block_ct + ct) * tile_bytes});
        }
        noc.async_read_barrier();
        weights.push_back(tap_count * block_ct);

        for (uint32_t k = 0; k < dc_tiles; ++k) {
            const uint32_t mt = m0 + k;
            uint32_t first_new_row = 0;
            if (k == 0) {
                for (uint32_t row = 0; row < halo_rows; ++row) {
                    const int32_t source_row = static_cast<int32_t>(mt * tile_height + row) - halo_rows;
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
                    {.page_id = mt * tile_height + row, .offset_bytes = col_bytes},
                    {.offset_bytes = (halo_rows + row) * row_bytes});
            }
            noc.async_read_barrier();

            activation.reserve_back(tap_count * block_ct);
            for (uint32_t tap = 0; tap < tap_count; ++tap) {
                noc.async_read(
                    self,
                    activation,
                    view_bytes,
                    {.noc_x = self_x, .noc_y = self_y, .addr = window_base + tap * row_bytes},
                    {.offset_bytes = tap * view_bytes});
            }
            grad_out.reserve_back(block_ct);
            for (uint32_t local_ct = 0; local_ct < block_ct; ++local_ct) {
                const uint32_t ct = ct_start + local_ct;
                if (ct < Qt) {
                    noc.async_read(
                        grad_q,
                        grad_out,
                        tile_bytes,
                        {.page_id = mt * Qt + ct},
                        {.offset_bytes = local_ct * tile_bytes});
                } else if (ct < Qt + Kt) {
                    noc.async_read(
                        grad_k,
                        grad_out,
                        tile_bytes,
                        {.page_id = mt * Kt + (ct - Qt)},
                        {.offset_bytes = local_ct * tile_bytes});
                } else {
                    noc.async_read(
                        grad_v,
                        grad_out,
                        tile_bytes,
                        {.page_id = mt * Vt + (ct - Qt - Kt)},
                        {.offset_bytes = local_ct * tile_bytes});
                }
            }
            noc.async_read_barrier();
            activation.push_back(tap_count * block_ct);
            grad_out.push_back(block_ct);
        }
    }
}
