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
constexpr uint32_t history_rows = tap_count - 1;

template <uint32_t block_ct, uint32_t Mt, uint32_t Qt, uint32_t Kt, uint32_t Vt>
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

    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t block_row_bytes = block_ct * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    static_assert(block_row_bytes % L1_ALIGNMENT == 0 && block_row_bytes % DRAM_ALIGNMENT == 0);
    const uint32_t tile_bytes = weights.get_entry_size();
    const uint32_t window_base = window.get_base_address();
    UnicastEndpoint self;
    const uint32_t self_x = my_x[noc.get_noc_id()];
    const uint32_t self_y = my_y[noc.get_noc_id()];

    DPRINT("reader: wi_start={} wi_count={} block_ct={} Mt={}\n", wi_start, wi_count, block_ct, Mt);

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t block = wi_start + item;
        const uint32_t ct_start = block * block_ct;

        weights.reserve_back(tap_count * block_ct);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            noc.async_read(tap0, weights, tile_bytes, {.page_id = ct_start + ct}, {.offset_bytes = ct * tile_bytes});
            noc.async_read(
                tap1, weights, tile_bytes, {.page_id = ct_start + ct}, {.offset_bytes = (block_ct + ct) * tile_bytes});
            noc.async_read(
                tap2,
                weights,
                tile_bytes,
                {.page_id = ct_start + ct},
                {.offset_bytes = (2 * block_ct + ct) * tile_bytes});
            noc.async_read(
                tap3,
                weights,
                tile_bytes,
                {.page_id = ct_start + ct},
                {.offset_bytes = (3 * block_ct + ct) * tile_bytes});
        }
        noc.async_read_barrier();
        weights.push_back(tap_count * block_ct);
        DPRINT("reader: block {} weights pushed ({} tiles)\n", block, tap_count * block_ct);

        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t row = 0; row < tile_height + history_rows; ++row) {
                const int32_t source_row =
                    static_cast<int32_t>(mt * tile_height + row) - static_cast<int32_t>(history_rows);
                if (source_row < 0) {
                    noc.async_read(
                        history,
                        window,
                        block_row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row + static_cast<int32_t>(history_rows)),
                         .offset_bytes = ct_start * tt::constants::TILE_WIDTH * sizeof(uint16_t)},
                        {.offset_bytes = row * block_row_bytes});
                } else {
                    noc.async_read(
                        input,
                        window,
                        block_row_bytes,
                        {.page_id = static_cast<uint32_t>(source_row),
                         .offset_bytes = ct_start * tt::constants::TILE_WIDTH * sizeof(uint16_t)},
                        {.offset_bytes = row * block_row_bytes});
                }
            }
            noc.async_read_barrier();
            DPRINT(
                "reader: block {} mt {} window ready (rows {}..{})\n",
                block,
                mt,
                mt * tile_height,
                mt * tile_height + tile_height + history_rows);

            for (uint32_t tap = 0; tap < tap_count; ++tap) {
                activation.reserve_back(block_ct);
                noc.async_read(
                    self,
                    activation,
                    tile_height * block_row_bytes,
                    {.noc_x = self_x, .noc_y = self_y, .addr = window_base + tap * block_row_bytes},
                    {});
                noc.async_read_barrier();
                activation.push_back(block_ct);
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
            grad_out.push_back(block_ct);
            DPRINT("reader: block {} mt {} views + grad_out pushed\n", block, mt);
        }
    }
}
