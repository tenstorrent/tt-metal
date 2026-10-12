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
constexpr uint32_t future_rows = tap_count - 1;

// Weight slot s holds tap[3 - s]: view_s is dc shifted by +s rows, and dx[t] = sum_s tap[3 - s] * dc[t + s].
template <uint32_t block_ct, uint32_t Mt>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto grad_preact = TensorAccessor(tensor::grad_preact);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer activation(dfb::act_rm);
    DataflowBuffer weights(dfb::weights);
    Scratchpad<uint16_t> window(scratch::act_window);
    Noc noc;

    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t rows = Mt * tile_height;
    constexpr uint32_t block_row_bytes = block_ct * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    static_assert(block_row_bytes % L1_ALIGNMENT == 0 && block_row_bytes % DRAM_ALIGNMENT == 0);
    const uint32_t tile_bytes = weights.get_entry_size();
    const uint32_t window_base = window.get_base_address();
    UnicastEndpoint self;
    const uint32_t self_x = my_x[noc.get_noc_id()];
    const uint32_t self_y = my_y[noc.get_noc_id()];

    DPRINT("reader: wi_start={} wi_count={} block_ct={} Mt={}\n", wi_start, wi_count, block_ct, Mt);

    constexpr uint32_t no_block = ~0U;
    uint32_t loaded_block = no_block;
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t block = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = block * block_ct;

        if (block != loaded_block) {
            weights.reserve_back(tap_count * block_ct);
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                noc.async_read(
                    tap3, weights, tile_bytes, {.page_id = ct_start + ct}, {.offset_bytes = ct * tile_bytes});
                noc.async_read(
                    tap2,
                    weights,
                    tile_bytes,
                    {.page_id = ct_start + ct},
                    {.offset_bytes = (block_ct + ct) * tile_bytes});
                noc.async_read(
                    tap1,
                    weights,
                    tile_bytes,
                    {.page_id = ct_start + ct},
                    {.offset_bytes = (2 * block_ct + ct) * tile_bytes});
                noc.async_read(
                    tap0,
                    weights,
                    tile_bytes,
                    {.page_id = ct_start + ct},
                    {.offset_bytes = (3 * block_ct + ct) * tile_bytes});
            }
            noc.async_read_barrier();
            weights.push_back(tap_count * block_ct);
            loaded_block = block;
            DPRINT("reader: block {} reversed weights pushed\n", block);
        }

        for (uint32_t row = 0; row < tile_height + future_rows; ++row) {
            const uint32_t source_row = mt * tile_height + row;
            if (source_row < rows) {
                noc.async_read(
                    grad_preact,
                    window,
                    block_row_bytes,
                    {.page_id = source_row, .offset_bytes = ct_start * tt::constants::TILE_WIDTH * sizeof(uint16_t)},
                    {.offset_bytes = row * block_row_bytes});
            } else {
                auto* zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(window_base + row * block_row_bytes);
                for (uint32_t i = 0; i < block_row_bytes / sizeof(uint32_t); ++i) {
                    zeros[i] = 0;
                }
            }
        }
        noc.async_read_barrier();
        DPRINT("reader: work {} (block {} mt {}) window ready\n", work, block, mt);

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
    }
}
