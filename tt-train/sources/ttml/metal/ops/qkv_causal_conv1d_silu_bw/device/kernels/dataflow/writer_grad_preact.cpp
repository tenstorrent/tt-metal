// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/dprint.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

constexpr uint32_t tap_count = 4;

template <uint32_t block_ct, uint32_t Mt, uint32_t Ct>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto grad_preact = TensorAccessor(tensor::grad_preact);
    const auto grad_tap_partial = TensorAccessor(tensor::grad_tap_partial);
    DataflowBuffer dc_rm(dfb::dc_rm);
    DataflowBuffer grad_tap(dfb::grad_tap);
    Noc noc;

    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t block_row_bytes = block_ct * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    const uint32_t tile_bytes = grad_tap.get_entry_size();

    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t block = wi_start + item;
        const uint32_t ct_start = block * block_ct;

        for (uint32_t mt = 0; mt < Mt; ++mt) {
            dc_rm.wait_front(block_ct);
            for (uint32_t row = 0; row < tile_height; ++row) {
                noc.async_write(
                    dc_rm,
                    grad_preact,
                    block_row_bytes,
                    {.offset_bytes = row * block_row_bytes},
                    {.page_id = mt * tile_height + row,
                     .offset_bytes = ct_start * tt::constants::TILE_WIDTH * sizeof(uint16_t)});
            }
            noc.async_write_barrier();
            dc_rm.pop_front(block_ct);
            DPRINT("writer: block {} mt {} grad_preact rows written\n", block, mt);
        }

        grad_tap.wait_front(tap_count * block_ct);
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                noc.async_write(
                    grad_tap,
                    grad_tap_partial,
                    tile_bytes,
                    {.offset_bytes = (tap * block_ct + ct) * tile_bytes},
                    {.page_id = tap * Ct + ct_start + ct});
            }
        }
        noc.async_write_barrier();
        grad_tap.pop_front(tap_count * block_ct);
        DPRINT("writer: block {} grad_tap_partial tiles written\n", block);
    }
}
