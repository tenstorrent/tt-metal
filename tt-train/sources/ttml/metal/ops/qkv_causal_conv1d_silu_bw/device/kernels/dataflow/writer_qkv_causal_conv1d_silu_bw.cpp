// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"
#include "tt-train/sources/ttml/metal/ops/qkv_causal_conv1d_silu_bw/device/hostdevcommon/qkv_causal_conv1d_silu_bw_layout.hpp"

using namespace qkv_conv_bw;

// Builds the 4 shifted grad_preact tiles of a tile row out of the grad_preact tiles compute packs, then writes
// grad_input. Each channel tile is staged as a left and a right strip of face rows (WriterScratch): the tile's 32 rows,
// then the first 3 rows of the next tile row, zero after the last one. grad_input row r reads grad_preact rows
// r .. r + 3, so tap t's shifted tile is strip rows [t, t + 32): four contiguous face copies out of the two strips,
// the reader's scheme with the halo below. Row r's copies need row r + 1, so they trail compute by a step and
// grad_input by two.
template <uint32_t channel_tiles_per_block, uint32_t tile_rows, uint32_t row_splits, uint32_t channel_tiles>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count) {
    const auto grad_input = TensorAccessor(tensor::grad_input);
#if COMPUTE_TAP_GRADS
    const auto grad_tap_partial = TensorAccessor(tensor::grad_tap_partial);
    DataflowBuffer partial(dfb::partial);  // the finished partial, accumulated in place by the packer
    const uint32_t partial_tile_bytes = partial.get_entry_size();  // fp32 tiles
#endif
    DataflowBuffer grad_preact(dfb::grad_preact_out);
    DataflowBuffer grad_preact_shifted(dfb::grad_preact_shifted_tile);
    DataflowBuffer output(dfb::output);
    Scratchpad<uint16_t> scratch(scratch::grad_preact_strips);
    Noc noc;

    constexpr uint32_t shifted_tiles = tap_count * channel_tiles_per_block;
    constexpr uint32_t halo_offset = tile_height * face_row_bytes;  // strip rows [32, 35)
    constexpr uint32_t halo_bytes = halo_rows * face_row_bytes;
    // The strips need a 64-byte aligned base; the scratchpad's is only known at run time.
    const uint32_t scratch_base = local_mem::detail::AddressAccess::address(scratch.local_mem());
    const uint32_t base = align_up(scratch_base, read_alignment) - scratch_base;
    const auto strip = [&](uint32_t parity, uint32_t ct, uint32_t side) {
        return base + ((parity * channel_tiles_per_block + ct) * 2 + side) * strip_stride;
    };

    // The next grad_preact tiles into strip rows [0, 32): face f is side f % 2 from row 16 * (f / 2).
    const auto stage = [&](uint32_t parity) {
        grad_preact.wait_front(channel_tiles_per_block);
        for (uint32_t ct = 0; ct < channel_tiles_per_block; ++ct) {
#pragma GCC unroll 4
            for (uint32_t face = 0; face < 4; ++face) {
                noc.async_write(
                    grad_preact,
                    scratch,
                    face_bytes,
                    {.offset_bytes = ct * tile_bytes + face * face_bytes},
                    {.offset_bytes = strip(parity, ct, face % 2) + (face / 2) * face_rows * face_row_bytes});
            }
        }
        noc.async_write_barrier();
        grad_preact.pop_front(channel_tiles_per_block);
    };
    // Strip rows [32, 35) of parity from rows [0, 3) of the other parity, which holds the next tile row.
    const auto complete = [&](uint32_t parity) {
        for (uint32_t ct = 0; ct < channel_tiles_per_block; ++ct) {
#pragma GCC unroll 2
            for (uint32_t side = 0; side < 2; ++side) {
                noc.async_read(
                    scratch,
                    scratch,
                    halo_bytes,
                    {.offset_bytes = strip(parity ^ 1, ct, side)},
                    {.offset_bytes = strip(parity, ct, side) + halo_offset});
            }
        }
        noc.async_read_barrier();
    };
    // Shifted tiles [tap][ct]: face f of tap t is strip f % 2 from row t + 16 * (f / 2).
    const auto push_shifted = [&](uint32_t parity) {
        grad_preact_shifted.reserve_back(shifted_tiles);
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            for (uint32_t ct = 0; ct < channel_tiles_per_block; ++ct) {
#pragma GCC unroll 4
                for (uint32_t face = 0; face < 4; ++face) {
                    noc.async_read(
                        scratch,
                        grad_preact_shifted,
                        face_bytes,
                        {.offset_bytes = strip(parity, ct, face % 2) + (tap + (face / 2) * face_rows) * face_row_bytes},
                        {.offset_bytes = (tap * channel_tiles_per_block + ct) * tile_bytes + face * face_bytes});
                }
            }
        }
        noc.async_read_barrier();
        grad_preact_shifted.push_back(shifted_tiles);
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

        uint32_t parity = 0;
        for (uint32_t step = 0; step < step_count; ++step) {
            stage(parity);
            if (step > 0) {
                complete(parity ^ 1);
                push_shifted(parity ^ 1);
            }
            // Compute runs grad_input one step behind the shifted copies (see the compute kernel), so row step - 2.
            if (step > 1) {
                write_output(first_tile_row + step - 2, first_tile_column);
            }
            parity ^= 1;
        }
        if (!has_halo) {
            // No tile row follows the last one: its strips end in zeros.
            const uint32_t last = parity ^ 1;
            for (uint32_t ct = 0; ct < channel_tiles_per_block; ++ct) {
                for (uint32_t side = 0; side < 2; ++side) {
                    noc.async_write_zeros(scratch, halo_bytes, {.offset_bytes = strip(last, ct, side) + halo_offset});
                }
            }
            noc.write_zeros_l1_barrier();
            push_shifted(last);  // before draining, so compute never waits for the last row's copies
        }
        if (step_count > 1) {
            write_output(first_tile_row + step_count - 2, first_tile_column);
        }
        if (!has_halo) {
            write_output(end_tile_row - 1, first_tile_column);
        }

#if COMPUTE_TAP_GRADS
        partial.wait_front(shifted_tiles);
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
        partial.pop_front(shifted_tiles);
#endif
    }
}
