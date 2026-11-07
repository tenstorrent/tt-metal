// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/qkv_causal_conv1d_silu_bw/device/hostdevcommon/qkv_causal_conv1d_silu_bw_layout.hpp"

using namespace qkv_conv_bw;

static_assert(read_alignment % DRAM_ALIGNMENT == 0 && read_alignment % L1_ALIGNMENT == 0);

// Per work item (channel_block, split): weights, then for every tile row the TILE input's faces staged as strips (3
// halo rows + 32 rows), the 4 shifted tiles copied out of the strips, and the matching grad_q/k/v tiles. The strip
// staging is the forward reader's (reader_qkv_causal_conv1d_silu.cpp), face loops unrolled like there.
template <
    uint32_t channel_tiles_per_block,
    uint32_t channel_tiles,
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
    // x_shifted[tap][ct]: the tile row copied 4 times, each copy one row lower. preact = sum_tap w_tap * x_shifted[tap]
    DataflowBuffer x_shifted(dfb::x_shifted_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer grad_out(dfb::grad_out);
    Scratchpad<uint16_t> scratch(scratch::x_window);
    Noc noc;

    constexpr uint32_t block_row_bytes = channel_tiles_per_block * tile_row_bytes;
    constexpr ReaderScratch layout{channel_tiles_per_block};
    // The strips need a 64-byte aligned base; the scratchpad's is only known at run time.
    const uint32_t scratch_base = local_mem::detail::AddressAccess::address(scratch.local_mem());
    const uint32_t base = align_up(scratch_base, read_alignment);
    const auto offset_of = [&](uint32_t address) { return address - scratch_base; };
    const uint32_t staging = offset_of(base + layout.staging());  // previous tile row's bottom faces [ct][side]
    const uint32_t rows = offset_of(base + layout.rows());        // the 3 history rows, row-major
    const auto strip = [&](uint32_t parity, uint32_t ct, uint32_t side) {
        return offset_of(base + strip_phase + ((parity * channel_tiles_per_block + ct) * 2 + side) * strip_stride);
    };
    // The row broadcast reads row 0, so only the two top faces of a tap tile are needed.
    constexpr uint32_t tap_bytes = 2 * face_bytes;

    DataflowBuffer ones(dfb::ones);
    // A tile of 1.0, needed because the FPU has no add-constant: silu' computes 1 + preact * (1 - s) with it. Two
    // bf16 ones per 32-bit store; every number is the same, so the tile layout does not matter.
    constexpr uint32_t two_bf16_ones = (uint32_t{BF16_ONE_BITS} << 16) | BF16_ONE_BITS;
    ones.reserve_back(1);
    {
        const auto lock = ones.scoped_write_lock(1);
        const auto ones_words = lock.get_ptr<uint32_t>();
        for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
            ones_words[i] = two_bf16_ones;
        }
    }
    ones.push_back(1);

    // Items are numbered channel block major, so consecutive items on a core usually share their taps: the taps are
    // pushed once per block, and compute pops them when its block changes.
    constexpr uint32_t no_block = ~0U;
    uint32_t loaded_block = no_block;
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t channel_block = work / row_splits;
        const uint32_t split = work % row_splits;
        const uint32_t first_tile_column = channel_block * channel_tiles_per_block;
        const uint32_t col_bytes = first_tile_column * tile_row_bytes;
        const uint32_t first_tile_row = split * tile_rows / row_splits;
        const uint32_t end_tile_row = (split + 1) * tile_rows / row_splits;
        const uint32_t step_count = end_tile_row - first_tile_row + (end_tile_row < tile_rows ? 1 : 0);

        if (channel_block != loaded_block) {
            weights.reserve_back(tap_count * channel_tiles_per_block);
            for (uint32_t tile_column = 0; tile_column < channel_tiles_per_block; ++tile_column) {
                // weights[tap][tile_column]; the accessors have distinct types, hence no array
                const auto read_tap = [&](const auto& tap, uint32_t index) {
                    noc.async_read(
                        tap,
                        weights,
                        tap_bytes,
                        {.page_id = first_tile_column + tile_column},
                        {.offset_bytes = (index * channel_tiles_per_block + tile_column) * tile_bytes});
                };
                read_tap(tap0, 0);
                read_tap(tap1, 1);
                read_tap(tap2, 2);
                read_tap(tap3, 3);
            }
            noc.async_read_barrier();
            weights.push_back(tap_count * channel_tiles_per_block);
            loaded_block = channel_block;
        }

        uint32_t parity = 0;
        for (uint32_t step = 0; step < step_count; ++step) {
            const uint32_t tile_row = first_tile_row + step;
            // The 3 rows above this tile row: the caller's history at the sequence start, the previous tile row's
            // bottom faces at an item start, else the previous strip's last rows.
            const bool from_history = step == 0 && tile_row == 0;
            const bool from_previous_tile = step == 0 && tile_row > 0;
            if (from_history) {
                for (uint32_t row = 0; row < halo_rows; ++row) {
                    noc.async_read(
                        history,
                        scratch,
                        block_row_bytes,
                        {.page_id = row, .offset_bytes = col_bytes},
                        {.offset_bytes = rows + row * block_row_bytes});
                }
            } else if (from_previous_tile) {
                for (uint32_t ct = 0; ct < channel_tiles_per_block; ++ct) {
                    noc.async_read(
                        input,
                        scratch,
                        2 * face_bytes,
                        {.page_id = (tile_row - 1) * channel_tiles + first_tile_column + ct,
                         .offset_bytes = 2 * face_bytes},
                        {.offset_bytes = staging + ct * 2 * face_bytes});
                }
            }
            // Issued before the face reads so the two DRAM round trips overlap; the barrier below covers both.
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
            // The tile's faces, into strip rows [3, 35).
            for (uint32_t ct = 0; ct < channel_tiles_per_block; ++ct) {
                const uint32_t page = tile_row * channel_tiles + first_tile_column + ct;
#pragma GCC unroll 4
                for (uint32_t face = 0; face < 4; ++face) {
                    noc.async_read(
                        input,
                        scratch,
                        face_bytes,
                        {.page_id = page, .offset_bytes = face * face_bytes},
                        {.offset_bytes =
                             strip(parity, ct, face % 2) + (halo_rows + (face / 2) * face_rows) * face_row_bytes});
                }
            }
            noc.async_read_barrier();

            // Strip rows [0, 3).
            constexpr uint32_t carried_bytes = halo_rows * face_row_bytes;
            for (uint32_t ct = 0; ct < channel_tiles_per_block; ++ct) {
#pragma GCC unroll 2
                for (uint32_t side = 0; side < 2; ++side) {
                    const uint32_t dst = strip(parity, ct, side);
                    if (from_history) {
                        for (uint32_t row = 0; row < halo_rows; ++row) {
                            noc.async_read(
                                scratch,
                                scratch,
                                face_row_bytes,
                                {.offset_bytes = rows + row * block_row_bytes + (ct * 2 + side) * face_row_bytes},
                                {.offset_bytes = dst + row * face_row_bytes});
                        }
                    } else {
                        // The last three rows of the previous tile row on this side.
                        const uint32_t previous_end =
                            from_previous_tile ? staging + ct * 2 * face_bytes + side * face_bytes + face_bytes
                                               : strip(parity ^ 1, ct, side) + strip_rows * face_row_bytes;
                        noc.async_read(
                            scratch,
                            scratch,
                            carried_bytes,
                            {.offset_bytes = previous_end - carried_bytes},
                            {.offset_bytes = dst});
                    }
                }
            }
            noc.async_read_barrier();

            // Shifted tiles [tap][ct]: face f of tap t is strip f % 2 from row t + 16 * (f / 2).
            x_shifted.reserve_back(tap_count * channel_tiles_per_block);
            for (uint32_t tap = 0; tap < tap_count; ++tap) {
                for (uint32_t ct = 0; ct < channel_tiles_per_block; ++ct) {
#pragma GCC unroll 4
                    for (uint32_t face = 0; face < 4; ++face) {
                        noc.async_read(
                            scratch,
                            x_shifted,
                            face_bytes,
                            {.offset_bytes =
                                 strip(parity, ct, face % 2) + (tap + (face / 2) * face_rows) * face_row_bytes},
                            {.offset_bytes = (tap * channel_tiles_per_block + ct) * tile_bytes + face * face_bytes});
                    }
                }
            }
            noc.async_read_barrier();
            x_shifted.push_back(tap_count * channel_tiles_per_block);
            grad_out.push_back(channel_tiles_per_block);
            parity ^= 1;
        }
    }
}
