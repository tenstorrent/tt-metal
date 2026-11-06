// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, route step, writer kernel.
//
// Stores the mask: uint32 row-major [1, NUM_HEADS, SHARD_BLOCKS_PADDED, NUM_SHARDS * SHARD_BLOCKS_PADDED], one row
// (page) per local Q block, one word per global padded K block. Mask row page = head * SHARD_BLOCKS_PADDED +
// 32 * q_block_tile + row; shard s owns the SHARD_BLOCKS_PADDED words at byte offset s * BLOCK_ROW_BYTES, an aligned
// partial-row write.
//   (head, Q-block tile) units: compute delivers SHARD_BLOCK_TILES bf16 0/1 tiles per shard. Each tile row is turned
//     into words in one of RING_PAGES row buffers and written directly.
//   (head, shard) units: compute already delivers the 32 rows as row-major uint32 in one page (phase I').
//
// Compile-time args: 0..10 as sol_route_compute.cpp (names explained there), then TensorAccessorArgs of the mask.
// Common runtime args: 0 mask address.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_mask_out = 16, cb_row_ring = 22;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t RING_PAGES = 8;

// Row `row` of num_tiles bf16 tiles -> 32 * num_tiles uint32 words (1 where nonzero), read two elements per word.
inline void tile_row_to_words(uint32_t tiles_addr, uint32_t row, uint32_t num_tiles, uint32_t out_addr) {
    CoreLocalMem<volatile uint32_t> out(out_addr);
    const uint32_t row_offset_bytes = (((row >> 4) << 1) * 256 + (row & 15) * 16) * 2;  // row start in the left face
    uint32_t word = 0;
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        for (uint32_t face = 0; face < 2; ++face) {  // left face, right face
            CoreLocalMem<volatile uint32_t> elem_pairs(tiles_addr + tile * TILE_BYTES + row_offset_bytes + face * 512);
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t pair = elem_pairs[i];
                out[word++] = (pair & 0xFFFFu) != 0 ? 1u : 0u;
                out[word++] = (pair >> 16) != 0 ? 1u : 0u;
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_SHARDS = get_compile_time_arg_val(1);
    constexpr uint32_t SHARD_BLOCK_TILES = get_compile_time_arg_val(2);
    constexpr uint32_t GRID_X = get_compile_time_arg_val(7);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(8);
    constexpr uint32_t NUM_UNITS = get_compile_time_arg_val(9);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(10);  // 1: (head, shard); 0: (head, Q-block tile)

    constexpr uint32_t SHARD_BLOCKS_PADDED = SHARD_BLOCK_TILES * 32;
    constexpr uint32_t BLOCK_ROW_BYTES = SHARD_BLOCKS_PADDED * 4;  // one shard's words of a mask row
    constexpr auto mask_args = TensorAccessorArgs<11>();

    // This core's units, [core_index * NUM_UNITS / NUM_CORES, (core_index + 1) * NUM_UNITS / NUM_CORES).
    constexpr uint32_t NUM_CORES = GRID_X * GRID_Y;
    const uint32_t core_index = static_cast<uint32_t>(get_absolute_logical_y()) * GRID_X + get_absolute_logical_x();
    const uint32_t unit_start = core_index * NUM_UNITS / NUM_CORES;
    const uint32_t unit_count = (core_index + 1) * NUM_UNITS / NUM_CORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(0));

    Noc noc;
    CircularBuffer mask_cb(cb_mask_out), row_ring_cb(cb_row_ring);
    uint32_t row_ring_base = 0;  // RING_PAGES pages of BLOCK_ROW_BYTES
    if constexpr (!UNITS_PER_SHARD) {
        row_ring_cb.reserve_back(RING_PAGES);
        row_ring_base = row_ring_cb.get_write_ptr();
    }
    uint32_t ring_used = 0;

    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        // One of the two ranges is a single value; the loops are the same either way.
        const uint32_t head = UNITS_PER_SHARD ? unit / NUM_SHARDS : unit / SHARD_BLOCK_TILES;
        const uint32_t q_block_tile_begin = UNITS_PER_SHARD ? 0 : unit % SHARD_BLOCK_TILES;
        const uint32_t q_block_tile_end = UNITS_PER_SHARD ? SHARD_BLOCK_TILES : q_block_tile_begin + 1;
        const uint32_t shard_begin = UNITS_PER_SHARD ? unit % NUM_SHARDS : 0;
        const uint32_t shard_end = UNITS_PER_SHARD ? shard_begin + 1 : NUM_SHARDS;

        for (uint32_t q_block_tile = q_block_tile_begin; q_block_tile < q_block_tile_end; ++q_block_tile) {
            const uint32_t first_mask_page = head * SHARD_BLOCKS_PADDED + q_block_tile * 32;
            for (uint32_t shard = shard_begin; shard < shard_end; ++shard) {
                if constexpr (UNITS_PER_SHARD) {
                    // Rows arrive row-major in one page: one NoC write per row.
                    mask_cb.wait_front(1);
                    const uint32_t rows_addr = mask_cb.get_read_ptr();
                    for (uint32_t row = 0; row < 32; ++row) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(rows_addr + row * BLOCK_ROW_BYTES),
                            mask,
                            BLOCK_ROW_BYTES,
                            {},
                            {.page_id = first_mask_page + row, .offset_bytes = shard * BLOCK_ROW_BYTES});
                    }
                    noc.async_write_barrier();
                    mask_cb.pop_front(1);
                } else {
                    mask_cb.wait_front(SHARD_BLOCK_TILES);
                    const uint32_t mask_tiles_addr = mask_cb.get_read_ptr();
                    for (uint32_t row = 0; row < 32; ++row) {
                        if (ring_used != 0 && (ring_used % RING_PAGES) == 0) {
                            noc.async_write_barrier();  // the ring wraps
                        }
                        const uint32_t ring_slot = row_ring_base + (ring_used % RING_PAGES) * BLOCK_ROW_BYTES;
                        tile_row_to_words(mask_tiles_addr, row, SHARD_BLOCK_TILES, ring_slot);
                        noc.async_write(
                            CoreLocalMem<uint32_t>(ring_slot),
                            mask,
                            BLOCK_ROW_BYTES,
                            {},
                            {.page_id = first_mask_page + row, .offset_bytes = shard * BLOCK_ROW_BYTES});
                        ++ring_used;
                    }
                    mask_cb.pop_front(SHARD_BLOCK_TILES);
                }
            }
        }
    }
    noc.async_write_barrier();
}
