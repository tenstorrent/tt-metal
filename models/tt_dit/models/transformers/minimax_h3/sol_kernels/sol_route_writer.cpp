// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, route step, writer kernel.
//
// Stores the mask rows. Two row layouts:
//   chunk-form (MODE 0 mask, MODE 2 union row): every shard's 0/1 chunk tiles are collected in an L1 row buffer
//     (32 rows x ALL_K_CHUNKS_PADDED words) and written as whole rows, pages
//     head * SHARD_Q_CHUNKS_PADDED + 32 * q_chunk_tile + row.
//   gather-form (MODE 1/2 mask): each shard's SHARD_BLOCKS_PADDED words of a row are written directly (aligned
//     partial-row writes).
// (head, shard) units (MODE 1 only): compute already delivers row-major uint32 rows, one NoC write per row.
//
// Compile-time args: 0..18 as sol_route_compute.cpp (names explained there), then TensorAccessorArgs of mask and
// union. Common runtime args: 0 mask address, 1 union address.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_mask_out = 16, cb_chunk_rows = 19, cb_union_out = 21, cb_row_ring = 22;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t RING_PAGES = 8;

// Is element (row, col) of a bf16 tile nonzero?
inline uint32_t tile_elem_nz(uint32_t tile_addr, uint32_t row, uint32_t col) {
    CoreLocalMem<volatile uint16_t> elems(tile_addr);
    const uint32_t face = ((row >> 4) << 1) | (col >> 4);
    return elems[face * 256 + (row & 15) * 16 + (col & 15)] != 0 ? 1u : 0u;
}

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
    constexpr uint32_t SHARD_Q_CHUNK_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t SHARD_K_CHUNK_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t SHARD_K_CHUNKS = get_compile_time_arg_val(7);
    constexpr uint32_t ALL_K_CHUNKS_PADDED = get_compile_time_arg_val(8);
    constexpr uint32_t MODE = get_compile_time_arg_val(9);
    constexpr uint32_t GRID_X = get_compile_time_arg_val(15);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(16);
    constexpr uint32_t NUM_UNITS = get_compile_time_arg_val(17);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(18);  // 1: (head, shard); 0: (head, Q-chunk tile)

    constexpr uint32_t SHARD_BLOCKS_PADDED = SHARD_BLOCK_TILES * 32;
    constexpr uint32_t SHARD_Q_CHUNKS_PADDED = SHARD_Q_CHUNK_TILES * 32;
    constexpr uint32_t MASK_TILES_PER_SHARD = MODE == 0 ? SHARD_K_CHUNK_TILES : SHARD_BLOCK_TILES;
    constexpr uint32_t CHUNK_ROW_BYTES = ALL_K_CHUNKS_PADDED * 4;
    constexpr uint32_t BLOCK_ROW_BYTES = SHARD_BLOCKS_PADDED * 4;  // one shard's words of a gather-form row
    constexpr auto mask_args = TensorAccessorArgs<19>();
    constexpr auto union_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();

    // Chunk-form rows need every shard's words, so they need (head, Q-chunk tile) units.
    static_assert(UNITS_PER_SHARD == 0 || MODE == 1, "(head, shard) units emit only gather-form rows (MODE 1)");
    // This core's units, [core_index * NUM_UNITS / NUM_CORES, (core_index + 1) * NUM_UNITS / NUM_CORES).
    constexpr uint32_t NUM_CORES = GRID_X * GRID_Y;
    const uint32_t core_index = static_cast<uint32_t>(get_absolute_logical_y()) * GRID_X + get_absolute_logical_x();
    const uint32_t unit_start = core_index * NUM_UNITS / NUM_CORES;
    const uint32_t unit_count = (core_index + 1) * NUM_UNITS / NUM_CORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(0));
    [[maybe_unused]] const auto union_rows = TensorAccessor(union_args, get_common_arg_val<uint32_t>(1));

    Noc noc;
    CircularBuffer mask_cb(cb_mask_out), chunk_rows_cb(cb_chunk_rows), union_cb(cb_union_out), row_ring_cb(cb_row_ring);
    chunk_rows_cb.reserve_back(32);
    const uint32_t chunk_rows_base = chunk_rows_cb.get_write_ptr();  // 32 pages of CHUNK_ROW_BYTES
    row_ring_cb.reserve_back(RING_PAGES);
    const uint32_t row_ring_base = row_ring_cb.get_write_ptr();  // RING_PAGES pages of BLOCK_ROW_BYTES
    uint32_t ring_used = 0;

    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        // One of the two ranges is a single value; the loops are the same either way.
        const uint32_t head = UNITS_PER_SHARD ? unit / NUM_SHARDS : unit / SHARD_Q_CHUNK_TILES;
        const uint32_t q_chunk_tile_begin = UNITS_PER_SHARD ? 0 : unit % SHARD_Q_CHUNK_TILES;
        const uint32_t q_chunk_tile_end = UNITS_PER_SHARD ? SHARD_Q_CHUNK_TILES : q_chunk_tile_begin + 1;
        const uint32_t shard_begin = UNITS_PER_SHARD ? unit % NUM_SHARDS : 0;
        const uint32_t shard_end = UNITS_PER_SHARD ? shard_begin + 1 : NUM_SHARDS;

        for (uint32_t q_chunk_tile = q_chunk_tile_begin; q_chunk_tile < q_chunk_tile_end; ++q_chunk_tile) {
            const uint32_t first_mask_page = head * SHARD_Q_CHUNKS_PADDED + q_chunk_tile * 32;
            // Clear the row buffer.
            if constexpr (MODE != 1) {
                CoreLocalMem<volatile uint32_t> words(chunk_rows_base);
                for (uint32_t i = 0; i < 32 * ALL_K_CHUNKS_PADDED; ++i) {
                    words[i] = 0;
                }
            }

            for (uint32_t shard = shard_begin; shard < shard_end; ++shard) {
                // ---- this shard's mask
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
                    mask_cb.wait_front(MASK_TILES_PER_SHARD);
                    const uint32_t mask_tiles_addr = mask_cb.get_read_ptr();
                    if constexpr (MODE == 0) {
                        for (uint32_t row = 0; row < 32; ++row) {
                            CoreLocalMem<volatile uint32_t> words(chunk_rows_base + row * CHUNK_ROW_BYTES);
                            for (uint32_t k_chunk = 0; k_chunk < SHARD_K_CHUNKS; ++k_chunk) {
                                words[shard * SHARD_K_CHUNKS + k_chunk] =
                                    tile_elem_nz(mask_tiles_addr + (k_chunk >> 5) * TILE_BYTES, row, k_chunk & 31);
                            }
                        }
                    } else {
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
                    }
                    mask_cb.pop_front(MASK_TILES_PER_SHARD);
                }

                // ---- this shard's union-row tiles (MODE 2)
                if constexpr (MODE == 2) {
                    union_cb.wait_front(SHARD_K_CHUNK_TILES);
                    const uint32_t union_tiles_addr = union_cb.get_read_ptr();
                    for (uint32_t row = 0; row < 32; ++row) {
                        CoreLocalMem<volatile uint32_t> words(chunk_rows_base + row * CHUNK_ROW_BYTES);
                        for (uint32_t k_chunk = 0; k_chunk < SHARD_K_CHUNKS; ++k_chunk) {
                            words[shard * SHARD_K_CHUNKS + k_chunk] =
                                tile_elem_nz(union_tiles_addr + (k_chunk >> 5) * TILE_BYTES, row, k_chunk & 31);
                        }
                    }
                    union_cb.pop_front(SHARD_K_CHUNK_TILES);
                }
            }

            // ---- write the complete chunk-form rows
            if constexpr (MODE != 1) {
                for (uint32_t row = 0; row < 32; ++row) {
                    if constexpr (MODE == 0) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(chunk_rows_base + row * CHUNK_ROW_BYTES),
                            mask,
                            CHUNK_ROW_BYTES,
                            {},
                            {.page_id = first_mask_page + row});
                    } else {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(chunk_rows_base + row * CHUNK_ROW_BYTES),
                            union_rows,
                            CHUNK_ROW_BYTES,
                            {},
                            {.page_id = first_mask_page + row});
                    }
                }
            }
            noc.async_write_barrier();
        }
    }
}
