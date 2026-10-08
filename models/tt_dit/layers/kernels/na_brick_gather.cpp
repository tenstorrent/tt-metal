// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Brick relayouts for the neighborhood-attention K/V halo (one bf16 brick = 32 sites x CHANNELS).
//
// A brick is one tile row: TILES = CHANNELS / 32 tile pages. As a row-major stick it is PARTS
// consecutive pages, the sub-columns the halo exchange cuts a stick into to stay within 4 KB.
//
// MODE 0: tiled band -> brick sticks. Brick b is tile pages TILES*b ..; it leaves as row-major
//         pages PARTS*b .. (site-major, the stick the halo exchange moves).
// MODE 1: brick sticks -> tiled bricks by a site table. Output site i of brick o is source site
//         table[o][i]; brick o leaves as tile pages TILES*o ...
//
// Both modes run one brick at a time through two brick-sized L1 halves of the scratch CB.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t SITES = 32;
constexpr uint32_t CHANNELS = get_compile_time_arg_val(2);
constexpr uint32_t PARTS = get_compile_time_arg_val(3);
constexpr uint32_t SITE_BYTES = CHANNELS * 2;
constexpr uint32_t TILE_BYTES = 2048;  // 32x32 bf16
constexpr uint32_t TILES = CHANNELS / 32;
constexpr uint32_t BRICK_BYTES = TILES * TILE_BYTES;
constexpr uint32_t PAGE_BYTES = BRICK_BYTES / PARTS;
constexpr uint32_t SITES_PER_PAGE = SITES / PARTS;
static_assert(CHANNELS % 32 == 0 && SITES % PARTS == 0, "a brick is whole tiles and whole sites per page");

// Byte offset in a brick's tiles of the 16-channel chunk q (0 .. CHANNELS/16 - 1) of site row r.
FORCE_INLINE uint32_t tile_offset(uint32_t r, uint32_t q) {
    return (q >> 1) * TILE_BYTES + (((r >> 4) << 1) + (q & 1)) * 512 + (r & 15) * 32;
}

// to_tiles: row-major brick at rows -> tiles at tiles; else the reverse.
template <bool to_tiles>
FORCE_INLINE void shuffle(uint32_t rows, uint32_t tiles) {
    for (uint32_t r = 0; r < SITES; ++r) {
        for (uint32_t q = 0; q < CHANNELS / 16; ++q) {
            volatile tt_l1_ptr uint32_t* row =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows + r * SITE_BYTES + q * 32);
            volatile tt_l1_ptr uint32_t* tile =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles + tile_offset(r, q));
#pragma GCC unroll 8
            for (uint32_t k = 0; k < 8; ++k) {
                if constexpr (to_tiles) {
                    tile[k] = row[k];
                } else {
                    row[k] = tile[k];
                }
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);

    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();

    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t staged = get_write_ptr(scratch_cb);
    const uint32_t out = staged + BRICK_BYTES;
    const uint32_t table_l1 = out + BRICK_BYTES;

    if constexpr (mode == 0) {
        for (uint32_t b = start; b < start + count; ++b) {
            for (uint32_t t = 0; t < TILES; ++t) {
                noc_async_read(src.get_noc_addr(TILES * b + t), staged + t * TILE_BYTES, TILE_BYTES);
            }
            noc_async_read_barrier();
            noc_async_write_barrier();
            shuffle<false>(out, staged);
            for (uint32_t p = 0; p < PARTS; ++p) {
                noc_async_write(out + p * PAGE_BYTES, dst.get_noc_addr(PARTS * b + p), PAGE_BYTES);
            }
        }
    } else {
        const auto table = TensorAccessor(table_args, table_addr);
        volatile tt_l1_ptr uint32_t* sites = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
        for (uint32_t o = start; o < start + count; ++o) {
            noc_async_read(table.get_noc_addr(o), table_l1, SITES * 4);
            noc_async_read_barrier();
            // Consecutive sites of one source page are read as one run.
            uint32_t i = 0;
            while (i < SITES) {
                const uint32_t s = sites[i];
                uint32_t n = 1;
                while (i + n < SITES && sites[i + n] == s + n && ((s + n) % SITES_PER_PAGE) != 0) {
                    ++n;
                }
                noc_async_read(
                    src.get_noc_addr(s / SITES_PER_PAGE, (s % SITES_PER_PAGE) * SITE_BYTES),
                    staged + i * SITE_BYTES,
                    n * SITE_BYTES);
                i += n;
            }
            noc_async_read_barrier();
            noc_async_write_barrier();
            shuffle<true>(staged, out);
            for (uint32_t t = 0; t < TILES; ++t) {
                noc_async_write(out + t * TILE_BYTES, dst.get_noc_addr(TILES * o + t), TILE_BYTES);
            }
        }
    }
    noc_async_write_barrier();
}
