// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Brick relayouts for the neighborhood-attention K/V halo (one bf16 brick = 32 sites x 64 channels).
//
// MODE 0: tiled band -> brick sticks. Brick b is tile pages 2b and 2b+1; it leaves as row-major
//         page b (4096 B, site-major, the stick the halo exchange moves).
// MODE 1: brick sticks -> tiled bricks by a site table. Output site i of brick o is source site
//         table[o][i] (page s / 32, byte (s % 32) * 128); brick o leaves as tile pages 2o and 2o+1.
//
// Both modes run one brick at a time through two 4 KB L1 halves of the scratch CB.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t SITES = 32;
constexpr uint32_t SITE_BYTES = 128;   // 64 bf16 channels
constexpr uint32_t TILE_BYTES = 2048;  // 32x32 bf16
constexpr uint32_t BRICK_BYTES = 2 * TILE_BYTES;

// Byte offset in a tile pair of the 16-channel chunk q (0..3) of site row r.
FORCE_INLINE uint32_t tile_offset(uint32_t r, uint32_t q) {
    return (q >> 1) * TILE_BYTES + (((r >> 4) << 1) + (q & 1)) * 512 + (r & 15) * 32;
}

// to_tiles: row-major brick at rows -> tile pair at tiles; else the reverse.
template <bool to_tiles>
FORCE_INLINE void shuffle(uint32_t rows, uint32_t tiles) {
    for (uint32_t r = 0; r < SITES; ++r) {
        for (uint32_t q = 0; q < 4; ++q) {
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
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();

    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t staged = get_write_ptr(scratch_cb);
    const uint32_t out = staged + BRICK_BYTES;
    const uint32_t table_l1 = out + BRICK_BYTES;

    if constexpr (mode == 0) {
        for (uint32_t b = start; b < start + count; ++b) {
            noc_async_read(src.get_noc_addr(2 * b), staged, TILE_BYTES);
            noc_async_read(src.get_noc_addr(2 * b + 1), staged + TILE_BYTES, TILE_BYTES);
            noc_async_read_barrier();
            noc_async_write_barrier();
            shuffle<false>(out, staged);
            noc_async_write(out, dst.get_noc_addr(b), BRICK_BYTES);
        }
    } else {
        const auto table = TensorAccessor(table_args, table_addr);
        volatile tt_l1_ptr uint32_t* sites = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
        for (uint32_t o = start; o < start + count; ++o) {
            noc_async_read(table.get_noc_addr(o), table_l1, SITES * 4);
            noc_async_read_barrier();
            // Consecutive sites of one source stick are read as one run.
            uint32_t i = 0;
            while (i < SITES) {
                const uint32_t s = sites[i];
                uint32_t n = 1;
                while (i + n < SITES && sites[i + n] == s + n && ((s + n) % SITES) != 0) {
                    ++n;
                }
                noc_async_read(
                    src.get_noc_addr(s / SITES, (s % SITES) * SITE_BYTES), staged + i * SITE_BYTES, n * SITE_BYTES);
                i += n;
            }
            noc_async_read_barrier();
            noc_async_write_barrier();
            shuffle<true>(staged, out);
            noc_async_write(out, dst.get_noc_addr(2 * o), TILE_BYTES);
            noc_async_write(out + TILE_BYTES, dst.get_noc_addr(2 * o + 1), TILE_BYTES);
        }
    }
    noc_async_write_barrier();
}
