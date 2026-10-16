// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Greedy decode head, step 1 (tt/greedy_head.py): per-core argmax of the decode users' logits rows over a range of
// tile columns of this chip's vocab shard.
//
// Input: logits [1, 1, 32, V] bf16 TILE, DRAM interleaved (one page per 32x32 tile column). Users sit in rows
// 0..users-1 (< 16): their values are the first users*32 B of face 0 (columns 0-15) and of face 1 (columns 16-31,
// at byte 512) of every tile.
// Output: row `core` of a [32, 32] uint32 TILE tensor (L1 interleaved, one page): columns 4u, 4u+1 = (key, local
// index) of user u's maximum, where key orders like the bf16 value (sign-adjusted bits). Ties keep the lowest index.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kTileBytesBf16 = 2048;
constexpr uint32_t kFaceBytesBf16 = 512;
constexpr uint32_t kFaceBytesU32 = 1024;
constexpr uint32_t kRowBytesU32 = 64;  // 16 uint32 per face row

FORCE_INLINE uint32_t sort_key(uint32_t bf16) {
    // Larger key <=> larger value; -0 and +0 differ by one key (argmax over them is unaffected in practice).
    return (bf16 & 0x8000u) ? (~bf16 & 0xFFFFu) : (bf16 | 0x8000u);
}

}  // namespace

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t partials_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_begin = get_arg_val<uint32_t>(2);
    const uint32_t tile_end = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);

    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t max_tiles = get_compile_time_arg_val(2);
    static_assert(users >= 1 && users <= 4, "users in rows 0-3 (4 entries of 4 uint32 per partial row)");
    constexpr uint32_t row_bytes = users * 32;  // `users` face rows of 16 bf16

    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);  // per tile: face 0 rows | face 1 rows, then the out row
    const uint32_t out_l1 = scratch + max_tiles * 2 * row_bytes;

    const InterleavedAddrGen<true> logits_gen = {.bank_base_address = logits_addr, .page_size = kTileBytesBf16};
    const uint32_t n = tile_end - tile_begin;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t tile = get_noc_addr(tile_begin + i, logits_gen);
        noc_async_read(tile, scratch + i * 2 * row_bytes, row_bytes);
        noc_async_read(tile + kFaceBytesBf16, scratch + i * 2 * row_bytes + row_bytes, row_bytes);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t u = 0; u < users; ++u) {
        uint32_t best_key = 0;
        uint32_t best_idx = tile_begin * 32;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t base = scratch + i * 2 * row_bytes;
            const volatile tt_l1_ptr uint16_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + u * 32);
            const volatile tt_l1_ptr uint16_t* f1 =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + row_bytes + u * 32);
            const uint32_t col0 = (tile_begin + i) * 32;
            for (uint32_t c = 0; c < 16; ++c) {
                const uint32_t k = sort_key(f0[c]);
                if (k > best_key) {
                    best_key = k;
                    best_idx = col0 + c;
                }
            }
            for (uint32_t c = 0; c < 16; ++c) {
                const uint32_t k = sort_key(f1[c]);
                if (k > best_key) {
                    best_key = k;
                    best_idx = col0 + 16 + c;
                }
            }
        }
        out[u * 4 + 0] = best_key;
        out[u * 4 + 1] = best_idx;
        out[u * 4 + 2] = 0;
        out[u * 4 + 3] = 0;
    }
    for (uint32_t u = users; u < 4; ++u) {
        out[u * 4 + 0] = 0;
        out[u * 4 + 1] = 0;
        out[u * 4 + 2] = 0;
        out[u * 4 + 3] = 0;
    }

    // Row `core` of the uint32 partials tile: face 0 (rows 0-15) or face 2 (rows 16-31), columns 0-15.
    const InterleavedAddrGen<false> part_gen = {.bank_base_address = partials_addr, .page_size = 4096};
    const uint32_t offset = (core < 16 ? 0 : 2 * kFaceBytesU32) + (core % 16) * kRowBytesU32;
    noc_async_write(out_l1, get_noc_addr(0, part_gen, offset), kRowBytesU32);
    noc_async_write_barrier();
}
