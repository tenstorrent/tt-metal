// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Greedy decode head, step 1 (tt/greedy_head.py): per-core argmax of the decode users' logits rows over a range of
// tile columns of this chip's vocab shard.
//
// Input: logits [1, 1, 32, V] bf16 or fp32 TILE, DRAM interleaved (one page per 32x32 tile column). Users sit in
// rows 0..users-1 (< 16): their values are the first users*16 elements of face 0 (columns 0-15) and of face 1
// (columns 16-31) of every tile.
// Output: row `core` of a [32, 32] uint32 TILE tensor (L1 interleaved, one page): columns 2u, 2u+1 = (key, local
// index) of user u's maximum (users 0-7 in the row's left face, 8-15 in its right face), where key orders like the
// value (sign-adjusted bits). Ties keep the lowest index.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kFaceBytesU32 = 1024;
constexpr uint32_t kRowBytesU32 = 64;  // 16 uint32 per face row

// Larger key <=> larger value; -0 and +0 differ by one key (argmax over them is unaffected in practice).
FORCE_INLINE uint32_t sort_key(uint16_t bf16) {
    const uint32_t b = bf16;
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}
FORCE_INLINE uint32_t sort_key(uint32_t fp32) {
    return fp32 ^ (static_cast<uint32_t>(static_cast<int32_t>(fp32) >> 31) | 0x80000000u);
}

// The 16 values of one face row: loaded first so the loads overlap, then compared in index order (strict: ties keep
// the lowest index).
template <typename T>
FORCE_INLINE void scan_row(uint32_t base, uint32_t col0, uint32_t& best_key, uint32_t& best_idx) {
    const volatile tt_l1_ptr T* v = reinterpret_cast<volatile tt_l1_ptr T*>(base);
    T vals[16];
#pragma GCC unroll 16
    for (uint32_t c = 0; c < 16; ++c) {
        vals[c] = v[c];
    }
#pragma GCC unroll 16
    for (uint32_t c = 0; c < 16; ++c) {
        const uint32_t k = sort_key(vals[c]);
        if (k > best_key) {
            best_key = k;
            best_idx = col0 + c;
        }
    }
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
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(3);  // 2: bf16, 4: fp32
    static_assert(
        users >= 1 && users <= 16, "users in rows 0-15 (8 entries of 2 uint32 per face row of a partial row)");
    static_assert(elem_bytes == 2 || elem_bytes == 4, "bf16 or fp32 logits");
    constexpr uint32_t tile_bytes = 1024 * elem_bytes;
    constexpr uint32_t face_bytes = 256 * elem_bytes;
    constexpr uint32_t row_bytes = users * 16 * elem_bytes;  // `users` face rows of 16 values

    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);  // per tile: face 0 rows | face 1 rows, then the out row
    const uint32_t out_l1 = scratch + max_tiles * 2 * row_bytes;

    const InterleavedAddrGen<true> logits_gen = {.bank_base_address = logits_addr, .page_size = tile_bytes};
    const uint32_t n = tile_end - tile_begin;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t tile = get_noc_addr(tile_begin + i, logits_gen);
        noc_async_read(tile, scratch + i * 2 * row_bytes, row_bytes);
        noc_async_read(tile + face_bytes, scratch + i * 2 * row_bytes + row_bytes, row_bytes);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t u = 0; u < users; ++u) {
        uint32_t best_key = 0;
        uint32_t best_idx = tile_begin * 32;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t base = scratch + i * 2 * row_bytes + u * 16 * elem_bytes;
            const uint32_t col0 = (tile_begin + i) * 32;
            if constexpr (elem_bytes == 2) {
                scan_row<uint16_t>(base, col0, best_key, best_idx);
                scan_row<uint16_t>(base + row_bytes, col0 + 16, best_key, best_idx);
            } else {
                scan_row<uint32_t>(base, col0, best_key, best_idx);
                scan_row<uint32_t>(base + row_bytes, col0 + 16, best_key, best_idx);
            }
        }
        out[u * 2 + 0] = best_key;
        out[u * 2 + 1] = best_idx;
    }
    constexpr uint32_t faces = users > 8 ? 2 : 1;  // face rows of the partial row
    for (uint32_t u = users; u < 8 * faces; ++u) {
        out[u * 2 + 0] = 0;
        out[u * 2 + 1] = 0;
    }

    // Row `core` of the uint32 partials tile: face 0 (rows 0-15) or face 2 (rows 16-31), columns 0-15; with more than
    // 8 users also the same row of face 1 or 3, columns 16-31.
    const InterleavedAddrGen<false> part_gen = {.bank_base_address = partials_addr, .page_size = 4096};
    const uint32_t offset = (core < 16 ? 0 : 2 * kFaceBytesU32) + (core % 16) * kRowBytesU32;
    noc_async_write(out_l1, get_noc_addr(0, part_gen, offset), kRowBytesU32);
    if constexpr (faces == 2) {
        noc_async_write(out_l1 + kRowBytesU32, get_noc_addr(0, part_gen, offset + kFaceBytesU32), kRowBytesU32);
    }
    noc_async_write_barrier();
}
