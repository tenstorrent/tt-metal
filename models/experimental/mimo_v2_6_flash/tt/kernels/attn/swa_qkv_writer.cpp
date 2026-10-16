// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC of the SWA decode qkv split + RoPE + KV-cache update. Q cores (one per query head): row u of each RoPE'd
// tile goes to row `head` of the same tile of user u's slice of the height-sharded SDPA query input (one shard of
// [32 heads, head_dim] per user, on that user's shard core). K and V cores (one per user): row `user` of the new key
// (RoPE'd, cb_rot) or value (cb_x) replaces row `row` of the bf16 copies of the user's cache tiles in place; compute
// packs them back to BFP8 and they are written to the cache pages they came from.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t kRoleQ = 0, kRoleK = 1, kRoleV = 2;

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rot = get_compile_time_arg_val(2);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(3);
    constexpr uint32_t cb_c16 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sig = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out8 = get_compile_time_arg_val(6);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t users = get_compile_time_arg_val(8);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t row_bytes = 32;  // 16 bf16 of one face row

    const uint32_t head = get_arg_val<uint32_t>(0);  // Q: query head; K / V: user
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t cache_addr = get_arg_val<uint32_t>(2);
    // args 3.. (Q): NoC x, y of each user's query shard core

    if constexpr (role == kRoleQ) {
        cb_wait_front(cb_rot, d_tiles);
        const uint32_t rot = get_read_ptr(cb_rot);
        for (uint32_t u = 0; u < users; u++) {
            const uint32_t x = get_arg_val<uint32_t>(3 + 2 * u);
            const uint32_t y = get_arg_val<uint32_t>(4 + 2 * u);
            for (uint32_t j = 0; j < d_tiles; j++) {
                const uint32_t src = rot + j * tile_bytes + u * row_bytes;
                const uint32_t dst = q_addr + j * tile_bytes + head * row_bytes;
                noc_async_write(src, get_noc_addr(x, y, dst), row_bytes);
                noc_async_write(src + face_bytes, get_noc_addr(x, y, dst + face_bytes), row_bytes);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_rot, d_tiles);
        return;
    }

    const uint32_t user = head;
    constexpr uint32_t cb_new = role == kRoleK ? cb_rot : cb_x;
    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
    const uint32_t first_page = meta[0];
    const uint32_t row = meta[1];
    const bool active = meta[2] == 1;
    cb_pop_front(cb_meta, 1);

    // Row `user` of the new key / value tiles into row `row` of the cache tiles (bf16, 32 x 32 in four 16 x 16
    // faces: row r's left half is face (r / 16) * 2, its right half the next face).
    cb_wait_front(cb_new, d_tiles);
    cb_wait_front(cb_c16, d_tiles);
    const uint32_t src_l1 = get_read_ptr(cb_new);
    const uint32_t dst_l1 = get_read_ptr(cb_c16);
    const uint32_t dst_row = (row / 16) * 2 * face_bytes + (row % 16) * row_bytes;
    for (uint32_t j = 0; j < d_tiles; j++) {
        volatile tt_l1_ptr uint32_t* src =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1 + j * tile_bytes + user * row_bytes);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1 + j * tile_bytes + dst_row);
        for (uint32_t w = 0; w < row_bytes / 4; w++) {
            dst[w] = src[w];
            dst[face_bytes / 4 + w] = src[face_bytes / 4 + w];
        }
    }
    cb_pop_front(cb_new, d_tiles);
    cb_reserve_back(cb_sig, 1);
    cb_push_back(cb_sig, 1);

    cb_wait_front(cb_out8, d_tiles);
    const uint32_t out_l1 = get_read_ptr(cb_out8);
    const InterleavedAddrGen<true> cache = {.bank_base_address = cache_addr, .page_size = cache_tile_bytes};
    for (uint32_t j = 0; active && j < d_tiles; j++) {
        noc_async_write(out_l1 + j * cache_tile_bytes, cache.get_noc_addr(first_page + j, 0), cache_tile_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out8, d_tiles);
}
