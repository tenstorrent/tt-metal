// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC of the SWA decode qkv split + RoPE + KV-cache update (SwaDecodeQkv in tt/attention.py). Every core reads
// the user rows 0..rows-1 of the six [*, 32] tiles of its slice of the fused qkv row (role Q: query head `head`, K:
// the key head, V: the value head) into cb_x: those rows of faces 0 and 1 land where they were, so a tile's row u is
// still user u. Q and K cores also read the user rows of the cos / sin tiles and the RoPE transformation tile. K and V
// cores then read their user's cache tiles: physical block = page_table[user][pos / block] (the table is indexed by
// absolute position), the six tiles of the block's tile row holding pos (an inactive user, pos < 0, reads block 0
// and is flagged so the writer leaves the cache alone).
//
// Users whose positions fall in the same cache tile row (a speculative verify's slots: one request at consecutive
// positions) are written by one core, the lowest such user's: it gets every member's row and user to patch in, the
// others write nothing. One launch then serves a verify, whose slots the writers would otherwise race on.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t kRoleQ = 0, kRoleK = 1, kRoleV = 2;

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(3);
    constexpr uint32_t cb_trans = get_compile_time_arg_val(4);
    constexpr uint32_t cb_cache = get_compile_time_arg_val(5);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(6);  // pos / page-table scratch
    constexpr uint32_t d_tiles = get_compile_time_arg_val(7);  // 6 (V cores: the V cache's 4 real columns)
    constexpr uint32_t block_size = get_compile_time_arg_val(8);
    constexpr uint32_t pos_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t pt_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t rows = get_compile_time_arg_val(12);   // the user rows: 4 or 8
    constexpr uint32_t users = get_compile_time_arg_val(13);  // the users with a K / V core
    constexpr uint32_t pt_chunk_bytes = 64;  // the aligned piece of the page table row read for one entry
    constexpr uint32_t tile_bytes = 2048;    // bf16 32x32
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t rows_bytes = rows * 32;  // the user rows of a bf16 face
    constexpr uint32_t row_tiles = block_size / 32;

    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile = get_arg_val<uint32_t>(1);  // this core's first qkv tile column
    const uint32_t cos_addr = get_arg_val<uint32_t>(2);
    const uint32_t sin_addr = get_arg_val<uint32_t>(3);
    const uint32_t trans_addr = get_arg_val<uint32_t>(4);
    const uint32_t user = get_arg_val<uint32_t>(5);
    const uint32_t pos_addr = get_arg_val<uint32_t>(6);
    const uint32_t pt_addr = get_arg_val<uint32_t>(7);
    const uint32_t cache_addr = get_arg_val<uint32_t>(8);

    const InterleavedAddrGen<false> qkv = {.bank_base_address = qkv_addr, .page_size = tile_bytes};
    cb_reserve_back(cb_x, d_tiles);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t j = 0; j < d_tiles; j++) {
        noc_async_read(qkv.get_noc_addr(first_tile + j, 0), x_l1 + j * tile_bytes, rows_bytes);
        noc_async_read(qkv.get_noc_addr(first_tile + j, face_bytes), x_l1 + j * tile_bytes + face_bytes, rows_bytes);
    }
    if constexpr (role != kRoleV) {
        const InterleavedAddrGen<true> cos = {.bank_base_address = cos_addr, .page_size = tile_bytes};
        const InterleavedAddrGen<true> sin = {.bank_base_address = sin_addr, .page_size = tile_bytes};
        cb_reserve_back(cb_cos, d_tiles);
        cb_reserve_back(cb_sin, d_tiles);
        const uint32_t cos_l1 = get_write_ptr(cb_cos);
        const uint32_t sin_l1 = get_write_ptr(cb_sin);
        for (uint32_t j = 0; j < d_tiles; j++) {
            noc_async_read(cos.get_noc_addr(j, 0), cos_l1 + j * tile_bytes, rows_bytes);
            noc_async_read(cos.get_noc_addr(j, face_bytes), cos_l1 + j * tile_bytes + face_bytes, rows_bytes);
            noc_async_read(sin.get_noc_addr(j, 0), sin_l1 + j * tile_bytes, rows_bytes);
            noc_async_read(sin.get_noc_addr(j, face_bytes), sin_l1 + j * tile_bytes + face_bytes, rows_bytes);
        }
        const InterleavedAddrGen<true> trans = {.bank_base_address = trans_addr, .page_size = tile_bytes};
        cb_reserve_back(cb_trans, 1);
        noc_async_read(trans.get_noc_addr(0, 0), get_write_ptr(cb_trans), tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, d_tiles);
    if constexpr (role != kRoleV) {
        cb_push_back(cb_cos, d_tiles);
        cb_push_back(cb_sin, d_tiles);
        cb_push_back(cb_trans, 1);
    }

    if constexpr (role != kRoleQ) {
        // Every user's position and cache tile row (block * row_tiles + tile row); this user's cache tiles.
        cb_reserve_back(cb_meta, 1);
        const uint32_t meta = get_write_ptr(cb_meta);
        const InterleavedAddrGen<true> pos_gen = {.bank_base_address = pos_addr, .page_size = pos_page_bytes};
        noc_async_read(pos_gen.get_noc_addr(0, 0), meta, pos_page_bytes);
        noc_async_read_barrier();
        int32_t pos[users];
        uint32_t entry_word[users];
        const InterleavedAddrGen<true> pt_gen = {.bank_base_address = pt_addr, .page_size = pt_page_bytes};
        const uint32_t chunks = meta + pos_page_bytes;  // one pt_chunk_bytes piece per user
        for (uint32_t v = 0; v < users; v++) {
            pos[v] = reinterpret_cast<volatile tt_l1_ptr int32_t*>(meta)[v];
            entry_word[v] = 0;
            if (pos[v] >= 0) {
                const uint32_t entry = static_cast<uint32_t>(pos[v]) / block_size;
                const uint32_t chunk = (entry * 4) & ~(pt_chunk_bytes - 1);
                noc_async_read(pt_gen.get_noc_addr(v, chunk), chunks + v * pt_chunk_bytes, pt_chunk_bytes);
                entry_word[v] = (entry * 4 - chunk) / 4;
            }
        }
        noc_async_read_barrier();
        uint32_t tile_key[users];
        for (uint32_t v = 0; v < users; v++) {
            tile_key[v] = 0;
            if (pos[v] >= 0) {
                const uint32_t block =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chunks + v * pt_chunk_bytes)[entry_word[v]];
                tile_key[v] = block * row_tiles + (static_cast<uint32_t>(pos[v]) % block_size) / 32;
            }
        }
        const bool active = pos[user] >= 0;
        bool leader = active;
        for (uint32_t v = 0; v < user; v++) {
            leader = leader && !(pos[v] >= 0 && tile_key[v] == tile_key[user]);
        }
        // Handed to the writer: [first cache page, members (0: write nothing), (row, user) per member].
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
        const uint32_t first_page = tile_key[user] * d_tiles;
        uint32_t members = 0;
        if (leader) {
            for (uint32_t v = user; v < users; v++) {
                if (pos[v] >= 0 && tile_key[v] == tile_key[user]) {
                    out[2 + 2 * members] = static_cast<uint32_t>(pos[v]) % 32;
                    out[3 + 2 * members] = v;
                    members++;
                }
            }
        }
        const InterleavedAddrGen<true> cache = {.bank_base_address = cache_addr, .page_size = cache_tile_bytes};
        cb_reserve_back(cb_cache, d_tiles);
        const uint32_t cache_l1 = get_write_ptr(cb_cache);
        for (uint32_t j = 0; j < d_tiles; j++) {
            noc_async_read(cache.get_noc_addr(first_page + j, 0), cache_l1 + j * cache_tile_bytes, cache_tile_bytes);
        }
        out[0] = first_page;
        out[1] = members;
        noc_async_read_barrier();
        cb_push_back(cb_cache, d_tiles);
        cb_push_back(cb_meta, 1);
    }
}
