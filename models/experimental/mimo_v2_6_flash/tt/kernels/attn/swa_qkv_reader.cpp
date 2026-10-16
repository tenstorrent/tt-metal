// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC of the SWA decode qkv split + RoPE + KV-cache update (SwaDecodeQkv in tt/attention.py). Every core reads
// rows 0-3 (the users) of the six [*, 32] tiles of its slice of the fused qkv row (role Q: query head `head`, K: the
// key head, V: the value head) into cb_x: rows 0-3 of faces 0 and 1 land where they were, so a tile's row u is
// still user u. Q and K cores also read rows 0-3 of the cos / sin tiles and the RoPE transformation tile. K and V
// cores then read their user's cache tiles: physical block = page_table[user][pos / block] (the table is indexed by
// absolute position), the six tiles of the block's tile row holding pos (an inactive user, pos < 0, reads block 0
// and is flagged so the writer leaves the cache alone).
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
    constexpr uint32_t d_tiles = get_compile_time_arg_val(7);  // 6
    constexpr uint32_t block_size = get_compile_time_arg_val(8);
    constexpr uint32_t pos_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t pt_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t cache_tile_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t pt_chunk_bytes = 64;  // the aligned piece of the page table row read for one entry
    constexpr uint32_t tile_bytes = 2048;  // bf16 32x32
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t rows_bytes = 128;  // rows 0-3 of a bf16 face
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
        // This user's position and the cache tiles that hold it.
        cb_reserve_back(cb_meta, 1);
        const uint32_t meta = get_write_ptr(cb_meta);
        const InterleavedAddrGen<true> pos_gen = {.bank_base_address = pos_addr, .page_size = pos_page_bytes};
        noc_async_read(pos_gen.get_noc_addr(0, 0), meta, pos_page_bytes);
        noc_async_read_barrier();
        const int32_t pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(meta)[user];
        uint32_t block = 0;
        uint32_t tile_row = 0;
        if (pos >= 0) {
            const uint32_t entry = static_cast<uint32_t>(pos) / block_size;
            const uint32_t chunk = (entry * 4) & ~(pt_chunk_bytes - 1);
            const InterleavedAddrGen<true> pt_gen = {.bank_base_address = pt_addr, .page_size = pt_page_bytes};
            noc_async_read(pt_gen.get_noc_addr(user, chunk), meta + pos_page_bytes, pt_chunk_bytes);
            noc_async_read_barrier();
            block = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta + pos_page_bytes)[(entry * 4 - chunk) / 4];
            tile_row = (static_cast<uint32_t>(pos) % block_size) / 32;
        }
        // Handed to the writer: [first cache page, row within the tile, active].
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
        const uint32_t first_page = (block * row_tiles + tile_row) * d_tiles;
        const uint32_t row = pos >= 0 ? static_cast<uint32_t>(pos) % 32 : 0;
        const InterleavedAddrGen<true> cache = {.bank_base_address = cache_addr, .page_size = cache_tile_bytes};
        cb_reserve_back(cb_cache, d_tiles);
        const uint32_t cache_l1 = get_write_ptr(cb_cache);
        for (uint32_t j = 0; j < d_tiles; j++) {
            noc_async_read(cache.get_noc_addr(first_page + j, 0), cache_l1 + j * cache_tile_bytes, cache_tile_bytes);
        }
        out[0] = first_page;
        out[1] = row;
        out[2] = pos >= 0 ? 1 : 0;
        noc_async_read_barrier();
        cb_push_back(cb_cache, d_tiles);
        cb_push_back(cb_meta, 1);
    }
}
