// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Row attention, reader (NCRISC) (dflash/row_attention.py): one core per (verify row r, kv head g). Pushes, in order:
//   cb_q: row r's two Q tiles ([32 head rows, 32 dims] x 2; heads in rows, BF16);
//   cb_aux: [0 scaler (1.0 in row 0 of each face), 1 first-chunk mask, 2 last-chunk mask, 3 single-chunk mask,
//            4 sink-column mask (keeps column 0, -1e30 at columns 1..31: compute adds it to the sink tile),
//            5 sink tile (column 0: the heads' sink logits / scale), 6 zero tile (the mask of a chunk without a cut)];
//   cb_k: the nk 32-position K chunks of kv head g ([32 positions, 32 dims] x 2 per chunk), then cb_v likewise.
// Positions: row r attends [lo, p] with p = pos[r] (lo = max(0, p + 1 - window) with a sliding window, else 0);
// chunk j covers positions c0 + 32 j .. c0 + 32 j + 31 (c0 = lo rounded down to 32). Masks are -inf at positions
// outside [lo, p] (full tiles: every row the same).
//
// runtime args: [q_addr, k_addr, v_addr, pos_addr, pt_addr, sink_addr, r, g]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_arg_val<uint32_t>(3);
    const uint32_t pt_addr = get_arg_val<uint32_t>(4);
    const uint32_t sink_addr = get_arg_val<uint32_t>(5);
    const uint32_t r = get_arg_val<uint32_t>(6);
    const uint32_t g = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_aux = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t nkv = get_compile_time_arg_val(5);
    constexpr uint32_t block = get_compile_time_arg_val(6);   // positions per page-table block
    constexpr uint32_t window = get_compile_time_arg_val(7);  // 0: full attention
    constexpr uint32_t kv_tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t pt_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t pos_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t max_chunks = get_compile_time_arg_val(11);
    constexpr auto q_args = TensorAccessorArgs<12>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto sink_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile = 2048;       // BF16 tile
    constexpr uint32_t tpb = block / 32;  // 32-position tiles per block
    const auto s_q = TensorAccessor(q_args, q_addr, tile);
    const auto s_k = TensorAccessor(k_args, k_addr, kv_tile_bytes);
    const auto s_v = TensorAccessor(v_args, v_addr, kv_tile_bytes);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, pos_page_bytes);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, pt_page_bytes);
    const auto s_sink = TensorAccessor(sink_args, sink_addr, tile);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);  // [pos page | page-table row]
    noc_async_read(s_pos.get_noc_addr(0), scr, pos_page_bytes);
    cb_reserve_back(cb_q, 2);
    cb_reserve_back(cb_aux, 7);
    const uint32_t aux = get_write_ptr(cb_aux);
    noc_async_read(s_sink.get_noc_addr(0), aux + 5 * tile, tile);
    // scaler tile: zero, then 1.0 in row 0 of each face
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < tile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, aux + z, MEM_ZEROS_SIZE);             // scaler (tile 0)
        noc_async_read(zeros, aux + 6 * tile + z, MEM_ZEROS_SIZE);  // zero mask (tile 6)
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const int32_t p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[r];
    const int32_t lo_ = window ? p + 1 - static_cast<int32_t>(window) : 0;
    const uint32_t lo = lo_ > 0 ? static_cast<uint32_t>(lo_) : 0;
    const uint32_t c0 = lo & ~31u;
    uint32_t nk = (static_cast<uint32_t>(p) - c0) / 32 + 1;
    // never overrun the CBs (a windowed row spans at most window / 32 + 1 chunks, which is what dflash/verify.py sizes
    // max_chunks to: no row is cut here)
    if (nk > max_chunks) {
        nk = max_chunks;
    }
    // only the page-table entries of positions [c0, p] (every core reading the whole row would hit one DRAM bank with
    // 16 x the row at long context), read from the 64-byte boundary at or below the first
    const uint32_t a0 = (c0 / block * 4) & ~63u;
    const uint32_t a1 = ((static_cast<uint32_t>(p) / block + 1) * 4 + 63) & ~63u;
    noc_async_read(s_pt.get_noc_addr(0) + a0, scr + 256, (a1 < pt_page_bytes ? a1 : pt_page_bytes) - a0);
    const uint32_t pt_first = a0 / 4;
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t first_cut = lo - c0;                              // masked columns at the start of chunk 0
    const uint32_t last_keep = (static_cast<uint32_t>(p) & 31) + 1;  // valid columns of the last chunk

    // K / V chunks (one barrier below)
    volatile tt_l1_ptr int32_t* pt = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr + 256);
    cb_reserve_back(cb_k, 2 * nk);
    cb_reserve_back(cb_v, 2 * nk);
    const uint32_t kl1 = get_write_ptr(cb_k), vl1 = get_write_ptr(cb_v);
    // fused into the QKV program (dflash/verify.py): the chunks entirely before the block's first position are in the
    // cache already (read now, while QKV runs); Q and the chunks holding the block's rows come after the QKV writers
    // signal (RA_WAIT_N increments of program semaphore RA_WAIT_SEM)
    const uint32_t start = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[0]);
    uint32_t j_new = 0;
    while (j_new < nk && c0 + 32 * j_new + 31 < start) {
        ++j_new;
    }
    auto read_chunk = [&](uint32_t j) {
        const uint32_t pos = c0 + 32 * j;
        const uint32_t blk = static_cast<uint32_t>(pt[pos / block - pt_first]);
        const uint32_t t0 = ((blk * nkv + g) * tpb + (pos % block) / 32) * 2;
        for (uint32_t d = 0; d < 2; ++d) {
            noc_async_read(s_k.get_noc_addr(t0 + d), kl1 + (2 * j + d) * kv_tile_bytes, kv_tile_bytes);
            noc_async_read(s_v.get_noc_addr(t0 + d), vl1 + (2 * j + d) * kv_tile_bytes, kv_tile_bytes);
        }
    };
    for (uint32_t j = 0; j < j_new; ++j) {
        read_chunk(j);
    }

    volatile tt_l1_ptr uint16_t* sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(aux);
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t c = 0; c < 16; ++c) {
            sc[f * 256 + c] = 0x3F80;
        }
    }
    // masks (full tiles, every row the same; word stores of the row pattern): 1 first chunk, 2 last chunk, 3 single
    // chunk, 4 the sink tile's columns 1..31 (added to the sink tile's column-0 logits)
    auto mask = [&](uint32_t t, uint32_t cut, uint32_t keep) {
        uint32_t w[2][8];
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t c0 = h * 16 + 2 * i, c1 = c0 + 1;
                const uint32_t lo = (c0 < cut || c0 >= keep) ? 0xF14Au : 0u;  // BF16 -1e30
                const uint32_t hi = (c1 < cut || c1 >= keep) ? 0xF14Au : 0u;
                w[h][i] = lo | (hi << 16);
            }
        }
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aux + t * tile);
        for (uint32_t f = 0; f < 4; ++f) {
            const uint32_t* pw = w[f & 1];
            for (uint32_t row = 0; row < 16; ++row) {
                volatile tt_l1_ptr uint32_t* q = m + f * 128 + row * 8;
                q[0] = pw[0];
                q[1] = pw[1];
                q[2] = pw[2];
                q[3] = pw[3];
                q[4] = pw[4];
                q[5] = pw[5];
                q[6] = pw[6];
                q[7] = pw[7];
            }
        }
    };
    // only the masks compute uses (row_attn_writer.cpp: a chunk without a cut takes the zero tile)
    if (nk == 1) {
        if (first_cut || last_keep < 32) {
            mask(3, first_cut, last_keep);
        }
    } else {
        if (first_cut) {
            mask(1, first_cut, 32);
        }
        if (last_keep < 32) {
            mask(2, 0, last_keep);
        }
    }
    mask(4, 0, 1);  // keep column 0 (the sink logit)
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(RA_WAIT_SEM)), RA_WAIT_N);
    noc_async_read(s_q.get_noc_addr(2 * r), get_write_ptr(cb_q), tile);
    noc_async_read(s_q.get_noc_addr(2 * r + 1), get_write_ptr(cb_q) + tile, tile);
    // Q and the older chunks (read while QKV ran) go to compute first; the block's chunks follow (compute scores
    // chunk by chunk)
    cb_push_back(cb_aux, 7);
    noc_async_read_barrier();
    cb_push_back(cb_q, 2);
    cb_push_back(cb_k, 2 * j_new);
    cb_push_back(cb_v, 2 * j_new);
    for (uint32_t j = j_new; j < nk; ++j) {
        read_chunk(j);
    }
    noc_async_read_barrier();
    cb_push_back(cb_k, 2 * (nk - j_new));
    cb_push_back(cb_v, 2 * (nk - j_new));
}
