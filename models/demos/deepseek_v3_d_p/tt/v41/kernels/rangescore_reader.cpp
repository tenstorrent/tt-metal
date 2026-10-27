// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the V4.1 single-query range scoring (rangescore_compute.cpp, tt/v41/indexer_kernels.py range_scores).
//
// One user's query against its own index keys in the decode pool: every core reads the same query row (bf16
// row-major [R, 32 * 128]: one [32 heads, 128] block, pushed as 4 tile-sized pages for a tilize) and head weights
// (bf16 row-major [R, 32], row 0 of a 32 x 32 row-major page whose other rows are zero), then the key tile rows of
// its chunks straight from the TILED pool ([users, 1, TR * 32, 128] bf16: key tile row g of user u is the four
// tile pages (u * TR + g) * 4 + d), which are the [rows, dims] K tiles the compute consumes (no tilize). The key
// tile rows are chunked by CHUNK; chunk c belongs to core c mod NCORES. The visible bound is entry `user` of the pin
// page (uint32 [1, 1, 1, PIN_ROWS]: the user's newest visible row, SENTINEL for none): visible tile rows
// vt = ceil((newest + 1) / 32), pushed to the compute through cb_meta (word 0); chunks at or past vt are not read.
// With DYN (the one-live-row capture) the user is not a runtime argument but word 0 of the pin
// page and its newest word 2 (a per-step host row: a capture must not bake in the slot); a user past the tile's
// rows reads row 0 of the query (the newest is then the sentinel: nothing is read).
//
// compile_time_args = [cb_qrm, cb_wrm, cb_k, cb_meta, CHUNK, TR, NCORES, PIN_ROWS, DYN, TensorAccessorArgs(q)...,
//                      TensorAccessorArgs(w)..., TensorAccessorArgs(k)..., TensorAccessorArgs(pin)...]
// runtime args = [q_addr, w_addr, k_addr, pin_addr, user, core_index]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t pin_addr = get_arg_val<uint32_t>(3);
    const uint32_t user_arg = get_arg_val<uint32_t>(4);
    const uint32_t core = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_qrm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_wrm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_k = get_compile_time_arg_val(2);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(3);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(4);   // key tile rows per chunk
    constexpr uint32_t TR = get_compile_time_arg_val(5);      // key tile rows per user (the score width / 32)
    constexpr uint32_t NCORES = get_compile_time_arg_val(6);  // chunk c on core c % NCORES
    constexpr uint32_t PIN_ROWS = get_compile_time_arg_val(7);
    constexpr uint32_t DYN = get_compile_time_arg_val(8);  // 1: the user is word 0 of the pin page, its newest word 2
    constexpr auto q_args = TensorAccessorArgs<9>();
    constexpr auto w_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto pin_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t DIM_TILES = 4;  // 128 dims
    constexpr uint32_t TILE_BYTES = 32 * 32 * 2;
    constexpr uint32_t Q_BYTES = 32 * 128 * 2;  // 32 heads x 128
    constexpr uint32_t W_BYTES = 32 * 2;        // 32 head weights
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    constexpr uint32_t NCHUNKS = (TR + CHUNK - 1) / CHUNK;

    const auto q = TensorAccessor(q_args, q_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto pin = TensorAccessor(pin_args, pin_addr);

    DataflowBuffer qrm(cb_qrm);
    DataflowBuffer wrm(cb_wrm);
    DataflowBuffer kt(cb_k);
    DataflowBuffer meta(cb_meta);

    // the bound: the pin page into the meta page (64-byte aligned, a DRAM read), then word 0 = visible tile rows
    meta.reserve_back(1);
    const uint32_t meta_l1 = meta.get_write_ptr();
    const uint32_t pin_l1 = (meta_l1 + 63) & ~63u;  // the page holds 64 + PIN_SLOT bytes
    noc_async_read(pin.get_noc_addr(0, 0), pin_l1, PIN_ROWS * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1);
    volatile tt_l1_ptr uint32_t* pins = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pin_l1);
    const uint32_t user = DYN ? (pins[0] < 32 ? pins[0] : 0) : user_arg;
    const uint32_t newest = DYN ? pins[2] : pins[user];
    const uint32_t visible = newest == SENTINEL ? 0 : newest + 1;
    uint32_t vt = (visible + 31) / 32;
    if (vt > TR) {
        vt = TR;
    }
    words[0] = vt;
    meta.push_back(1);

    // the query (both pages of cb_wrm: zero rows 1..31 once; row 0 is the user's weights)
    {
        volatile tt_l1_ptr uint32_t* wz = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wrm.get_write_ptr());
        for (uint32_t i = 0; i < 2 * TILE_BYTES / 4; ++i) {
            wz[i] = 0;
        }
    }
    qrm.reserve_back(DIM_TILES);
    noc_async_read(q.get_noc_addr(user, 0), qrm.get_write_ptr(), Q_BYTES);
    wrm.reserve_back(1);
    noc_async_read(w.get_noc_addr(user, 0), wrm.get_write_ptr(), W_BYTES);
    noc_async_read_barrier();
    qrm.push_back(DIM_TILES);
    wrm.push_back(1);

    const uint32_t tile_base = user * TR * DIM_TILES;
    for (uint32_t c = core; c < NCHUNKS; c += NCORES) {
        const uint32_t t0 = c * CHUNK;
        if (t0 >= vt) {
            break;
        }
        uint32_t t1 = t0 + CHUNK;
        if (t1 > vt) {
            t1 = vt;
        }
        // whole chunks of pages always (a partial push would leave the CB's pointers off the wrap); the compute
        // reads the first t1 - t0 tile rows
        kt.reserve_back(DIM_TILES * CHUNK);
        uint32_t l1 = kt.get_write_ptr();
        for (uint32_t id = tile_base + t0 * DIM_TILES; id < tile_base + t1 * DIM_TILES; ++id) {
            noc_async_read(k.get_noc_addr(id, 0), l1, TILE_BYTES);
            l1 += TILE_BYTES;
        }
        noc_async_read_barrier();
        kt.push_back(DIM_TILES * CHUNK);
    }
}
