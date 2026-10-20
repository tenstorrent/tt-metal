// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// index_score_decode reader (custom_ops/index_score_decode/op.py). Work item = (user b, k-band): the user's
// query block (Hi = 32 index heads as the 32 rows of Dt tiles), a "w-row tile" (row 0 = the user's Hi head
// weights, the rest zero) and then the band's pooled-key tiles streamed d-major per chunk of KC k-tiles
// (CB slot i*KC + c = d-tile i of k-tile c: the layout matmul_block wants for a kt x ct B block).
//
// compile-time args: [Dt, KC, Tt, k_tile_bytes (2048 bf16 / 1088 bfp8_b)] + TensorAccessorArgs(q, k, w)
// runtime args: [item0, n_items, stride, q_addr, k_addr, w_addr, n_bands, band_chunks, total_chunks]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t KC = get_compile_time_arg_val(1);
    constexpr uint32_t Tt = get_compile_time_arg_val(2);
    constexpr uint32_t k_tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t slot_base = get_compile_time_arg_val(4);    // user b's pool slot = slot_base + b * slot_stride
    constexpr uint32_t slot_stride = get_compile_time_arg_val(5);  // (1: user b = slot b; 0: every user = slot_base)
    constexpr uint32_t slot_pages = get_compile_time_arg_val(6);   // tile pages per slot (the cache width x Dt)
    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_w = 2, cb_wfaces = 6;
    constexpr uint32_t tile_bytes = 2048;  // bf16 32x32
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t row_bytes = 32;  // 16 bf16 of one face row

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t q_addr = get_arg_val<uint32_t>(3);
    const uint32_t k_addr = get_arg_val<uint32_t>(4);
    const uint32_t w_addr = get_arg_val<uint32_t>(5);
    const uint32_t n_bands = get_arg_val<uint32_t>(6);
    const uint32_t band_chunks = get_arg_val<uint32_t>(7);
    const uint32_t total_chunks = get_arg_val<uint32_t>(8);

    const auto q_acc = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k_acc = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, tile_bytes);

    for (uint32_t it = 0; it < n_items; ++it) {
        const uint32_t item = item0 + it * stride;
        const uint32_t b = item / n_bands;
        const uint32_t band = item % n_bands;
        const uint32_t c_begin = band * band_chunks;
        const uint32_t c_end = (c_begin + band_chunks < total_chunks) ? (c_begin + band_chunks) : total_chunks;
        if (c_begin >= c_end) {
            continue;  // the compute / writer skip the same items
        }
        // the user's query block: Dt tiles [32 heads x 32 dims] (q [1, B, 32, D] TILE: tile (b, i) = page b*Dt + i)
        cb_reserve_back(cb_q, Dt);
        {
            const uint32_t l1 = get_write_ptr(cb_q);
            for (uint32_t i = 0; i < Dt; ++i) {
                noc_async_read(q_acc.get_noc_addr(b * Dt + i), l1 + i * tile_bytes, tile_bytes);
            }
        }
        // the w-row tile: zero, then row 0 of each face column = row b of the w tile (w [1,1,B,Hi] TILE, B <= 32 per
        // tile page: page b/32, face row-group (b%32)/16, row b%16). DRAM reads are 64-byte aligned, so the row's two
        // faces (1 KB, contiguous) land in a scratch CB and the two 32-byte row pieces are copied by the RISC.
        cb_reserve_back(cb_w, 1);
        {
            const uint32_t l1 = get_write_ptr(cb_w);
            const uint32_t faces = get_write_ptr(cb_wfaces);
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
            for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
                z[i] = 0;
            }
            const uint32_t page = b / 32;
            const uint32_t r = b % 32;
            noc_async_read(w_acc.get_noc_addr(page, (r / 16) * 2 * face_bytes), faces, 2 * face_bytes);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(faces);
            const uint32_t rw = (r % 16) * (row_bytes / 4);  // the row's first word inside a face
            for (uint32_t i = 0; i < row_bytes / 4; ++i) {
                z[i] = f[rw + i];                                    // face 0 row 0 <- face (r/16)*2, row r%16
                z[face_bytes / 4 + i] = f[face_bytes / 4 + rw + i];  // face 1 row 0 <- face (r/16)*2 + 1, row r%16
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, Dt);
        cb_push_back(cb_w, 1);
        // the band's pooled keys, KC k-tiles per chunk, d-major in the CB
        const uint32_t k_base = (slot_base + b * slot_stride) * slot_pages;
        for (uint32_t c0 = c_begin; c0 < c_end; ++c0) {
            cb_reserve_back(cb_k, KC * Dt);
            const uint32_t l1 = get_write_ptr(cb_k);
            for (uint32_t c = 0; c < KC; ++c) {
                const uint32_t kt = c0 * KC + c;
                for (uint32_t i = 0; i < Dt; ++i) {
                    noc_async_read(
                        k_acc.get_noc_addr(k_base + kt * Dt + i), l1 + (i * KC + c) * k_tile_bytes, k_tile_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_k, KC * Dt);
        }
    }
}
