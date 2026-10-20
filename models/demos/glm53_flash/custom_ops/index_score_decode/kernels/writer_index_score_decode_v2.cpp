// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// index_score_decode writer v2 (custom_ops/index_score_decode/op.py): the v1 writer plus (a) -inf (bf16 0xFF80) for
// every column at or beyond the user's bound (valid_rows [B, 1] uint32: the pools at and beyond the user's
// incomplete pool never rank, so the top-k needs no per-row bound) and (b) an optional segmented output layout:
// seg_cols columns per page, page b * segs_per_user + segment (the two-stage top-k's per-segment rows).
// Per chunk: row 0 of each of the KC out tiles (two 32-byte face pieces) is assembled into one 64*KC-byte line,
// masked, and written at the chunk's column of the user's row / segment.
//
// compile-time args: [KC] + TensorAccessorArgs(out) + [seg_cols, segs_per_user] + TensorAccessorArgs(valid)
// runtime args: [item0, n_items, stride, out_addr, out_page_bytes, n_bands, band_chunks, total_chunks, valid_addr,
//                valid_page_bytes]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KC = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    constexpr uint32_t v2_base = out_args.next_compile_time_args_offset();
    constexpr uint32_t seg_cols = get_compile_time_arg_val(v2_base);  // elements per output page (W / segments)
    constexpr uint32_t segs_per_user = get_compile_time_arg_val(v2_base + 1);
    constexpr auto valid_args = TensorAccessorArgs<v2_base + 2>();
    constexpr uint32_t cb_out = 4, cb_line = 5, cb_valid = 7;
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t piece_words = 8;  // 32 bytes
    constexpr uint32_t line_bytes = KC * 64;
    constexpr uint32_t chunk_cols = KC * 32;
    constexpr uint16_t NEG_INF_BF16 = 0xFF80;

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t out_page_bytes = get_arg_val<uint32_t>(4);
    const uint32_t n_bands = get_arg_val<uint32_t>(5);
    const uint32_t band_chunks = get_arg_val<uint32_t>(6);
    const uint32_t total_chunks = get_arg_val<uint32_t>(7);
    const uint32_t valid_addr = get_arg_val<uint32_t>(8);
    const uint32_t valid_page_bytes = get_arg_val<uint32_t>(9);

    const auto out_acc = TensorAccessor(out_args, out_addr, out_page_bytes);
    const auto valid_acc = TensorAccessor(valid_args, valid_addr, valid_page_bytes);
    const uint32_t line_base = get_write_ptr(cb_line);  // 2 lines, alternated (scratch, never pushed)
    const uint32_t valid_l1 = get_write_ptr(cb_valid);

    uint32_t n = 0;
    for (uint32_t it = 0; it < n_items; ++it) {
        const uint32_t item = item0 + it * stride;
        const uint32_t b = item / n_bands;
        const uint32_t band = item % n_bands;
        const uint32_t c_begin = band * band_chunks;
        const uint32_t c_end = (c_begin + band_chunks < total_chunks) ? (c_begin + band_chunks) : total_chunks;
        if (c_begin >= c_end) {
            continue;
        }
        // the user's bound (pools): page b of valid_rows [B, 1]
        noc_async_read(valid_acc.get_noc_addr(b), valid_l1, valid_page_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_l1)[0];
        for (uint32_t c0 = c_begin; c0 < c_end; ++c0) {
            cb_wait_front(cb_out, KC);
            const uint32_t line = line_base + (n & 1u) * line_bytes;
            if (n >= 2) {
                noc_async_write_barrier();  // the line written two chunks ago has drained
            }
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(line);
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < KC; ++j) {
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + j * tile_bytes);
                for (uint32_t w = 0; w < piece_words; ++w) {
                    dst[j * 16 + w] = t[w];                                 // face 0, row 0: pools 0..15
                    dst[j * 16 + piece_words + w] = t[face_bytes / 4 + w];  // face 1, row 0: pools 16..31
                }
            }
            const uint32_t col0 = c0 * chunk_cols;
            if (valid < col0 + chunk_cols) {
                volatile tt_l1_ptr uint16_t* h = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(line);
                const uint32_t from = (valid > col0) ? (valid - col0) : 0;
                for (uint32_t e = from; e < chunk_cols; ++e) {
                    h[e] = NEG_INF_BF16;
                }
            }
            const uint32_t page = b * segs_per_user + col0 / seg_cols;
            const uint32_t off = (col0 % seg_cols) * 2;
            noc_async_write(line, out_acc.get_noc_addr(page, off), line_bytes);
            cb_pop_front(cb_out, KC);
            ++n;
        }
    }
    noc_async_write_barrier();
}
