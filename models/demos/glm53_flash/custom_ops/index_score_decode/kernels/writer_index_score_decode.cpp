// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// index_score_decode writer (custom_ops/index_score_decode/op.py). Per chunk: row 0 of each of the KC out tiles
// (face 0 row 0 = pools 0..15, face 1 row 0 = pools 16..31: two 32-byte pieces) is assembled into one
// 64*KC-byte line and written to the user's score row at the chunk's column offset.
//
// compile-time args: [KC] + TensorAccessorArgs(out)
// runtime args: [item0, n_items, stride, out_addr, out_row_bytes, n_bands, band_chunks, total_chunks]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KC = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_out = 4, cb_line = 5;
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t piece_words = 8;  // 32 bytes
    constexpr uint32_t line_bytes = KC * 64;

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t out_row_bytes = get_arg_val<uint32_t>(4);
    const uint32_t n_bands = get_arg_val<uint32_t>(5);
    const uint32_t band_chunks = get_arg_val<uint32_t>(6);
    const uint32_t total_chunks = get_arg_val<uint32_t>(7);

    const auto out_acc = TensorAccessor(out_args, out_addr, out_row_bytes);
    const uint32_t line_base = get_write_ptr(cb_line);  // 2 lines, alternated (scratch, never pushed)

    uint32_t n = 0;
    for (uint32_t it = 0; it < n_items; ++it) {
        const uint32_t item = item0 + it * stride;
        const uint32_t b = item / n_bands;
        const uint32_t band = item % n_bands;
        const uint32_t c_begin = band * band_chunks;
        const uint32_t c_end = (c_begin + band_chunks < total_chunks) ? (c_begin + band_chunks) : total_chunks;
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
            noc_async_write(line, out_acc.get_noc_addr(b, c0 * line_bytes), line_bytes);
            cb_pop_front(cb_out, KC);
            ++n;
        }
    }
    noc_async_write_barrier();
}
