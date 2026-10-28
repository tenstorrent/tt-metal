// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// ConcatHeadsDecode: per work unit (column tile c, head group g: heads g*hg .. min(H, (g+1)*hg)) read the B input
// tiles (b, c) (page b*cols + c), build the group's output tiles (h, c): row b <- row h of input tile b, rows >= B
// zero, and write them to page h*cols + c. Unit id = c * groups + g.
// A tile row r lives in faces (r/16)*2 (columns 0-15) and +1 (columns 16-31) at word offset (r%16)*8.
// compile-time args: [cb_in, cb_out, B, H, cols, hg, groups] + TensorAccessorArgs(x, out)
// runtime: [x_addr, out_addr, first unit, unit count]
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t hg = get_compile_time_arg_val(5);
    constexpr uint32_t groups = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;
    const auto s_x = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t in_base = get_write_ptr(cb_in);
    const uint32_t out_base = get_write_ptr(cb_out);
    volatile tt_l1_ptr uint32_t* ob = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_base);
    // rows >= B of every output tile are zero: clear the padding rows of the hg staging tiles once (the copies
    // below rewrite rows < B of every tile)
    for (uint32_t t = 0; t < hg; ++t) {
        volatile tt_l1_ptr uint32_t* ot = ob + t * (tile_bytes / 4);
        for (uint32_t b = B; b < 32; ++b) {
            const uint32_t face = (b >> 4) * 2, word = (b & 15) * 8;
            for (uint32_t j = 0; j < 8; ++j) {
                ot[face * 128 + word + j] = 0;
                ot[(face + 1) * 128 + word + j] = 0;
            }
        }
    }
    uint32_t last_c = 0xFFFFFFFFu;
    for (uint32_t u = u0; u < u0 + n; ++u) {
        const uint32_t c = u / groups;
        const uint32_t g = u - c * groups;
        const uint32_t h0 = g * hg;
        const uint32_t h1 = (h0 + hg < H) ? h0 + hg : H;
        if (c != last_c) {  // the column's B input tiles (reused across this core's head groups of one column)
            for (uint32_t b = 0; b < B; ++b) {
                noc_async_read(s_x.get_noc_addr(b * cols + c), in_base + b * tile_bytes, tile_bytes);
            }
            noc_async_read_barrier();
            last_c = c;
        }
        for (uint32_t h = h0; h < h1; ++h) {
            volatile tt_l1_ptr uint32_t* ot = ob + (h - h0) * (tile_bytes / 4);
            const uint32_t src_face = (h >> 4) * 2, src_word = (h & 15) * 8;
            for (uint32_t b = 0; b < B; ++b) {
                volatile tt_l1_ptr uint32_t* it =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base + b * tile_bytes);
                const uint32_t dst_face = (b >> 4) * 2, dst_word = (b & 15) * 8;
                for (uint32_t j = 0; j < 8; ++j) {
                    ot[dst_face * 128 + dst_word + j] = it[src_face * 128 + src_word + j];
                    ot[(dst_face + 1) * 128 + dst_word + j] = it[(src_face + 1) * 128 + src_word + j];
                }
            }
        }
        for (uint32_t h = h0; h < h1; ++h) {
            noc_async_write(out_base + (h - h0) * tile_bytes, s_out.get_noc_addr(h * cols + c), tile_bytes);
        }
        noc_async_write_barrier();
    }
}
