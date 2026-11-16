// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// PackRows (WS2, batch-1 decode): row 0 of every tile of two [1,1,1,W] bf16 TILE tensors a and b (na and
// nb tiles, interleaved) packed densely into a [1,1,32,32*n_out] bf16 TILE tensor, one output tile per core:
// output tile T row r <- row 0 of input tile i = 32*T + r (tiles of a first, then b); unused rows are zero.
// A tile row r lives in faces (r/16)*2 (columns 0-15) and +1 (columns 16-31) at byte offset (r%16)*32.
// compile-time args: [cb_out, na, nb] + TensorAccessorArgs(a, b, out)
// runtime args: [a_addr, b_addr, out_addr, T]
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t T = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t na = get_compile_time_arg_val(1);
    constexpr uint32_t nb = get_compile_time_arg_val(2);
    constexpr auto a_args = TensorAccessorArgs<3>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048, face_bytes = 512, row_bytes = 32;
    const auto s_a = TensorAccessor(a_args, a_addr, tile_bytes);
    const auto s_b = TensorAccessor(b_args, b_addr, tile_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t base = get_write_ptr(cb_out);
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    // rows past the last input tile are zero; the reads below write every other row
    const uint32_t first_empty = (na + nb > T * 32) ? (na + nb - T * 32 < 32 ? na + nb - T * 32 : 32) : 0;
    if (first_empty % 16 == 0) {
        // whole empty faces: zero them from the core's hardware zeros region (disjoint from the reads below)
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = (first_empty / 16) * 2 * face_bytes; off < tile_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
        }
    } else {
        for (uint32_t r = first_empty; r < 32; ++r) {
            const uint32_t word = (r >> 4) * 2 * (face_bytes / 4) + (r & 15) * (row_bytes / 4);
            for (uint32_t j = 0; j < row_bytes / 4; ++j) {
                words[word + j] = 0;
                words[word + face_bytes / 4 + j] = 0;
            }
        }
    }
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t i = T * 32 + r;
        if (i >= na + nb) {
            break;
        }
        const uint64_t src = i < na ? s_a.get_noc_addr(i) : s_b.get_noc_addr(i - na);
        const uint32_t dst = base + (r >> 4) * 2 * face_bytes + (r & 15) * row_bytes;
        noc_async_read(src, dst, row_bytes);
        noc_async_read(src + face_bytes, dst + face_bytes, row_bytes);
    }
    noc_async_read_barrier();
    noc_async_write(base, s_out.get_noc_addr(T), tile_bytes);
    noc_async_write_barrier();
}
