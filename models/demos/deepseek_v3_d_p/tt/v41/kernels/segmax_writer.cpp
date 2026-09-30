// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the V4.1 row-major segment max (segmax_compute.cpp): per chunk it collects column 0 of the compute's
// result tiles into the chunk's OUT_PER_CHUNK consecutive outputs, pins the row's newest element (optional) and
// writes them to the output row (row-major bf16, OUT_WIDTH per row, a multiple of 32).
//
// Result tile (j, b) (b < 4 for SEG 8, else b = 0) row r holds output BLOCK * r + j (SEG 32) or
// 4 (BLOCK * r + j) + b (SEG 8) of the chunk.
//
// Pin: row i of this chip (chunk query index q = query_first + i, query_first read from the per-chip pin tensor) has
// its newest element at column e = (pin_base + q) & pin_mask of the input; if e < W the output e / SEG becomes +inf.
//
// compile_time_args = [cb_out, cb_pin, SEG, BLOCK, OUT_WIDTH, VALID_OUT, TensorAccessorArgs(out)...,
//                      TensorAccessorArgs(pin)...]
// runtime args      = [dst_addr, row_start, row_count, chunks_per_row, pin_addr, pin_base, pin_mask]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);
    const uint32_t chunks = get_arg_val<uint32_t>(3);
    const uint32_t pin_addr = get_arg_val<uint32_t>(4);
    const uint32_t pin_base = get_arg_val<uint32_t>(5);
    const uint32_t pin_mask = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pin = get_compile_time_arg_val(1);
    constexpr uint32_t SEG = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_WIDTH = get_compile_time_arg_val(4);
    constexpr uint32_t VALID_OUT = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr auto pin_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t PER_TILE = SEG == 8 ? 4 : 1;
    constexpr uint32_t RESULTS = BLOCK * PER_TILE;
    constexpr uint32_t OUT_PER_CHUNK = BLOCK * 32 * PER_TILE;
    constexpr uint32_t TILE_U16 = 32 * 32;
    constexpr uint16_t POS_INF = 0x7F80;

    const auto out = TensorAccessor(out_args, dst_addr);
    DataflowBuffer res(cb_out);
    DataflowBuffer scratch(cb_pin);  // [0, 64): the pin page; [64, 64 + 2 * OUT_PER_CHUNK * 2): two output chunks

    const uint32_t base_l1 = scratch.get_write_ptr();
    const auto pin = TensorAccessor(pin_args, pin_addr);
    noc_async_read(pin.get_noc_addr(0, 0), base_l1, 32);
    noc_async_read_barrier();
    const uint32_t query_first = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_l1)[0];
    const uint32_t buf_l1[2] = {base_l1 + 64, base_l1 + 64 + OUT_PER_CHUNK * 2};

    uint32_t slot = 0;
    for (uint32_t row = row_start; row < row_start + row_count; ++row) {
        const uint32_t e = (pin_base + query_first + row) & pin_mask;
        const uint32_t pin_col = e / SEG < VALID_OUT ? e / SEG : 0xFFFFFFFFu;
        const uint64_t row_addr = out.get_noc_addr(row, 0);
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t first = c * OUT_PER_CHUNK;
            res.wait_front(RESULTS);
            // the buffer written two chunks ago must have left L1
            noc_async_writes_flushed();
            volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_l1[slot]);
            volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(res.get_read_ptr());
            for (uint32_t t = 0; t < RESULTS; ++t) {
                const uint32_t j = t / PER_TILE, b = t % PER_TILE;
                volatile tt_l1_ptr uint16_t* tile = src + t * TILE_U16;
                for (uint32_t r = 0; r < 32; ++r) {
                    // column 0: rows 0-15 in face 0, rows 16-31 in face 2 (16 x 16 faces, row-major inside)
                    const uint16_t v = tile[r < 16 ? r * 16 : 512 + (r - 16) * 16];
                    dst[(BLOCK * r + j) * PER_TILE + b] = v;
                }
            }
            res.pop_front(RESULTS);
            if (pin_col >= first && pin_col < first + OUT_PER_CHUNK) {
                dst[pin_col - first] = POS_INF;
            }
            if (first < OUT_WIDTH) {
                const uint32_t n = OUT_WIDTH - first < OUT_PER_CHUNK ? OUT_WIDTH - first : OUT_PER_CHUNK;
                noc_async_write(buf_l1[slot], row_addr + first * 2, n * 2);
            }
            slot ^= 1;
        }
    }
    noc_async_write_barrier();
}
