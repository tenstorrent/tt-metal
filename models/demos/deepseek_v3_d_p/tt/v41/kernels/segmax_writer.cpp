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
// With PIN_ROWS > 0 the pin tensor holds one entry per row instead (PIN_ROWS uint32, read whole): q = pin[i], and a
// SENTINEL entry pins nothing (the decode tile: entry u is user u's newest row, or the sentinel when it is idle).
//
// Bound (runtime arg 7, TT_V41_PREFILL_INDEX_BOUND): with 1 the
// outputs past the pinned one (segments starting after the row's newest element, i.e. rows the query may not see)
// are written -inf, so an unmasked score (the kernel's output without the additive visibility mask) selects as the
// masked one did. 0: as before (the input carries the mask, or the pin is a column within a gathered superblock).
//
// compile_time_args = [cb_out, cb_pin, SEG, BLOCK, OUT_WIDTH, VALID_OUT, PIN_ROWS, TensorAccessorArgs(out)...,
//                      TensorAccessorArgs(pin)...]
// runtime args = [dst_addr, row_start, row_count, chunks_per_row, pin_addr, pin_base, pin_mask, bound]

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
    const uint32_t bound = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pin = get_compile_time_arg_val(1);
    constexpr uint32_t SEG = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_WIDTH = get_compile_time_arg_val(4);
    constexpr uint32_t VALID_OUT = get_compile_time_arg_val(5);
    constexpr uint32_t PIN_ROWS = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr auto pin_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    constexpr uint32_t PIN_BYTES = PIN_ROWS == 0 ? 32 : PIN_ROWS * 4;
    constexpr uint32_t PIN_SLOT = (PIN_BYTES + 63) & ~63u;

    constexpr uint32_t PER_TILE = SEG == 8 ? 4 : 1;
    constexpr uint32_t RESULTS = BLOCK * PER_TILE;
    constexpr uint32_t OUT_PER_CHUNK = BLOCK * 32 * PER_TILE;
    constexpr uint32_t TILE_U16 = 32 * 32;
    constexpr uint16_t POS_INF = 0x7F80;
    constexpr uint16_t NEG_INF = 0xFF80;
    constexpr uint32_t NO_PIN = 0xFFFFFFFFu;

    const auto out = TensorAccessor(out_args, dst_addr);
    DataflowBuffer res(cb_out);
    // [0, PIN_SLOT): the pin page (64-byte aligned: a DRAM read); then two output chunks
    DataflowBuffer scratch(cb_pin);

    const uint32_t base_l1 = (scratch.get_write_ptr() + 63) & ~63u;
    const auto pin = TensorAccessor(pin_args, pin_addr);
    noc_async_read(pin.get_noc_addr(0, 0), base_l1, PIN_BYTES);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* pins = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_l1);
    const uint32_t query_first = pins[0];
    const uint32_t buf_l1[2] = {base_l1 + PIN_SLOT, base_l1 + PIN_SLOT + OUT_PER_CHUNK * 2};

    uint32_t slot = 0;
    for (uint32_t row = row_start; row < row_start + row_count; ++row) {
        const uint32_t q = PIN_ROWS == 0 ? query_first + row : pins[row];
        const uint32_t e = (pin_base + q) & pin_mask;
        const uint32_t pin_col = (q != SENTINEL || PIN_ROWS == 0) && e / SEG < VALID_OUT ? e / SEG : NO_PIN;
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
            if (bound != 0 && pin_col != NO_PIN && pin_col + 1 < first + OUT_PER_CHUNK) {
                // the segments after the pinned one hold rows past the query: not visible
                for (uint32_t o = pin_col + 1 > first ? pin_col + 1 : first; o < first + OUT_PER_CHUNK; ++o) {
                    dst[o - first] = NEG_INF;
                }
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
