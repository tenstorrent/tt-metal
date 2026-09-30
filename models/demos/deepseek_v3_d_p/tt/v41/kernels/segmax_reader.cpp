// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the V4.1 row-major segment max (segmax_compute.cpp, tt/v41/indexer.py segment_max): fills the reduce
// scaler tile and, for SEG 8, the four column-mask tiles once, then streams this core's rows of the bf16 row-major
// input in chunks of BLOCK * 1024 contiguous elements (BLOCK tiles' worth of row-major data). A row's last chunk is
// completed with -inf past the row's W elements.
//
// compile_time_args = [cb_rm, cb_scaler, cb_mask, SEG, BLOCK, TensorAccessorArgs(input)...]
// runtime args      = [src_addr, row_start, row_count, row_bytes]
//
// Scaler tile: bf16 1.0 in row 0 of every face (MAX row reduce of all 32 columns). Mask tile b (SEG 8): 0 in columns
// [8b, 8b + 8), -inf elsewhere, every row (added before the row reduce, so the max covers block b only).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);
    const uint32_t row_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(2);
    constexpr uint32_t SEG = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    constexpr uint32_t FACE_U32 = 16 * 16 / 2;  // bf16 face, two values per word
    constexpr uint32_t ROW_U32 = 16 / 2;
    constexpr uint32_t ONE_PAIR = 0x3F803F80u;     // two bf16 1.0
    constexpr uint32_t NEGINF_PAIR = 0xFF80FF80u;  // two bf16 -inf
    constexpr uint32_t CHUNK_BYTES = BLOCK * 32 * 32 * 2;

    DataflowBuffer scaler(cb_scaler);
    scaler.reserve_back(1);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler.get_write_ptr());
    for (uint32_t w = 0; w < 4 * FACE_U32; ++w) {
        sc[w] = (w % FACE_U32) < ROW_U32 ? ONE_PAIR : 0u;
    }
    scaler.push_back(1);

    if constexpr (SEG == 8) {
        DataflowBuffer mask(cb_mask);
        mask.reserve_back(4);
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask.get_write_ptr());
        for (uint32_t b = 0; b < 4; ++b) {
            for (uint32_t face = 0; face < 4; ++face) {
                for (uint32_t w = 0; w < FACE_U32; ++w) {
                    // word w of the face holds columns 2 (w % 8), 2 (w % 8) + 1 of its face row; one 8-column
                    // block never splits a word
                    const uint32_t col = (face % 2) * 16 + 2 * (w % ROW_U32);
                    m[(b * 4 + face) * FACE_U32 + w] = col / 8 == b ? 0u : NEGINF_PAIR;
                }
            }
        }
        mask.push_back(4);
    }

    const auto src = TensorAccessor(src_args, src_addr);
    DataflowBuffer rm(cb_rm);
    const uint32_t chunks = (row_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES;
    for (uint32_t row = row_start; row < row_start + row_count; ++row) {
        const uint64_t base = src.get_noc_addr(row, 0);
        for (uint32_t c = 0; c < chunks; ++c) {
            rm.reserve_back(BLOCK);
            const uint32_t l1 = rm.get_write_ptr();
            const uint32_t offset = c * CHUNK_BYTES;
            const uint32_t bytes = row_bytes - offset < CHUNK_BYTES ? row_bytes - offset : CHUNK_BYTES;
            noc_async_read(base + offset, l1, bytes);
            if (bytes < CHUNK_BYTES) {
                // row_bytes is a multiple of 64 (W a multiple of 32), so the pad is whole words
                volatile tt_l1_ptr uint32_t* pad = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + bytes);
                for (uint32_t w = 0; w < (CHUNK_BYTES - bytes) / 4; ++w) {
                    pad[w] = NEGINF_PAIR;
                }
            }
            noc_async_read_barrier();
            rm.push_back(BLOCK);
        }
    }
}
