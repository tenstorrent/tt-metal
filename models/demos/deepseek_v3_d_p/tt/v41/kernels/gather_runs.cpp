// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-local gather of fixed-length runs (DeepSeek-V4.1 candidate selection, tt/v41/indexer_kernels.py gather_runs):
//   out[i, r * RUN : (r + 1) * RUN] = src[i, ids[i, r] * RUN : (ids[i, r] + 1) * RUN]   (-inf if ids[i, r] >= nruns)
// src bf16 row-major [R, W] and out bf16 row-major [R, K * RUN] DRAM-interleaved (one page per row), ids uint32
// row-major [R, K] (one page per row; the 0xFFFFFFFF sentinel reads as -inf). RUN_BYTES is 16 (blocks of 8) or 64
// (superblocks of 32). Data movement only: runs on both RISC-Vs of a core (each its own row range and scratch CB).
//
// Per row the source row is streamed through L1 in SEG_BYTES segments (sequential DRAM reads: 2048 random 64-byte
// reads per row measured 2.7 ms on [640, 56320] rows, about 13x a sequential pass over them), and each segment's
// runs are copied into the output row, which is written once. The runs are bucketed by segment first (a counting
// sort of r by segment), so every run is visited once.
//
// compile_time_args = [cb_scratch, RUN_BYTES, K, SEG_BYTES, TensorAccessorArgs(src)..., TensorAccessorArgs(ids)...,
//                      TensorAccessorArgs(out)...]
// runtime args      = [src_addr, ids_addr, out_addr, row_start, row_count, nruns, row_bytes]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t row_count = get_arg_val<uint32_t>(4);
    const uint32_t nruns = get_arg_val<uint32_t>(5);
    const uint32_t row_bytes = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t RUN_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t SEG_BYTES = get_compile_time_arg_val(3);  // a multiple of 64
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto ids_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr uint32_t RUN_U32 = RUN_BYTES / 4;
    constexpr uint32_t MAX_SEGS = 64;
    constexpr uint32_t NEGINF_PAIR = 0xFF80FF80u;

    const auto src = TensorAccessor(src_args, src_addr);
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // scratch (64-byte aligned): segment (SEG_BYTES) | output row (K * RUN_BYTES) | ids row (K * 4) | order (K * 4)
    DataflowBuffer scratch(cb_scratch);
    const uint32_t seg_l1 = (scratch.get_write_ptr() + 63) & ~63u;
    const uint32_t out_l1 = seg_l1 + SEG_BYTES;
    const uint32_t ids_l1 = out_l1 + K * RUN_BYTES;
    volatile tt_l1_ptr uint32_t* id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
    volatile tt_l1_ptr uint32_t* order = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1 + K * 4);
    volatile tt_l1_ptr uint32_t* seg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seg_l1);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);

    const uint32_t segs = (row_bytes + SEG_BYTES - 1) / SEG_BYTES;  // <= MAX_SEGS (checked by the host)
    uint32_t bucket[MAX_SEGS + 1];
    for (uint32_t row = row_start; row < row_start + row_count; ++row) {
        noc_async_read(ids.get_noc_addr(row, 0), ids_l1, K * 4);
        const uint64_t row_noc = src.get_noc_addr(row, 0);
        noc_async_write_barrier();  // the previous row's output has left L1
        noc_async_read_barrier();

        // bucket the runs by segment; runs past the row read -inf
        for (uint32_t s = 0; s <= segs; ++s) {
            bucket[s] = 0;
        }
        for (uint32_t r = 0; r < K; ++r) {
            const uint32_t c = id[r];
            if (c < nruns) {
                bucket[(c * RUN_BYTES) / SEG_BYTES + 1]++;
            } else {
                for (uint32_t w = 0; w < RUN_U32; ++w) {
                    dst[r * RUN_U32 + w] = NEGINF_PAIR;
                }
            }
        }
        for (uint32_t s = 0; s < segs; ++s) {
            bucket[s + 1] += bucket[s];
        }
        for (uint32_t r = 0; r < K; ++r) {
            const uint32_t c = id[r];
            if (c < nruns) {
                order[bucket[(c * RUN_BYTES) / SEG_BYTES]++] = r;
            }
        }
        // bucket[s] is now the end of segment s's runs in order[] (the start of segment s + 1's)

        uint32_t next = 0;
        for (uint32_t s = 0; s < segs; ++s) {
            const uint32_t end = bucket[s];
            if (next == end) {
                continue;
            }
            const uint32_t first = s * SEG_BYTES;
            const uint32_t bytes = row_bytes - first < SEG_BYTES ? row_bytes - first : SEG_BYTES;
            noc_async_read(row_noc + first, seg_l1, bytes);
            noc_async_read_barrier();
            for (; next < end; ++next) {
                const uint32_t r = order[next];
                const uint32_t from = (id[r] * RUN_BYTES - first) / 4;
                for (uint32_t w = 0; w < RUN_U32; ++w) {
                    dst[r * RUN_U32 + w] = seg[from + w];
                }
            }
        }
        noc_async_write(out_l1, out.get_noc_addr(row, 0), K * RUN_BYTES);
    }
    noc_async_write_barrier();
}
