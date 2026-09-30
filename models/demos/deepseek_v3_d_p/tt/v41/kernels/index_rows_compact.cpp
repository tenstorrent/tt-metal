// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// In-place compaction of the leading index rows of V4.1 sparse attention (tt/v41/attention.py compact_leading_rows):
// per row i of rows [row_start, row_start + row_count) of a uint32 row-major [R, K] tensor (DRAM-interleaved, one
// page per row), with m = the number of SENTINEL entries among its first W (the window's missing rows),
//   row[j] = row[j + m]  for j < K - m,   row[j] = SENTINEL  for j >= K - m
// i.e. the row shifted left by m with a sentinel fill. Rows with m == 0 are not written. Data movement only.
//
// compile_time_args = [cb_scratch, K, W, TensorAccessorArgs(rows)...]
// runtime args      = [rows_addr, row_start, row_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t rows_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t row_count = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr auto rows_args = TensorAccessorArgs<3>();
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    static_assert(W <= K, "the window part lies inside the row");

    const auto rows = TensorAccessor(rows_args, rows_addr);
    DataflowBuffer scratch(cb_scratch);
    const uint32_t row_l1 = (scratch.get_write_ptr() + 63) & ~63u;  // one row (K * 4 bytes), 64-byte aligned
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);

    for (uint32_t i = row_start; i < row_start + row_count; ++i) {
        const uint64_t row_noc = rows.get_noc_addr(i, 0);
        noc_async_write_barrier();  // the previous row has left L1
        noc_async_read(row_noc, row_l1, K * 4);
        noc_async_read_barrier();
        uint32_t m = 0;
        for (uint32_t j = 0; j < W; ++j) {
            m += row[j] == SENTINEL;
        }
        if (m == 0) {
            continue;
        }
        for (uint32_t j = 0; j + m < K; ++j) {  // forward copy: reads stay ahead of writes
            row[j] = row[j + m];
        }
        for (uint32_t j = K - m; j < K; ++j) {
            row[j] = SENTINEL;
        }
        noc_async_write(row_l1, row_noc, K * 4);
    }
    noc_async_write_barrier();
}
