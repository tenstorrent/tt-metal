// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// row_gather reader (model-local generic op: custom_ops/row_gather/op.py). Output rows [row0, row0 + nrows) of this
// core: out[i] = table[idx[i]]. The index list is ONE row-major page; it is read in 64 B aligned chunks of idx_chunk
// ids. Each batch of `batch` table rows (one page each) is issued as `batch` NOC reads with a single barrier, so
// `batch` reads are in flight per core (ttnn.embedding's row-major reader waits on every row), and pushed to the CB;
// the writer drains a batch while the next one is read (the CB holds two batches).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t table_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t row0 = get_arg_val<uint32_t>(2);
    const uint32_t nrows = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t idx_chunk = get_compile_time_arg_val(4);  // ids per index read (a multiple of batch and of 16)
    constexpr uint32_t idx_page_bytes = get_compile_time_arg_val(5);
    constexpr auto table_args = TensorAccessorArgs<6>();
    constexpr auto idx_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();

    const auto s_table = TensorAccessor(table_args, table_addr, row_bytes);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, idx_page_bytes);

    Noc noc;
    CircularBuffer cbr(cb_rows), cbi(cb_idx);
    cbi.reserve_back(1);  // private index scratch: reserved once, never pushed
    const uint32_t idx_base = get_write_ptr(cb_idx);
    const CoreLocalMem<volatile uint32_t> ids(idx_base);

    for (uint32_t done = 0; done < nrows; done += idx_chunk) {
        const uint32_t n_chunk = (nrows - done) < idx_chunk ? (nrows - done) : idx_chunk;
        const uint32_t off = (row0 + done) * 4u;  // 64 B aligned: row0 and idx_chunk are multiples of 16
        uint32_t nbytes = (n_chunk * 4u + 63u) & ~63u;
        if (off + nbytes > idx_page_bytes) {
            nbytes = idx_page_bytes - off;
        }
        noc.async_read(s_idx, CoreLocalMem<uint32_t>(idx_base), nbytes, {.page_id = 0, .offset_bytes = off}, {});
        noc.async_read_barrier();
        for (uint32_t b0 = 0; b0 < n_chunk; b0 += batch) {
            const uint32_t nb = (n_chunk - b0) < batch ? (n_chunk - b0) : batch;
            cbr.reserve_back(batch);
            for (uint32_t j = 0; j < nb; j++) {
                noc.async_read(s_table, cbr, row_bytes, {.page_id = ids[b0 + j]}, {.offset_bytes = j * row_bytes});
            }
            noc.async_read_barrier();
            cbr.push_back(batch);
        }
    }
}
