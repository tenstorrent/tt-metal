// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// row_gather data-movement kernel (model-local generic op: custom_ops/row_gather/op.py). out[r] = src[index[r]] for
// this core's contiguous range of output rows: the core reads its slice of the index vector (one NoC read), then
// streams the rows through a ring of L1 row buffers (ROWS_IN_FLIGHT reads issued, a read barrier, the writes issued,
// the ring reused once its writes have drained). An index at or beyond the source row count writes a zero row.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t rows_in_flight = get_compile_time_arg_val(1);
    constexpr uint32_t src_rows = get_compile_time_arg_val(2);
    constexpr uint32_t index_bytes_per_entry = 4u;
    constexpr auto src_a = TensorAccessorArgs<3>();
    constexpr auto idx_a = TensorAccessorArgs<src_a.next_compile_time_args_offset()>();
    constexpr auto out_a = TensorAccessorArgs<idx_a.next_compile_time_args_offset()>();

    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t n_rows = get_arg_val<uint32_t>(1);
    const uint32_t src_addr = get_arg_val<uint32_t>(2);
    const uint32_t idx_addr = get_arg_val<uint32_t>(3);
    const uint32_t out_addr = get_arg_val<uint32_t>(4);
    const uint32_t idx_page_bytes = get_arg_val<uint32_t>(5);
    if (n_rows == 0) {
        return;
    }

    const auto src_acc = TensorAccessor(src_a, src_addr, row_bytes);
    const auto idx_acc = TensorAccessor(idx_a, idx_addr, idx_page_bytes);
    const auto out_acc = TensorAccessor(out_a, out_addr, row_bytes);

    auto align64 = [](uint32_t a) { return (a + 63u) & ~63u; };
    const uint32_t ring_base = align64(CircularBuffer(0).get_write_ptr());
    const uint32_t idx_base = align64(CircularBuffer(1).get_write_ptr());
    const uint32_t zero_base = align64(CircularBuffer(2).get_write_ptr());

    Noc noc;
    // the core's index slice: r0 is a multiple of 16 entries, so the 64 B offset is aligned; the slice is read whole
    noc.async_read(
        idx_acc,
        CoreLocalMem<uint32_t>(idx_base),
        align64(n_rows * index_bytes_per_entry),
        {.page_id = 0, .offset_bytes = r0 * index_bytes_per_entry},
        {});
    // a zero row for out-of-range indices
    {
        CoreLocalMem<volatile uint32_t> z(zero_base);
        for (uint32_t w = 0; w < row_bytes / 4u; w++) {
            z[w] = 0u;
        }
    }
    noc.async_read_barrier();
    const CoreLocalMem<volatile uint32_t> idx(idx_base);

    for (uint32_t base = 0; base < n_rows; base += rows_in_flight) {
        const uint32_t n = (n_rows - base < rows_in_flight) ? (n_rows - base) : rows_in_flight;
        // the ring's previous writes must have drained before its buffers are refilled
        noc.async_write_barrier();
        for (uint32_t j = 0; j < n; j++) {
            const uint32_t s = idx[base + j];
            if (s < src_rows) {
                noc.async_read(
                    src_acc, CoreLocalMem<uint32_t>(ring_base + j * row_bytes), row_bytes, {.page_id = s}, {});
            }
        }
        noc.async_read_barrier();
        for (uint32_t j = 0; j < n; j++) {
            const uint32_t s = idx[base + j];
            const uint32_t from = (s < src_rows) ? (ring_base + j * row_bytes) : zero_base;
            noc.async_write(out_acc, CoreLocalMem<uint32_t>(from), row_bytes, {.page_id = r0 + base + j}, {});
        }
    }
    noc.async_write_barrier();
}
