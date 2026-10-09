// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row scatter of the DeepSeek-V4.1 decode pool (tt/v41/pool.py scatter_rows): for i in [row_start, row_start +
// row_count), copy row src_base + i of ``src`` to row dst_base + i * stride + idx[i] of ``dst`` unless idx[i] is
// SENTINEL. Both tensors are row-major, DRAM-interleaved, one page per row of the same byte width; ``idx`` is one
// uint32 row-major page. The destination row comes from a device tensor, so a captured trace writes wherever the
// step's inputs say (the ring slot, a user's compressed row), while the per-tensor bases are the capture's
// constants. Data movement only.
//
// compile_time_args = [cb_scratch, ROW_BYTES, TensorAccessorArgs(src)..., TensorAccessorArgs(dst)...,
//                      TensorAccessorArgs(idx)...]
// runtime args      = [src_addr, dst_addr, idx_addr, src_base, dst_base, stride, row_start, row_count, rows]
// (``rows``: the length of idx; the whole page is read, a DRAM read must start at an aligned address)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t src_base = get_arg_val<uint32_t>(3);
    const uint32_t dst_base = get_arg_val<uint32_t>(4);
    const uint32_t stride = get_arg_val<uint32_t>(5);
    const uint32_t row_start = get_arg_val<uint32_t>(6);
    const uint32_t row_count = get_arg_val<uint32_t>(7);
    const uint32_t rows = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    constexpr uint32_t ROW_L1 = (ROW_BYTES + 63) & ~63u;

    if (row_count == 0) {
        return;
    }
    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    DataflowBuffer scratch(cb_scratch);
    const uint32_t base_l1 = (scratch.get_write_ptr() + 63) & ~63u;
    const uint32_t idx_l1 = base_l1;  // all the indices, then two row buffers
    const uint32_t idx_bytes = (rows * 4 + 63) & ~63u;
    const uint32_t row_l1[2] = {base_l1 + idx_bytes, base_l1 + idx_bytes + ROW_L1};
    volatile tt_l1_ptr uint32_t* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    noc_async_read(idx.get_noc_addr(0, 0), idx_l1, rows * 4);
    noc_async_read_barrier();
    uint32_t buf = 0;
    for (uint32_t n = 0; n < row_count; ++n) {
        const uint32_t target = indices[row_start + n];
        if (target == SENTINEL) {
            continue;
        }
        const uint32_t i = row_start + n;
        noc_async_read(src.get_noc_addr(src_base + i, 0), row_l1[buf], ROW_BYTES);
        noc_async_read_barrier();
        noc_async_write(row_l1[buf], dst.get_noc_addr(dst_base + i * stride + target, 0), ROW_BYTES);
        buf ^= 1;
        if (buf == 0) {
            noc_async_write_barrier();  // both buffers' writes have left L1 before they are reused
        }
    }
    noc_async_write_barrier();
}
