// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t pages = get_compile_time_arg_val(0);
    constexpr uint32_t depth = get_compile_time_arg_val(1);
    constexpr bool copy_payload = get_compile_time_arg_val(2);
    constexpr uint32_t bytes = 1088;
    constexpr auto copy_args = TensorAccessorArgs<3>();
    constexpr auto receipt_args = TensorAccessorArgs<copy_args.next_compile_time_args_offset()>();
    const auto copied = TensorAccessor(copy_args, get_arg_val<uint32_t>(0), bytes);
    const auto receipt = TensorAccessor(receipt_args, get_arg_val<uint32_t>(1), 32);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t worker = get_arg_val<uint32_t>(5);
    const uint32_t blocks = (count + pages - 1) / pages;
    uint32_t sum_first = 0, sum_last = 0, weighted = 0;
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t slot = block % depth;
        const uint32_t offset = block * pages;
        const uint32_t valid = count - offset < pages ? count - offset : pages;
        cb_wait_front(slot, pages);
        const uint32_t src = get_read_ptr(slot);
        auto* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        const uint32_t left = data[0];
        const uint32_t right = data[valid * bytes / 4 - 1];
        sum_first += left;
        sum_last += right;
        weighted += (block + 1) * (left ^ right);
        if constexpr (copy_payload) {
            for (uint32_t page = 0; page < valid; ++page) {
                noc_async_write_page(first + (offset + page) * stride, copied, src + page * bytes);
            }
            noc_async_write_barrier();
        }
        // The slot cannot be reused before both inspection and any copy finish.
        cb_pop_front(slot, pages);
    }
    // Writer-private scratch; no other processor owns CB16 or its contents.
    const uint32_t out = get_write_ptr(16);
    auto* record = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    record[0] = count;
    record[1] = blocks;
    record[2] = sum_first;
    record[3] = sum_last;
    record[4] = weighted;
    record[5] = first;
    record[6] = stride;
    record[7] = 0xB17ECAFE;
    noc_async_write_page(worker, receipt, out);
    noc_async_write_barrier();
}
