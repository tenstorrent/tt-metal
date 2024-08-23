// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t pad_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_src_stick_id = get_arg_val<uint32_t>(2);
    const uint32_t row_size_diff = get_arg_val<uint32_t>(3);
    const uint32_t dst_N = get_arg_val<uint32_t>(4);
    const uint32_t data_size_bytes = get_arg_val<uint32_t>(5);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(6);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool pad_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cb_id = tt::CB::c_in0;

    const InterleavedAddrGen<src_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = data_size_bytes
    };
    const InterleavedAddrGen<pad_is_dram> s1 = {
        .bank_base_address = pad_addr,
        .page_size = data_size_bytes
    };

    // pad based on page
    uint32_t src_stick_id = start_src_stick_id;
    uint32_t src_start_col_idx = row_size_diff / 2;
    uint32_t src_end_col_idx = dst_N - src_start_col_idx;
    for (uint32_t i = 0; i < num_rows_per_core; i++) {
        for (uint32_t dst_col_idx = 0; dst_col_idx < dst_N; dst_col_idx++) {
            cb_reserve_back(cb_id, 1);
            uint32_t l1_addr = get_write_ptr(cb_id);
            if (dst_col_idx < src_start_col_idx || dst_col_idx >= src_end_col_idx) {
                // add pad value to cb
                uint64_t pad_noc_addr = get_noc_addr(0, s1);
                noc_async_read(pad_noc_addr, l1_addr, data_size_bytes);
            }
            else {
                // add original src data to cb
                uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0);
                noc_async_read(src_noc_addr, l1_addr, data_size_bytes);
                src_stick_id++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id, 1);
            l1_addr += data_size_bytes;
        }
    }
}
