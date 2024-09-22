// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"


void kernel_main() {

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_dst_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t dst_N = get_arg_val<uint32_t>(2);
    const uint32_t data_size_bytes = get_arg_val<uint32_t>(3);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(4);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id = tt::CB::c_in0;

    const InterleavedAddrGen<dst_is_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = data_size_bytes
    };

    uint32_t dst_stick_id = start_dst_stick_id;
    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; row_idx++) {
        for (uint32_t dst_col_idx = 0; dst_col_idx < dst_N; dst_col_idx++) {
            cb_wait_front(cb_id, 1);
            uint32_t l1_addr = get_read_ptr(cb_id);
            uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, s0);
            noc_async_write(l1_addr, dst_noc_addr, data_size_bytes);
            noc_async_write_barrier();
            dst_stick_id++;
            cb_pop_front(cb_id, 1);
        }
    }

}
