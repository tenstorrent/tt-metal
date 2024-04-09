// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);

    uint64_t dst_noc_addr = get_noc_addr(1, 0, dst_addr);

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

    cb_wait_front(cb_id_out0, 1);
    noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, 1);
}
