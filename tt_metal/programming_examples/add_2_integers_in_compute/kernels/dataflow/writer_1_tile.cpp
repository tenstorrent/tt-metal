// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_addr);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

    cb_wait_front(cb_id_out0, 1);
    noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, 1);
}
