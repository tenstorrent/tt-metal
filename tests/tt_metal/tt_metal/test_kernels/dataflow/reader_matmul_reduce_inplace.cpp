// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the matmul_reduce_inplace isolated test.
// Loads `total_tiles` from DRAM into c_0, then one col-identity tile into c_1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t total_tiles = get_arg_val<uint32_t>(4);
    uint32_t src0_bytes = get_arg_val<uint32_t>(5);
    uint32_t src1_bytes = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    uint64_t src0_noc = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);
    uint64_t src1_noc = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);

    cb_reserve_back(cb_id_in0, total_tiles);
    uint32_t l1_w0 = get_write_ptr(cb_id_in0);
    noc_async_read(src0_noc, l1_w0, src0_bytes);

    cb_reserve_back(cb_id_in1, 1);
    uint32_t l1_w1 = get_write_ptr(cb_id_in1);
    noc_async_read(src1_noc, l1_w1, src1_bytes);

    noc_async_read_barrier();

    cb_push_back(cb_id_in0, total_tiles);
    cb_push_back(cb_id_in1, 1);
}
