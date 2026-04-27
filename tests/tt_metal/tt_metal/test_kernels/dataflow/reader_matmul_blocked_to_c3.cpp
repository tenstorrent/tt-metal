// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Variant of reader_matmul_blocked that pushes in0 into c_3 (instead of c_0)
// and in1 into c_1. Used by the TransposePreKBlock isolated test — the
// transpose functor drains c_3 into c_0, so the reader must bypass c_0.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_blocks = get_arg_val<uint32_t>(4);
    uint32_t in0_block_tile_cnt = get_arg_val<uint32_t>(5);
    uint32_t in1_block_tile_cnt = get_arg_val<uint32_t>(6);
    uint32_t in0_block_size_bytes = get_arg_val<uint32_t>(7);
    uint32_t in1_block_size_bytes = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_in0_xp = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    for (uint32_t i = 0; i < num_blocks; i++) {
        uint64_t src0_noc = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);
        uint64_t src1_noc = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);

        cb_reserve_back(cb_id_in0_xp, in0_block_tile_cnt);
        cb_reserve_back(cb_id_in1, in1_block_tile_cnt);

        uint32_t l1_w0 = get_write_ptr(cb_id_in0_xp);
        uint32_t l1_w1 = get_write_ptr(cb_id_in1);

        noc_async_read(src0_noc, l1_w0, in0_block_size_bytes);
        noc_async_read(src1_noc, l1_w1, in1_block_size_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0_xp, in0_block_tile_cnt);
        cb_push_back(cb_id_in1, in1_block_tile_cnt);

        src0_addr += in0_block_size_bytes;
        src1_addr += in1_block_size_bytes;
    }
}
