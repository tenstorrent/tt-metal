// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t local_sem_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(2);
    uint32_t src_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t dst_dram_addr = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;  // index=0
    uint32_t tile_size_bytes = get_tile_size(cb_id_in0) * num_tiles;
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

    uint64_t src_dram_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_dram_addr);
    uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_dram_addr);

    volatile tt_l1_ptr uint32_t* local_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_sem_addr);
    noc_semaphore_wait(local_sem, 1);
    uint64_t noc_local_sem_addr = get_noc_addr(local_sem_addr);
    noc_semaphore_inc(noc_local_sem_addr, -1);
    noc_async_atomic_barrier();
    noc_async_read(src_dram_noc_addr, l1_write_addr, tile_size_bytes);
    noc_async_read_barrier();
    noc_async_write(l1_write_addr, dst_dram_noc_addr, tile_size_bytes);
    noc_async_write_barrier();
}
