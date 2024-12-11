// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram = get_arg_val<uint32_t>(2);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(3);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(4);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(5);

    // NoC coords (x,y) depending on DRAM location on-chip
    uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_dram);
    uint64_t src1_dram_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_dram);
    uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_dram);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;  // index=0
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;  // index=1

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

    // Read data from DRAM -> L1 circular buffers
    noc_async_read(src0_dram_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
    noc_async_read_barrier();
    noc_async_read(src1_dram_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
    noc_async_read_barrier();

    // Do simple add in RiscV core
    uint32_t* dat0 = (uint32_t*)l1_write_addr_in0;
    uint32_t* dat1 = (uint32_t*)l1_write_addr_in1;

    dat0[0] = dat0[0] + dat1[0];

    // Write data from L1 circulr buffer (in0) -> DRAM
    noc_async_write(l1_write_addr_in0, dst_dram_noc_addr, ublock_size_bytes_0);
    noc_async_write_barrier();
}
