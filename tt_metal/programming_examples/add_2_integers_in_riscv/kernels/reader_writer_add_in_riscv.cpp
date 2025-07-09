// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram = get_arg_val<uint32_t>(2);
    uint32_t src0_l1 = get_arg_val<uint32_t>(3);
    uint32_t src1_l1 = get_arg_val<uint32_t>(4);
    uint32_t dst_l1 = get_arg_val<uint32_t>(5);

    // NoC coords (x,y) depending on DRAM location on-chip
    uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<true>(/*bank_id=*/0, src0_dram);
    uint64_t src1_dram_noc_addr = get_noc_addr_from_bank_id<true>(/*bank_id=*/0, src1_dram);
    uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(/*bank_id=*/0, dst_dram);

    // Read data from DRAM -> L1 circular buffers
    noc_async_read(src0_dram_noc_addr, src0_l1, sizeof(uint32_t));
    noc_async_read_barrier();
    noc_async_read(src1_dram_noc_addr, src1_l1, sizeof(uint32_t));
    noc_async_read_barrier();

    // Do simple add in RiscV core
    uint32_t* dat0 = (uint32_t*)src0_l1;
    uint32_t* dat1 = (uint32_t*)src1_l1;
    uint32_t* out0 = (uint32_t*)dst_l1;

    DPRINT << "Adding integers: " << src0_l1 << " + " << src1_l1 << "\n";

    (*out0) = (*dat0) + (*dat1);

    // Write data from L1 circulr buffer (in0) -> DRAM
    noc_async_write(dst_l1, dst_dram, sizeof(uint32_t));
    noc_async_write_barrier();
}
