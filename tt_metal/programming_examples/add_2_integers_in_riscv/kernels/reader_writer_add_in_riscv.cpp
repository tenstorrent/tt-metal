// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"

void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram = get_arg_val<uint32_t>(2);
    uint32_t src0_l1 = get_arg_val<uint32_t>(3);
    uint32_t src1_l1 = get_arg_val<uint32_t>(4);
    uint32_t dst_l1 = get_arg_val<uint32_t>(5);

    // Create address generators for the input buffers. Consider these the
    // pointers for interleaved buffers. The parameters here must match the
    // parameters used in the host code when creating the buffers.
    InterleavedAddrGen<true> src0 = {.bank_base_address = src0_dram, .page_size = sizeof(uint32_t)};
    InterleavedAddrGen<true> src1 = {.bank_base_address = src1_dram, .page_size = sizeof(uint32_t)};
    InterleavedAddrGen<true> dst = {.bank_base_address = dst_dram, .page_size = sizeof(uint32_t)};

    // Read data from DRAM -> local L1
    // Note that noc_async_read() is a non-blocking operation. It will return immediately and the data will be
    // might not be available immediately. Thus we need to use noc_async_read_barrier() to wait until the data
    // is available before we can use it.
    // To perform NoC transfers, we need to calculate the address via get_noc_addr() function.
    // The first argument is the page index, which is 0 in this case as we are reading the first page of the buffer.
    // The second argument is the address generator for the buffer.
    // "Page" simply means a unit of data.
    uint64_t src0_dram_noc_addr = src0.get_noc_addr(0);
    uint64_t src1_dram_noc_addr = src1.get_noc_addr(0);
    noc_async_read(src0_dram_noc_addr, src0_l1, sizeof(uint32_t));
    noc_async_read(src1_dram_noc_addr, src1_l1, sizeof(uint32_t));
    noc_async_read_barrier();

    // Do simple add in RiscV core
    uint32_t* dat0 = (uint32_t*)src0_l1;
    uint32_t* dat1 = (uint32_t*)src1_l1;
    uint32_t* out0 = (uint32_t*)dst_l1;

    DPRINT("Adding integers: {} + {}\n", *dat0, *dat1);

    (*out0) = (*dat0) + (*dat1);

    // Force the store above to land in L1 before the noc_async_write below reads dst_l1 as its source.
    // A baby-RISCV store can retire before it lands, and the RISCV and NoC are independent L1 clients with
    // no ordering between them (MemoryOrdering.md), so the NoC could otherwise read a stale value.
    // load_blocking stalls the pipeline until the store is processed.
    (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1));

    // Write data from L1 -> DRAM. Again this is a non-blocking operation.
    // Thus we need to use noc_async_write_barrier() to wait until the data is
    // is written to DRAM before we can continue.
    uint64_t dst_dram_noc_addr = dst.get_noc_addr(0);
    noc_async_write(dst_l1, dst_dram_noc_addr, sizeof(uint32_t));
    noc_async_write_barrier();
}
