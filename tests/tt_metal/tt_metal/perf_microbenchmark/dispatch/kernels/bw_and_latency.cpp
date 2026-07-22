// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CB id this instance operates on. Defaults to 0 (single-kernel read/write test). The concurrent
// read+write DRAM-contention mode (-rw) instantiates two kernels on the same core, one per NoC, and
// gives them distinct CBs so the reader's fill and the writer's drain use separate L1 regions.
#ifndef CB_ID
#define CB_ID 0
#endif

// DRAM_BANKED addressing knobs. BANK_BASE_ADDR = allocated buffer's base (0 = raw). TARGET_BANK >= 0 pins
// every access to one DRAM bank (page_id = TARGET_BANK + j*NUM_DRAM_BANKS); -1 = interleave across banks.
// Coords come from the device bank->NoC table (get_dram_noc_addr, per-NoC), so no manual NoC coord math.
#ifndef BANK_BASE_ADDR
#define BANK_BASE_ADDR 0
#endif
#ifndef TARGET_BANK
#define TARGET_BANK (-1)
#endif

void kernel_main() {
#if NOP_COUNT
    for (int i = 0; i < ITERATIONS; i++) {
#pragma GCC unroll 4096
        for (int j = 0; j < NOP_COUNT; j++) {
            asm("nop");
        }
    }
#else
#ifdef PAGE_SIZE
    uint32_t page_size = PAGE_SIZE;
#else
    uint32_t page_size = get_arg_val<uint32_t>(0);
#endif

    cb_reserve_back(CB_ID, PAGE_COUNT);
    uint32_t cb_addr = get_write_ptr(CB_ID);
    for (int i = 0; i < ITERATIONS; i++) {
        uint32_t read_ptr = cb_addr;
        uint32_t write_ptr = cb_addr;
        for (int j = 0; j < PAGE_COUNT; j++) {
#if DRAM_BANKED
#if TARGET_BANK >= 0
            const uint32_t page_id = TARGET_BANK + j * NUM_DRAM_BANKS;  // pin all accesses to one bank
#else
            const uint32_t page_id = j;  // interleave across all banks
#endif
            // get_dram_noc_addr resolves the bank's NoC coord from the device bank->NoC table for THIS
            // kernel's NoC (noc_index) automatically -- reader on NOC0, writer on NOC1, no manual flip.
            uint64_t noc_addr = get_dram_noc_addr(page_id, page_size, BANK_BASE_ADDR);
#else
            uint64_t noc_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), NOC_MEM_ADDR);
#endif

#if ISSUE_MCAST
            uint64_t dst_noc_multicast_addr =
                get_noc_multicast_addr(NOC_ADDR_X, NOC_ADDR_Y, MCAST_NOC_END_ADDR_X, MCAST_NOC_END_ADDR_Y, write_ptr);
            noc_async_write_multicast(read_ptr, dst_noc_multicast_addr, page_size, NUM_MCAST_DESTS, LINKED);
#elif WRITE
#if WRITE_DRAM
#if DRAM_BANKED
            noc_async_write(write_ptr, noc_addr, page_size);  // dest bank NoC coord from bank->NoC table
#else
            uint64_t noc_write_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), NOC_MEM_ADDR);
            noc_async_write(write_ptr, noc_write_addr, page_size);
#endif
#else
            uint64_t noc_write_addr = NOC_XY_ADDR(NOC_X(NOC_ADDR_X), NOC_Y(NOC_ADDR_Y), write_ptr);
            noc_async_write(NOC_MEM_ADDR, noc_write_addr, page_size);
#endif
#elif READ_ONE_PACKET
            noc_async_read_one_packet(noc_addr, read_ptr, page_size);
#else
            noc_async_read(noc_addr, read_ptr, page_size);
#endif

#if LATENCY
#if WRITE
#if LINKED
            noc_async_write_multicast(cb_addr, dst_noc_multicast_addr, page_size, NUM_MCAST_DESTS, false);
#endif
            noc_async_write_barrier();
#else
            noc_async_read_barrier();
#endif
#endif
            read_ptr += page_size;
            write_ptr += page_size;
        }
    }
#if !LATENCY
#if WRITE
#if LINKED
    uint64_t dst_noc_multicast_addr =
        get_noc_multicast_addr(NOC_ADDR_X, NOC_ADDR_Y, MCAST_NOC_END_ADDR_X, MCAST_NOC_END_ADDR_Y, cb_addr);
    noc_async_write_multicast(cb_addr, dst_noc_multicast_addr, page_size, NUM_MCAST_DESTS, false);
#endif
    noc_async_write_barrier();
#else
    noc_async_read_barrier();
#endif
#endif
#endif
}
