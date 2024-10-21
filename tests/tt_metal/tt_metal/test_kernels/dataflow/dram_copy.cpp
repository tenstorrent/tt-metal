// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or other RISCs
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t dram_src_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(4);
    std::uint32_t dram_dst_noc_x        = get_arg_val<uint32_t>(5);
    std::uint32_t dram_dst_noc_y        = get_arg_val<uint32_t>(6);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);

#if defined(SIGNAL_COMPLETION_TO_DISPATCHER)
        //We will assert later. This kernel will hang.
        //Need to signal completion to dispatcher before hanging so that
        //Dispatcher Kernel is able to finish.
        //Device Close () requires fast dispatch kernels to finish.
#if defined(COMPILE_FOR_ERISC)
        tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);
#else
        tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
#endif
        uint64_t dispatch_addr = NOC_XY_ADDR(mailboxes->go_message.master_x, mailboxes->go_message.master_y, DISPATCH_MESSAGE_ADDR);
        noc_fast_atomic_increment(noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, NOC_UNICAST_WRITE_VC, 1, 31, false);
#endif

    // DRAM NOC src address
    std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);
    noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, dram_buffer_size);
    noc_async_read_barrier();

    // DRAM NOC dst address
    std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr);
    noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, dram_buffer_size);
    noc_async_write_barrier();
}
