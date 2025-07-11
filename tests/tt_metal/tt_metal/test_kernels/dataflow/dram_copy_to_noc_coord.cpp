// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t local_buffer_addr = get_arg_val<uint32_t>(0);

    std::uint32_t buffer_src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t src_noc_x = get_arg_val<uint32_t>(2);
    std::uint32_t src_noc_y = get_arg_val<uint32_t>(3);

    std::uint32_t buffer_dst_addr = get_arg_val<uint32_t>(4);
    std::uint32_t dst_noc_x = get_arg_val<uint32_t>(5);
    std::uint32_t dst_noc_y = get_arg_val<uint32_t>(6);

    std::uint32_t buffer_size = get_arg_val<uint32_t>(7);

    bool use_inline_dw_write = static_cast<bool>(get_arg_val<uint32_t>(8));

#if defined(SIGNAL_COMPLETION_TO_DISPATCHER)
    // We will assert later. This kernel will hang.
    // Need to signal completion to dispatcher before hanging so that
    // Dispatcher Kernel is able to finish.
    // Device Close () requires fast dispatch kernels to finish.
#if defined(COMPILE_FOR_ERISC)
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);
#else
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
#endif
    uint64_t dispatch_addr = NOC_XY_ADDR(
        NOC_X(mailboxes->go_messages[mailboxes->go_message_index].master_x),
        NOC_Y(mailboxes->go_messages[mailboxes->go_message_index].master_y),
        DISPATCH_MESSAGE_ADDR +
            NOC_STREAM_REG_SPACE_SIZE * mailboxes->go_messages[mailboxes->go_message_index].dispatch_message_offset);
    noc_fast_write_dw_inline<DM_DEDICATED_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        1 << REMOTE_DEST_BUF_WORDS_FREE_INC,
        dispatch_addr,
        0xF,  // byte-enable
        NOC_UNICAST_WRITE_VC,
        false,  // mcast
        true    // posted
    );
#endif

    // NOC src address
    std::uint64_t buffer_src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, buffer_src_addr);
    noc_async_read(buffer_src_noc_addr, local_buffer_addr, buffer_size);
    noc_async_read_barrier();

    // NOC dst address
    std::uint64_t buffer_dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, buffer_dst_addr);
    if (use_inline_dw_write) {
        auto src_data = reinterpret_cast<volatile uint32_t*>(local_buffer_addr);
        // Just write something to trigger the watcher assertion. Result data doesn't matter.
        noc_inline_dw_write(buffer_dst_noc_addr, src_data[0] /*val*/);
    } else {
        noc_async_write(local_buffer_addr, buffer_dst_noc_addr, buffer_size);
        noc_async_write_barrier();
    }
}
