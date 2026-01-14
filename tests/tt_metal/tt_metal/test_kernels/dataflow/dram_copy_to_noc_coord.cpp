// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#if defined(COMPILE_FOR_ERISC)
#include "internal/ethernet/tunneling.h"
#endif

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
    bool bad_linked_transaction = static_cast<bool>(get_arg_val<uint32_t>(9));
    std::uint32_t l1_overflow_addr = get_arg_val<uint32_t>(10);
    std::uint32_t eth_src_overflow_addr = get_arg_val<uint32_t>(11);
    std::uint32_t eth_dest_overflow_addr = get_arg_val<uint32_t>(12);

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

    if (l1_overflow_addr) {
        experimental::CoreLocalMem<std::uint32_t> l1_overflow_buffer(l1_overflow_addr);
        l1_overflow_buffer[0] = 0xDEADBEEF;
    }

    if (eth_src_overflow_addr || eth_dest_overflow_addr) {
        // Destructive if not caught properly
#if defined(COMPILE_FOR_ERISC)
        internal_::eth_send_packet(0, eth_src_overflow_addr, eth_dest_overflow_addr, 4);
#endif
    }

    // NOC src address
    experimental::Noc noc;
    experimental::CoreLocalMem<std::uint32_t> local_buffer(local_buffer_addr);
    experimental::UnicastEndpoint src_unicast_endpoint;

    noc.async_read(
        src_unicast_endpoint,
        local_buffer,
        buffer_size,
        {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = buffer_src_addr},
        {});
    noc.async_read_barrier();

    // NOC dst address
    if (bad_linked_transaction) {
        experimental::MulticastEndpoint dst_mcast_endpoint;
        noc.async_write_multicast(
            local_buffer,
            dst_mcast_endpoint,
            buffer_size,
            1,
            {},
            {.noc_x_start = dst_noc_x,
             .noc_y_start = dst_noc_y,
             .noc_x_end = dst_noc_x,
             .noc_y_end = dst_noc_y,
             .addr = buffer_dst_addr},
            true);
        // linked transaction not closed, the next unicast will hang.
    }

    experimental::UnicastEndpoint dst_unicast_endpoint;
    if (use_inline_dw_write) {
        // Just write something to trigger the watcher assertion. Result data doesn't matter.
        noc.inline_dw_write(
            dst_unicast_endpoint, local_buffer[0], {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = buffer_dst_addr});
    } else {
        noc.async_write(
            local_buffer,
            dst_unicast_endpoint,
            buffer_size,
            {},
            {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = buffer_dst_addr});
        noc.async_write_barrier();
    }
}
