// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t src_buffer_size = get_arg_val<uint32_t>(2);

    uint32_t local_addr = get_arg_val<uint32_t>(3);

    uint32_t dst_addr = get_arg_val<uint32_t>(4);
    uint32_t dst_noc_x_start = get_arg_val<uint32_t>(5);
    uint32_t dst_noc_y_start = get_arg_val<uint32_t>(6);
    uint32_t dst_noc_x_end = get_arg_val<uint32_t>(7);
    uint32_t dst_noc_y_end = get_arg_val<uint32_t>(8);
    uint32_t num_dests = get_arg_val<uint32_t>(9);

    constexpr bool use_loopback = get_compile_time_arg_val(0);

#if defined(SIGNAL_COMPLETION_TO_DISPATCHER)
    // Signal completion to dispatcher before potentially hanging on sanitizer.
    // This allows Device Close() to complete even if kernel hangs.
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
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

    // Read src buffer into local L1 buffer
    constexpr auto bank_type = experimental::AllocatorBankType::DRAM;
    experimental::CoreLocalMem<std::uint32_t> local_buffer(local_addr);
    experimental::Noc noc;
    noc.async_read(
        experimental::AllocatorBank<bank_type>(),
        local_buffer,
        src_buffer_size,
        {.bank_id = bank_id, .addr = src_addr},
        {});
    noc.async_read_barrier();

    // multicast local L1 buffer to all destination cores
    experimental::MulticastEndpoint dst_mcast_endpoint;
    constexpr auto mcast_mode =
        use_loopback ? experimental::Noc::McastMode::INCLUDE_SRC : experimental::Noc::McastMode::EXCLUDE_SRC;
    noc.async_write_multicast<mcast_mode>(
        local_buffer,
        dst_mcast_endpoint,
        src_buffer_size,
        num_dests,
        {},
        {.noc_x_start = dst_noc_x_start,
         .noc_y_start = dst_noc_y_start,
         .noc_x_end = dst_noc_x_end,
         .noc_y_end = dst_noc_y_end,
         .addr = dst_addr});
    noc.async_write_barrier();
}
