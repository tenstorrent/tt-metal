// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "internal/firmware_common.h"
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
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
    DPRINT << HEX() << buffer_src_addr << ENDL();
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
    bool use_multicast_semaphore_inc = static_cast<bool>(get_arg_val<uint32_t>(13));
    std::uint32_t mcast_dst_end_x = get_arg_val<uint32_t>(14);
    std::uint32_t mcast_dst_end_y = get_arg_val<uint32_t>(15);

    // We will assert later. This kernel will hang.
    // Need to signal completion to dispatcher before hanging so that
    // Dispatcher Kernel is able to finish.
    // Device Close () requires fast dispatch kernels to finish.
    volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
#if defined(COMPILE_FOR_IDLE_ERISC) or defined(COMPILE_FOR_DM)
    go_message_in->signal = RUN_MSG_DONE;
#else
    uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
    notify_dispatch_core_done(dispatch_addr, noc_index);
#endif

    if (l1_overflow_addr) {
        experimental::CoreLocalMem<std::uint32_t> l1_overflow_buffer(l1_overflow_addr);
        l1_overflow_buffer[0] = 0xDEADBEEF;
    }

    if (eth_src_overflow_addr || eth_dest_overflow_addr) {
        // Destructive if not caught properly
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
        internal_::eth_send_packet(0, eth_src_overflow_addr, eth_dest_overflow_addr, 4);
#endif
    }

    if (use_multicast_semaphore_inc) {
        // Use invalid multicast range to trigger the watcher assertion
        // dst_noc_x/y is start, mcast_dst_end_x/y is end
        uint64_t dst_multicast_noc_addr =
            get_noc_multicast_addr(dst_noc_x, dst_noc_y, mcast_dst_end_x, mcast_dst_end_y, buffer_dst_addr);
        noc_semaphore_inc_multicast(dst_multicast_noc_addr, 1, 1);
        noc_async_atomic_barrier();
        return;  // Don't do normal operations
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
