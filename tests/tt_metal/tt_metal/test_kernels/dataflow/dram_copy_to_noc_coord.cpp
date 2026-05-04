// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "internal/firmware_common.h"
#include "api/compile_time_args.h"
#include "internal/hw_thread.h"
#include "experimental/kernel_args.h"
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
#include "internal/ethernet/tunneling.h"
#endif

#if defined(ARCH_QUASAR)
#define SANITIZE_GET_RTA(N) get_vararg(N)
#else
#define SANITIZE_GET_RTA(N) get_arg_val<uint32_t>(N)
#endif

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
#if defined(COMPILE_FOR_DM)
    uint32_t thread_idx = internal_::get_hw_thread_idx();

#if defined(TEST_MULTI_DM_SANITIZE_RACE)
    // Having explicit sync barrier helps stress test CAS in sanitize.h since
    // testing showed DM0 (which wakes other DMs) most likely wins without a barrier
    constexpr uint32_t num_dms = get_arg(args::num_dms);
    constexpr uint32_t multi_dm_base_addr = get_arg(args::multi_dm_base_addr);
    constexpr uint32_t multi_dm_base_size = get_arg(args::multi_dm_base_size);
    constexpr uint32_t l1_sync_addr = get_arg(args::l1_sync_addr);
    uint64_t* l1_ptr = reinterpret_cast<uint64_t*>(l1_sync_addr);
    __atomic_add_fetch(l1_ptr, 1, __ATOMIC_RELAXED);
    while (__atomic_load_n(l1_ptr, __ATOMIC_ACQUIRE) != num_dms) {
    }
#else
    // Single DM test: only specified dm_id executes, others exit early
    constexpr uint32_t dm_id = get_arg(args::dm_id);
    if (thread_idx != dm_id) {
        return;
    }
#endif
#endif
    std::uint32_t local_buffer_addr = SANITIZE_GET_RTA(0);

    std::uint32_t buffer_src_addr = SANITIZE_GET_RTA(1);
    std::uint32_t src_noc_x = SANITIZE_GET_RTA(2);
    std::uint32_t src_noc_y = SANITIZE_GET_RTA(3);

    std::uint32_t buffer_dst_addr = SANITIZE_GET_RTA(4);
    std::uint32_t dst_noc_x = SANITIZE_GET_RTA(5);
    std::uint32_t dst_noc_y = SANITIZE_GET_RTA(6);

    std::uint32_t buffer_size = SANITIZE_GET_RTA(7);

#if defined(COMPILE_FOR_DM) && defined(TEST_MULTI_DM_SANITIZE_RACE)
    buffer_dst_addr = (multi_dm_base_addr | static_cast<uint32_t>(thread_idx));
    buffer_size = (multi_dm_base_size | static_cast<uint32_t>(thread_idx));
#endif

    bool use_inline_dw_write = static_cast<bool>(SANITIZE_GET_RTA(8));
    bool bad_linked_transaction = static_cast<bool>(SANITIZE_GET_RTA(9));
    std::uint32_t l1_overflow_addr = SANITIZE_GET_RTA(10);
    std::uint32_t eth_src_overflow_addr = SANITIZE_GET_RTA(11);
    std::uint32_t eth_dest_overflow_addr = SANITIZE_GET_RTA(12);
    bool use_multicast_semaphore_inc = static_cast<bool>(SANITIZE_GET_RTA(13));
    std::uint32_t mcast_dst_end_x = SANITIZE_GET_RTA(14);
    std::uint32_t mcast_dst_end_y = SANITIZE_GET_RTA(15);

    // We will assert later. This kernel will hang.
    // Need to signal completion to dispatcher before hanging so that
    // Dispatcher Kernel is able to finish.
    // Device Close () requires fast dispatch kernels to finish.
    volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    // Signal completion to dispatcher before assert hangs the kernel
    // SD signaling: IDLE_ERISC (all archs) and Quasar DM require RUN_MSG_DONE
    // TODO: Remove COMPILE_FOR_DM once FD is enabled on Quasar
#if defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_DM)
    go_message_in->signal = RUN_MSG_DONE;
#else
    uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
    notify_dispatch_core_done(dispatch_addr, noc_index);
#endif

    if (l1_overflow_addr) {
        CoreLocalMem<std::uint32_t> l1_overflow_buffer(l1_overflow_addr);
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
    Noc noc;
    CoreLocalMem<std::uint32_t> local_buffer(local_buffer_addr);
    UnicastEndpoint src_unicast_endpoint;

    noc.async_read(
        src_unicast_endpoint,
        local_buffer,
        buffer_size,
        {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = buffer_src_addr},
        {});
    noc.async_read_barrier();

    // NOC dst address
    if (bad_linked_transaction) {
        MulticastEndpoint dst_mcast_endpoint;
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

    UnicastEndpoint dst_unicast_endpoint;
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
