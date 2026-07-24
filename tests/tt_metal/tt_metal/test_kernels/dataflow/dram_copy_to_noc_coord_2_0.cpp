// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of dram_copy_to_noc_coord.cpp.
// Compiled only for TENSIX data-movement cores (BRISC / NCRISC / DM). Ethernet
// callers continue to use dram_copy_to_noc_coord.cpp via the legacy host API.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "internal/firmware_common.h"
#include "internal/hw_thread.h"
#include "experimental/kernel_args.h"

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs. Any two RISC processors cannot use the same CMD_BUF. non_blocking APIs shouldn't be mixed with slow
 * noc.h APIs; explicit flushes need to be used since the calls are non-blocking.
 */
void kernel_main() {
#if defined(COMPILE_FOR_DM)
    uint32_t thread_idx = internal_::get_hw_thread_idx();

#if defined(TEST_MULTI_DM_SANITIZE_RACE)
    // Explicit sync barrier stress-tests the CAS in sanitize.h since DM0 (which wakes
    // other DMs) most reliably wins without one.
    constexpr uint32_t num_dms = get_arg(args::num_dms);
    constexpr uint32_t multi_dm_base_addr = get_arg(args::multi_dm_base_addr);
    constexpr uint32_t multi_dm_base_size = get_arg(args::multi_dm_base_size);
    constexpr uint32_t l1_sync_addr = get_arg(args::l1_sync_addr);
    uint64_t* l1_ptr = reinterpret_cast<uint64_t*>(l1_sync_addr);
    __atomic_add_fetch(l1_ptr, 1, __ATOMIC_RELAXED);
    while (__atomic_load_n(l1_ptr, __ATOMIC_ACQUIRE) != num_dms) {
    }
#else
    // Single-DM test: only the specified dm_id executes; others exit early.
    constexpr uint32_t dm_id = get_arg(args::dm_id);
    if (thread_idx != dm_id) {
        return;
    }
#endif
#endif

    std::uint32_t local_buffer_addr = get_arg(args::local_buffer_addr);

    std::uint32_t buffer_src_addr = get_arg(args::buffer_src_addr);
    std::uint32_t src_noc_x = get_arg(args::src_noc_x);
    std::uint32_t src_noc_y = get_arg(args::src_noc_y);

    std::uint32_t buffer_dst_addr = get_arg(args::buffer_dst_addr);
    std::uint32_t dst_noc_x = get_arg(args::dst_noc_x);
    std::uint32_t dst_noc_y = get_arg(args::dst_noc_y);

    std::uint32_t buffer_size = get_arg(args::buffer_size);

#if defined(COMPILE_FOR_DM) && defined(TEST_MULTI_DM_SANITIZE_RACE)
    buffer_dst_addr = (multi_dm_base_addr | static_cast<uint32_t>(thread_idx));
    buffer_size = (multi_dm_base_size | static_cast<uint32_t>(thread_idx));
#endif

    bool use_inline_dw_write = static_cast<bool>(get_arg(args::use_inline_dw_write));
    bool bad_linked_transaction = static_cast<bool>(get_arg(args::bad_linked_transaction));
    std::uint32_t l1_overflow_addr = get_arg(args::l1_overflow_addr);
    std::uint32_t eth_src_overflow_addr = get_arg(args::eth_src_overflow_addr);
    std::uint32_t eth_dest_overflow_addr = get_arg(args::eth_dest_overflow_addr);
    bool use_multicast_semaphore_inc = static_cast<bool>(get_arg(args::use_multicast_semaphore_inc));
    std::uint32_t mcast_dst_end_x = get_arg(args::mcast_dst_end_x);
    std::uint32_t mcast_dst_end_y = get_arg(args::mcast_dst_end_y);
    bool use_write_with_state = static_cast<bool>(get_arg(args::use_write_with_state));
    bool use_inline_dw_write_from_state = static_cast<bool>(get_arg(args::use_inline_dw_write_from_state));
    bool use_inline_dw_write_with_state = static_cast<bool>(get_arg(args::use_inline_dw_write_with_state));

    // We will assert later. This kernel will hang.
    // Need to signal completion to dispatcher before hanging so that
    // Dispatcher Kernel is able to finish.
    // Device::close() requires fast-dispatch kernels to finish.
    volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    // SD enabled on all archs: notify completion via RUN_MSG_DONE to mailbox. FD notify path
    // posts to a dispatcher absent under SD and wedges the NOC.
#if defined(WATCHER_KERNEL_SLOW_DISPATCH)
    go_message_in->signal = RUN_MSG_DONE;
#else
    uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
    notify_dispatch_core_done(dispatch_addr, noc_index);
#endif

    if (l1_overflow_addr) {
        CoreLocalMem<std::uint32_t> l1_overflow_buffer(l1_overflow_addr);
        l1_overflow_buffer[0] = 0xDEADBEEF;
    }

    // eth_src_overflow_addr / eth_dest_overflow_addr only fire from ETH cores, which
    // never run this 2.0 kernel. Leave the args in the schema for parity, but ignore here.

    if (use_multicast_semaphore_inc) {
        // Invalid multicast range triggers the watcher assertion.
        uint64_t dst_multicast_noc_addr =
            get_noc_multicast_addr(dst_noc_x, dst_noc_y, mcast_dst_end_x, mcast_dst_end_y, buffer_dst_addr);
        noc_semaphore_inc_multicast(dst_multicast_noc_addr, 1, 1);
        noc_async_atomic_barrier();
        return;
    }

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
        // Linked transaction left open: the next unicast hangs (intentional).
    }

    UnicastEndpoint dst_unicast_endpoint;
    if (use_inline_dw_write) {
        noc.inline_dw_write(
            dst_unicast_endpoint, local_buffer[0], {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = buffer_dst_addr});
    } else if (use_write_with_state) {
        // Stateful write: the destination coordinate is programmed into NOC_RET_ADDR by set_async_write_state.
        // Exercises DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_WITH_ADDR_AND_SIZE_STATE, which must reconstruct the
        // destination from NOC_RET_ADDR (not the sender's own coordinate in NOC_TARG_ADDR). buffer_size is kept
        // <= NOC_MAX_BURST_SIZE by the host so this takes the one-packet path with a deterministic length.
        noc.set_async_write_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            dst_unicast_endpoint, buffer_size, {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = buffer_dst_addr});
        noc.async_write_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            local_buffer,
            dst_unicast_endpoint,
            buffer_size,
            {},
            {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = buffer_dst_addr});
        noc.async_write_barrier();
    } else if (use_inline_dw_write_from_state) {
        // set_state programs the inline/atomic command buffer (AT/simple on Quasar); sanitize reads that state back
        // before issue.
        uint64_t dst = get_noc_addr(dst_noc_x, dst_noc_y, buffer_dst_addr);
        noc_inline_dw_write_set_state<false /*posted*/, true /*set_val*/>(dst, local_buffer[0], 0xF);
        DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc_index, write_at_cmd_buf);
    } else if (use_inline_dw_write_with_state) {
#if defined(ARCH_WORMHOLE) || defined(ARCH_QUASAR)
        // with_state mirrors cq_noc_inline_dw_write_with_state: program WR/complex command-buffer state, then
        // sanitize before issue. CQ_NOC_send is 0, so the write is not issued.
        uint64_t dst = get_noc_addr(dst_noc_x, dst_noc_y, buffer_dst_addr);
        noc_inline_dw_write_init_state<NCRISC_WR_REG_CMD_BUF, CQ_NOC_mkp>(noc_index, NOC_UNICAST_WRITE_VC);
        noc_inline_dw_write_with_state<NCRISC_WR_REG_CMD_BUF, CQ_NOC_INLINE_NDVB, CQ_NOC_wait, CQ_NOC_send>(
            noc_index, dst, local_buffer[0], 0xF);
        DEBUG_SANITIZE_NOC_ADDR_FROM_STATE(noc_index, NCRISC_WR_REG_CMD_BUF);
#endif
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
