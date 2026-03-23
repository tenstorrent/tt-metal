// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified L1 write kernel for NOC estimator tests.
// Supports unicast single/multi, multicast, and multicast linked modes.
// For unicast modes: stateful=true uses set_state+with_state, stateful=false uses async_write.
// For multicast modes: always uses async_write_multicast (no stateful variant exists).

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"
#include "log_helpers.hpp"

void kernel_main() {
    // ============ Compile-time arguments ============
    // Common args
    constexpr uint32_t src_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dst_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr uint32_t mode = get_compile_time_arg_val(5);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(6);
    constexpr uint32_t stateful = get_compile_time_arg_val(7);
    constexpr uint32_t num_virtual_channels = get_compile_time_arg_val(8);

    // Multicast-specific args (unused modes set to 0)
    constexpr uint32_t loopback = get_compile_time_arg_val(9);
    uint32_t mcast_start_x = get_compile_time_arg_val(10);
    uint32_t mcast_start_y = get_compile_time_arg_val(11);
    uint32_t mcast_end_x = get_compile_time_arg_val(12);
    uint32_t mcast_end_y = get_compile_time_arg_val(13);

    // Unicast single: packed destination coordinate
    constexpr uint32_t packed_dest_coord = get_compile_time_arg_val(14);

    // Metadata args (for logging only, not used in kernel logic)
    constexpr uint32_t memory_type = get_compile_time_arg_val(15);
    constexpr uint32_t mechanism = get_compile_time_arg_val(16);
    constexpr uint32_t pattern = get_compile_time_arg_val(17);
    constexpr uint32_t same_axis = get_compile_time_arg_val(18);
    constexpr uint32_t loopback_meta = get_compile_time_arg_val(19);

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_ep;
    experimental::MulticastEndpoint multicast_ep;

    // ============ MODE 0: UNICAST SINGLE (one_to_one) ============
    if constexpr (mode == WRITER_MODE_UNICAST_SINGLE) {
        constexpr uint32_t dest_x = packed_dest_coord >> 16;
        constexpr uint32_t dest_y = packed_dest_coord & 0xFFFF;

        if constexpr (stateful) {
            {
                DeviceZoneScopedN("RISCV0");
                noc.set_async_write_state(
                    unicast_ep, bytes_per_transaction, {.noc_x = dest_x, .noc_y = dest_y, .addr = dst_addr});
                for (uint32_t i = 0; i < num_of_transactions; i++) {
                    noc.async_write_with_state(
                        unicast_ep,
                        unicast_ep,
                        bytes_per_transaction,
                        {.addr = src_addr},
                        {.noc_x = dest_x, .noc_y = dest_y, .addr = dst_addr});
                }
                noc.async_write_barrier();
            }
        } else {
            {
                DeviceZoneScopedN("RISCV0");
                for (uint32_t i = 0; i < num_of_transactions; i++) {
                    uint32_t vc = i % num_virtual_channels;
                    noc.async_write(
                        unicast_ep,
                        unicast_ep,
                        bytes_per_transaction,
                        {.addr = src_addr},
                        {.noc_x = dest_x, .noc_y = dest_y, .addr = dst_addr},
                        vc);
                }
                noc.async_write_barrier();
            }
        }

        log_estimator_metadata(
            test_id,
            noc.get_noc_id(),
            num_of_transactions,
            bytes_per_transaction,
            memory_type,
            mechanism,
            pattern,
            0,
            same_axis,
            stateful,
            loopback_meta);
    }
    // ============ MODE 1: UNICAST MULTI (one_to_all unicast, all_to_all) ============
    else if constexpr (mode == WRITER_MODE_UNICAST_MULTI) {
        {
            DeviceZoneScopedN("RISCV0");
            for (uint32_t sub = 0; sub < num_subordinates; sub++) {
                uint32_t dest_packed = get_arg_val<uint32_t>(sub);
                uint32_t dest_x = dest_packed >> 16;
                uint32_t dest_y = dest_packed & 0xFFFF;

                if constexpr (stateful) {
                    noc.set_async_write_state(
                        unicast_ep, bytes_per_transaction, {.noc_x = dest_x, .noc_y = dest_y, .addr = dst_addr});
                    for (uint32_t i = 0; i < num_of_transactions; i++) {
                        noc.async_write_with_state(
                            unicast_ep,
                            unicast_ep,
                            bytes_per_transaction,
                            {.addr = src_addr},
                            {.noc_x = dest_x, .noc_y = dest_y, .addr = dst_addr});
                    }
                } else {
                    for (uint32_t i = 0; i < num_of_transactions; i++) {
                        // uint32_t vc = i % num_virtual_channels;
                        noc.async_write(
                            unicast_ep,
                            unicast_ep,
                            bytes_per_transaction,
                            {.addr = src_addr},
                            {.noc_x = dest_x, .noc_y = dest_y, .addr = dst_addr},
                            0);
                    }
                }
            }
            noc.async_write_barrier();
        }

        log_estimator_metadata(
            test_id,
            noc.get_noc_id(),
            num_of_transactions * num_subordinates,
            bytes_per_transaction,
            memory_type,
            mechanism,
            pattern,
            num_subordinates,
            same_axis,
            stateful,
            loopback_meta);
    }
    // ============ MODE 2: MULTICAST (non-linked) ============
    else if constexpr (mode == WRITER_MODE_MULTICAST) {
        if (noc_index == 1) {
            uint32_t tmp;
            tmp = mcast_start_x;
            mcast_start_x = mcast_end_x;
            mcast_end_x = tmp;
            tmp = mcast_start_y;
            mcast_start_y = mcast_end_y;
            mcast_end_y = tmp;
        }

        constexpr experimental::Noc::McastMode mcast_mode =
            loopback ? experimental::Noc::McastMode::INCLUDE_SRC : experimental::Noc::McastMode::EXCLUDE_SRC;

        {
            DeviceZoneScopedN("RISCV0");
            for (uint32_t i = 0; i < num_of_transactions; i++) {
                noc.async_write_multicast<mcast_mode>(
                    unicast_ep,
                    multicast_ep,
                    bytes_per_transaction,
                    num_subordinates,
                    {.addr = src_addr},
                    {.noc_x_start = mcast_start_x,
                     .noc_y_start = mcast_start_y,
                     .noc_x_end = mcast_end_x,
                     .noc_y_end = mcast_end_y,
                     .addr = dst_addr});
            }
            noc.async_write_barrier();
        }

        log_estimator_metadata(
            test_id,
            noc.get_noc_id(),
            num_of_transactions,
            bytes_per_transaction,
            memory_type,
            mechanism,
            pattern,
            num_subordinates,
            same_axis,
            stateful,
            loopback_meta);
    }
    // ============ MODE 3: MULTICAST LINKED ============
    else if constexpr (mode == WRITER_MODE_MULTICAST_LINKED) {
        if (noc_index == 1) {
            uint32_t tmp;
            tmp = mcast_start_x;
            mcast_start_x = mcast_end_x;
            mcast_end_x = tmp;
            tmp = mcast_start_y;
            mcast_start_y = mcast_end_y;
            mcast_end_y = tmp;
        }

        constexpr experimental::Noc::McastMode mcast_mode =
            loopback ? experimental::Noc::McastMode::INCLUDE_SRC : experimental::Noc::McastMode::EXCLUDE_SRC;

        {
            DeviceZoneScopedN("RISCV0");
            // All but the last packet: linked=true to reserve the VC path
            for (uint32_t i = 0; i < num_of_transactions - 1; i++) {
                noc.async_write_multicast<mcast_mode>(
                    unicast_ep,
                    multicast_ep,
                    bytes_per_transaction,
                    num_subordinates,
                    {.addr = src_addr},
                    {.noc_x_start = mcast_start_x,
                     .noc_y_start = mcast_start_y,
                     .noc_x_end = mcast_end_x,
                     .noc_y_end = mcast_end_y,
                     .addr = dst_addr},
                    true);  // linked
            }
            // Last packet: unlinked to release the VC
            noc.async_write_multicast<mcast_mode>(
                unicast_ep,
                multicast_ep,
                bytes_per_transaction,
                num_subordinates,
                {.addr = src_addr},
                {.noc_x_start = mcast_start_x,
                 .noc_y_start = mcast_start_y,
                 .noc_x_end = mcast_end_x,
                 .noc_y_end = mcast_end_y,
                 .addr = dst_addr});
            noc.async_write_barrier();
        }

        log_estimator_metadata(
            test_id,
            noc.get_noc_id(),
            num_of_transactions,
            bytes_per_transaction,
            memory_type,
            mechanism,
            pattern,
            num_subordinates,
            same_axis,
            stateful,
            loopback_meta);
    }
}
