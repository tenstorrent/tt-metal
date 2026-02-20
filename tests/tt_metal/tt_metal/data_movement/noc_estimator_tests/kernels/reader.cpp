// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified L1 read kernel for NOC estimator tests.
// Supports single source and multi source read modes.
// stateful=true uses set_state+with_state, stateful=false uses async_read.
// All modes use the experimental Noc API.

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"
#include "log_helpers.hpp"

void kernel_main() {
    // ============ Compile-time arguments ============
    constexpr uint32_t local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t mode = get_compile_time_arg_val(4);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(5);
    constexpr uint32_t stateful = get_compile_time_arg_val(6);
    constexpr uint32_t num_virtual_channels = get_compile_time_arg_val(7);

    // Metadata args (for logging only)
    constexpr uint32_t memory_type = get_compile_time_arg_val(8);
    constexpr uint32_t mechanism = get_compile_time_arg_val(9);
    constexpr uint32_t pattern = get_compile_time_arg_val(10);
    constexpr uint32_t same_axis = get_compile_time_arg_val(11);
    constexpr uint32_t loopback_meta = get_compile_time_arg_val(12);

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_ep;

    // ============ MODE 0: READ SINGLE (one_from_one) ============
    if constexpr (mode == READER_MODE_SINGLE) {
        // Runtime args: source core coordinates
        uint32_t src_x = get_arg_val<uint32_t>(0);
        uint32_t src_y = get_arg_val<uint32_t>(1);

        if constexpr (stateful) {
            {
                DeviceZoneScopedN("RISCV1");
                noc.set_async_read_state(
                    unicast_ep, bytes_per_transaction, {.noc_x = src_x, .noc_y = src_y, .addr = local_addr});
                for (uint32_t i = 0; i < num_of_transactions; i++) {
                    noc.async_read_with_state(
                        unicast_ep,
                        unicast_ep,
                        bytes_per_transaction,
                        {.noc_x = src_x, .noc_y = src_y, .addr = local_addr},
                        {.addr = local_addr});
                }
                noc.async_read_barrier();
            }
        } else {
            {
                DeviceZoneScopedN("RISCV1");
                for (uint32_t i = 0; i < num_of_transactions; i++) {
                    uint32_t vc = i % num_virtual_channels;
                    noc.async_read(
                        unicast_ep,
                        unicast_ep,
                        bytes_per_transaction,
                        {.noc_x = src_x, .noc_y = src_y, .addr = local_addr},
                        {.addr = local_addr},
                        vc);
                }
                noc.async_read_barrier();
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
    // ============ MODE 1: READ MULTI (one_from_all, all_from_all) ============
    else if constexpr (mode == READER_MODE_MULTI) {
        {
            DeviceZoneScopedN("RISCV1");
            for (uint32_t sub = 0; sub < num_subordinates; sub++) {
                uint32_t src_x = get_arg_val<uint32_t>(sub * 2);
                uint32_t src_y = get_arg_val<uint32_t>(sub * 2 + 1);

                if constexpr (stateful) {
                    noc.set_async_read_state(
                        unicast_ep, bytes_per_transaction, {.noc_x = src_x, .noc_y = src_y, .addr = local_addr});

                    for (uint32_t i = 0; i < num_of_transactions; i++) {
                        noc.async_read_with_state(
                            unicast_ep,
                            unicast_ep,
                            bytes_per_transaction,
                            {.noc_x = src_x, .noc_y = src_y, .addr = local_addr},
                            {.addr = local_addr});
                    }
                } else {
                    for (uint32_t i = 0; i < num_of_transactions; i++) {
                        uint32_t vc = i % num_virtual_channels;
                        noc.async_read(
                            unicast_ep,
                            unicast_ep,
                            bytes_per_transaction,
                            {.noc_x = src_x, .noc_y = src_y, .addr = local_addr},
                            {.addr = local_addr},
                            vc);
                    }
                }
            }
            noc.async_read_barrier();
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
}
