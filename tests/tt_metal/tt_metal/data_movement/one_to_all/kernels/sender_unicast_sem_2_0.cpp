// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

// Sender unicast semaphore kernel (device 2.0 API).
// Per transaction, waits for all subordinates to signal readiness, then for each
// subordinate: unicasts data via Noc::async_write and signals that subordinate's
// receiver_sem via the new Semaphore<>::relay_unicast(dst_sem, ...) method (which
// targets the receiver's distinct L1 offset, not the sender's local valid_sem slot).
void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t mst_base_addr = get_compile_time_arg_val(0);
    constexpr uint32_t sub_base_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_transaction = get_compile_time_arg_val(3);
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(4);
    constexpr uint32_t test_id = get_compile_time_arg_val(5);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(6);
    constexpr uint32_t num_virtual_channels = get_compile_time_arg_val(7);

    // Semaphore arguments
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t sender_valid_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(10);

    // Derivative values
    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;

    Semaphore<> sender_sem(sender_sem_id);
    Semaphore<> sender_valid_sem(sender_valid_sem_id);
    Semaphore<> receiver_sem(receiver_sem_id);

    {
        DeviceZoneScopedN("RISCV0");

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            sender_sem.wait(num_subordinates);
            sender_sem.set(0);

            for (uint32_t subordinate_num = 0; subordinate_num < num_subordinates; subordinate_num++) {
                uint32_t dest_coord_packed = get_arg_val<uint32_t>(subordinate_num);
                uint32_t dest_coord_x = dest_coord_packed >> 16;
                uint32_t dest_coord_y = dest_coord_packed & 0xFFFF;

                uint32_t current_virtual_channel = i % num_virtual_channels;

                noc.async_write<NocOptions::CUSTOM_VC>(
                    unicast_endpoint,
                    unicast_endpoint,
                    bytes_per_transaction,
                    {.addr = mst_base_addr},
                    {.noc_x = dest_coord_x, .noc_y = dest_coord_y, .addr = sub_base_addr},
                    {.vc = current_virtual_channel});

                // Flush: async_write uses caller-supplied VC; relay_unicast goes through
                // noc_semaphore_set_remote which hardcodes NOC_UNICAST_WRITE_VC. Same NoC,
                // different VC has no ordering guarantee, so the sem could overtake the data.
                noc.async_writes_flushed();
                sender_valid_sem.relay_unicast(noc, receiver_sem, dest_coord_x, dest_coord_y);
            }
        }
        noc.async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_subordinates);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
    DeviceTimestampedData("NoC Index", noc_index);
    DeviceTimestampedData("Number of subordinates", num_subordinates);
}
