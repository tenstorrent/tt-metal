/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_consumer.hpp"


void kernel_main() {
    constexpr u32 tensix_soft_reset_addr = get_compile_time_arg_val(0);
    bool db_buf_switch = false;
    volatile u32* db_semaphore_addr = reinterpret_cast<volatile u32*>(SEMAPHORE_BASE);

    static constexpr u32 command_start_addr = L1_UNRESERVED_BASE; // Space between L1_UNRESERVED_BASE -> data_start is for commands

    u64 producer_noc_encoding = u64(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y)) << 32;
    u64 consumer_noc_encoding = u64(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;

    // TODO(agrebenisan): Add wrap/dispatch functionality back in
    while (true) {
        // Wait for producer to supply a command
        db_acquire(db_semaphore_addr, consumer_noc_encoding);

        // For each instruction, we need to jump to the relevant part of the device command
        u32 command_start_addr = get_command_slot_addr(db_buf_switch);
        u32 dispatch_go_signal_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(u32));
        u32 buffer_transfer_start_addr = dispatch_go_signal_start_addr + (DeviceCommand::NUM_POSSIBLE_GO_SIGNALS * sizeof(u32));
        u32 program_transfer_start_addr = buffer_transfer_start_addr + ((DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION * DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS) * sizeof(u32));

        volatile tt_l1_ptr u32* command_ptr = reinterpret_cast<volatile u32*>(command_start_addr);
        u32 finish = command_ptr[DeviceCommand::finish_idx];       // Whether to notify the host that we have finished
        u32 num_workers = command_ptr[DeviceCommand::num_workers_idx];  // If num_workers > 0, it means we are launching a program
        u32 num_multicast_messages = command_ptr[DeviceCommand::num_multicast_messages_idx];
        u32 num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];   // How many WriteBuffer commands we are running
        u32 is_program = command_ptr[DeviceCommand::is_program_buffer_idx];
        u32 page_size = command_ptr[DeviceCommand::page_size_idx];
        u32 consumer_cb_size = command_ptr[DeviceCommand::consumer_cb_size_idx];
        u32 num_pages = command_ptr[DeviceCommand::num_pages_idx];

        if (is_program) {
            command_ptr = reinterpret_cast<volatile tt_l1_ptr u32*>(program_transfer_start_addr);
            write_program(num_pages, command_ptr, producer_noc_encoding, consumer_cb_size, db_buf_switch);
            command_ptr = reinterpret_cast<volatile tt_l1_ptr u32*>(dispatch_go_signal_start_addr);
            launch_program(num_workers, num_multicast_messages, command_ptr, tensix_soft_reset_addr);
        } else {
            command_ptr = reinterpret_cast<volatile tt_l1_ptr u32*>(buffer_transfer_start_addr);
            write_buffers(command_ptr, num_buffer_transfers, consumer_cb_size, producer_noc_encoding, db_buf_switch);
        }

        if (finish) {
            notify_host_complete();
        }

        // notify producer that it has completed a command
        noc_semaphore_inc(producer_noc_encoding | get_semaphore(0), 1);
        db_buf_switch = not db_buf_switch;
        noc_async_write_barrier(); // Barrier for now
    }
}
