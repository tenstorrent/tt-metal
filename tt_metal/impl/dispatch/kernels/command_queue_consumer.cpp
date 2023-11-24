// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_consumer.hpp"

void kernel_main() {
    constexpr uint32_t tensix_soft_reset_addr = get_compile_time_arg_val(0);
    bool db_buf_switch = false;
    volatile uint32_t* db_semaphore_addr = reinterpret_cast<volatile uint32_t*>(SEMAPHORE_BASE);

    static constexpr uint32_t command_start_addr = L1_UNRESERVED_BASE; // Space between L1_UNRESERVED_BASE -> data_start is for commands

    uint64_t producer_noc_encoding = uint64_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y)) << 32;
    uint64_t consumer_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;

    while (true) {
        // Wait for producer to supply a command
        db_acquire(db_semaphore_addr, consumer_noc_encoding);

        // For each instruction, we need to jump to the relevant part of the device command
        uint32_t command_start_addr = get_command_slot_addr(db_buf_switch);
        uint32_t buffer_transfer_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t));
        uint32_t program_transfer_start_addr = buffer_transfer_start_addr + ((DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION * DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS) * sizeof(uint32_t));

        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        uint32_t finish = command_ptr[DeviceCommand::finish_idx];       // Whether to notify the host that we have finished
        uint32_t num_workers = command_ptr[DeviceCommand::num_workers_idx];  // If num_workers > 0, it means we are launching a program
        uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];   // How many WriteBuffer commands we are running
        uint32_t is_program = command_ptr[DeviceCommand::is_program_buffer_idx];
        uint32_t page_size = command_ptr[DeviceCommand::page_size_idx];
        uint32_t consumer_cb_size = command_ptr[DeviceCommand::consumer_cb_size_idx];
        uint32_t consumer_cb_num_pages = command_ptr[DeviceCommand::consumer_cb_num_pages_idx];
        uint32_t num_pages = command_ptr[DeviceCommand::num_pages_idx];
        uint32_t producer_consumer_transfer_num_pages = command_ptr[DeviceCommand::producer_consumer_transfer_num_pages_idx];
        uint32_t num_cores = command_ptr[DeviceCommand::num_cores];

        if (is_program) {
            write_and_launch_program(program_transfer_start_addr, num_pages, command_ptr, producer_noc_encoding, consumer_cb_size, consumer_cb_num_pages, producer_consumer_transfer_num_pages, db_buf_switch);
            wait_for_program_completion(num_workers, tensix_soft_reset_addr);
        } else {
            command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_transfer_start_addr);
            write_buffers(command_ptr, num_buffer_transfers, num_cores, consumer_cb_size, consumer_cb_num_pages, producer_noc_encoding, producer_consumer_transfer_num_pages, db_buf_switch);
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
