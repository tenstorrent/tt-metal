// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_consumer.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"

// Dispatches fast dispatch commands to worker cores. Currently only runs on remote devices
void kernel_main() {
    constexpr uint32_t cmd_base_addr = get_compile_time_arg_val(0);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(1);
    constexpr uint32_t signaller_cmd_base_addr = get_compile_time_arg_val(2);
    constexpr uint32_t signaller_data_buffer_size = get_compile_time_arg_val(3);

    volatile tt_l1_ptr uint32_t* db_rx_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to 0 by host
    volatile tt_l1_ptr uint32_t* db_tx_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));  // Should be num command slots in the remote signaller

    uint32_t producer_noc_encoding = uint32_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y));
    uint32_t dispatcher_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));
    uint32_t signaller_noc_encoding = uint32_t(NOC_XY_ENCODING(SIGNALLER_NOC_X, SIGNALLER_NOC_Y));

    bool db_rx_buf_switch = false;
    bool db_tx_buf_switch = false;
    while (true) {
        // Wait for producer to supply a command
        db_acquire(db_rx_semaphore_addr, ((uint64_t)dispatcher_noc_encoding << 32));

        // For each instruction, we need to jump to the relevant part of the device command
        uint32_t command_start_addr = get_command_slot_addr<cmd_base_addr, data_buffer_size>(db_rx_buf_switch);
        uint32_t buffer_transfer_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t));

        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;
        volatile tt_l1_ptr uint32_t *buffer_transfer_command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_transfer_start_addr);
        uint32_t is_program = header->is_program_buffer;

        const uint32_t dst_buf_type = buffer_transfer_command_ptr[5];
        bool reading_buffer = (!is_program) & ((BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY);

        tt_l1_ptr db_cb_config_t *db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, db_rx_buf_switch);
        const tt_l1_ptr db_cb_config_t *remote_producer_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, db_rx_buf_switch);

        uint32_t producer_consumer_transfer_num_pages = header->producer_consumer_transfer_num_pages;
        if (is_program) {
            uint32_t program_transfer_start_addr = buffer_transfer_start_addr + ((DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION * DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS) * sizeof(uint32_t));
            uint32_t num_pages = header->num_pages;
            uint32_t num_workers = header->num_workers;  // If num_workers > 0, it means we are launching a program
            write_and_launch_program(
                db_cb_config,
                remote_producer_db_cb_config,
                program_transfer_start_addr,
                num_pages,
                command_ptr,
                ((uint64_t)producer_noc_encoding << 32),
                producer_consumer_transfer_num_pages);
            wait_for_program_completion(num_workers);
        } else if (!reading_buffer) {
            uint32_t num_buffer_transfers = header->num_buffer_transfers;   // How many WriteBuffer commands we are running
            uint32_t sharded_buffer_num_cores = header->sharded_buffer_num_cores;
            write_remote_buffers(
                db_cb_config,
                remote_producer_db_cb_config,
                buffer_transfer_command_ptr,
                num_buffer_transfers,
                sharded_buffer_num_cores,
                ((uint64_t)producer_noc_encoding << 32),
                producer_consumer_transfer_num_pages);
        }

        // relay command to remote signaller
        wait_consumer_space_available(db_tx_semaphore_addr);    // Check that there is space in the remote signaller
        const tt_l1_ptr db_cb_config_t *signaller_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, db_tx_buf_switch);
        uint32_t consumer_cb_num_pages = header->consumer_cb_num_pages;
        uint32_t page_size = header->page_size;
        uint32_t consumer_cb_size = header->consumer_cb_size;
        program_consumer_cb<cmd_base_addr, data_buffer_size, signaller_cmd_base_addr, signaller_data_buffer_size>(
            db_cb_config,
            signaller_db_cb_config,
            db_tx_buf_switch,
            ((uint64_t)signaller_noc_encoding << 32),
            consumer_cb_num_pages,
            page_size,
            consumer_cb_size);
        relay_command<cmd_base_addr, signaller_cmd_base_addr, signaller_data_buffer_size>(db_tx_buf_switch, ((uint64_t)signaller_noc_encoding << 32));

        update_producer_consumer_sync_semaphores(((uint64_t)dispatcher_noc_encoding << 32), ((uint64_t)signaller_noc_encoding << 32), db_tx_semaphore_addr, get_semaphore(0));

        // if (reading_buffer) {
            // Command is requesting to read data back from device, need to transfer buffer data to the remote signaller
            // read_remote_buffers();
        // }
        db_tx_buf_switch = not db_tx_buf_switch;

        // notify producer that it has completed a command
        noc_semaphore_inc(((uint64_t)producer_noc_encoding << 32) | get_semaphore(0), 1);
        db_rx_buf_switch = not db_rx_buf_switch;
        noc_async_write_barrier(); // Barrier for now
    }
}
