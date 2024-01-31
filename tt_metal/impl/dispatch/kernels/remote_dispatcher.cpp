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

    constexpr bool db_rx_buf_switch = false;
    constexpr bool db_tx_buf_switch = false;
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
        uint32_t num_pages = header->num_pages;

        const uint32_t dst_buf_type = buffer_transfer_command_ptr[5];
        bool reading_buffer = (!is_program) & (num_pages > 0 & (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY);

        tt_l1_ptr db_cb_config_t *rx_db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, true);
        const tt_l1_ptr db_cb_config_t *remote_producer_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, false);

        uint32_t producer_consumer_transfer_num_pages = header->producer_router_transfer_num_pages;
        if (is_program) {
            uint32_t program_transfer_start_addr = buffer_transfer_start_addr + ((DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION * DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS) * sizeof(uint32_t));
            uint32_t num_workers = header->num_workers;  // If num_workers > 0, it means we are launching a program
            write_and_launch_program(
                rx_db_cb_config,
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
                rx_db_cb_config,
                remote_producer_db_cb_config,
                buffer_transfer_command_ptr,
                num_buffer_transfers,
                sharded_buffer_num_cores,
                ((uint64_t)producer_noc_encoding << 32),
                producer_consumer_transfer_num_pages);
        }

        // Relay command to remote signaller
        wait_consumer_space_available(db_tx_semaphore_addr);    // Check that there is space in the remote signaller
        relay_command<cmd_base_addr, signaller_cmd_base_addr, signaller_data_buffer_size>(db_tx_buf_switch, ((uint64_t)signaller_noc_encoding << 32));
        update_producer_consumer_sync_semaphores(((uint64_t)dispatcher_noc_encoding << 32), ((uint64_t)signaller_noc_encoding << 32), db_tx_semaphore_addr, get_semaphore(1));

        if (reading_buffer) {
            // Command is requesting to read data back from device, need to read buffer data and transfer to the remote signaller
            // Use same API as prefetcher core to produce data for remote signaller, src buffer will either be in DRAM or L1
            uint32_t consumer_cb_num_pages = header->consumer_cb_num_pages;
            uint32_t page_size = header->page_size;
            uint32_t consumer_cb_size = header->consumer_cb_size;
            uint32_t data_section_addr = command_start_addr + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
            program_local_cb(data_section_addr, consumer_cb_num_pages, page_size, consumer_cb_size);
            tt_l1_ptr db_cb_config_t *tx_db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, false);
            const tt_l1_ptr db_cb_config_t *signaller_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, true); // CB between dispatcher and signaller
            program_consumer_cb<cmd_base_addr, data_buffer_size, signaller_cmd_base_addr, signaller_data_buffer_size>(
                tx_db_cb_config,
                signaller_db_cb_config,
                db_tx_buf_switch,
                ((uint64_t)signaller_noc_encoding << 32),
                consumer_cb_num_pages,
                page_size,
                consumer_cb_size);

            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            uint32_t sharded_buffer_num_cores = header->sharded_buffer_num_cores;
            uint32_t producer_router_transfer_num_pages = header->producer_router_transfer_num_pages;
            produce_for_eth_src_router<false, signaller_cmd_base_addr, signaller_data_buffer_size>(
                command_ptr,
                num_buffer_transfers,
                sharded_buffer_num_cores,
                consumer_cb_size,   // use consumer metadata because dispatcher is "consumer" from the command's pov but is the "producer" for return path to completion queue via signaller
                consumer_cb_num_pages,  // use consumer metadata because dispatcher is "consumer" from the command's pov but is the "producer" for return path to completion queue via signaller
                ((uint64_t)signaller_noc_encoding << 32),
                producer_router_transfer_num_pages,
                db_tx_buf_switch,
                tx_db_cb_config,
                signaller_db_cb_config
            );
        }

        // notify remote command processor that it has completed a command
        noc_semaphore_inc(((uint64_t)producer_noc_encoding << 32) | get_semaphore(1), 1);
        noc_async_write_barrier(); // Barrier for now
    }
}
