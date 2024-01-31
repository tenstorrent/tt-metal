// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_consumer.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"
#include "debug/dprint.h"

// Dispatches fast dispatch commands to worker cores. Currently only runs on remote devices
void kernel_main() {
    constexpr uint32_t cmd_base_addr = get_compile_time_arg_val(0);
    constexpr uint32_t data_section_addr = get_compile_time_arg_val(1);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(2);
    constexpr uint32_t producer_cmd_base_addr = get_compile_time_arg_val(3);
    constexpr uint32_t producer_data_buffer_size = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_cmd_base_addr = get_compile_time_arg_val(5);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(6);

    volatile tt_l1_ptr uint32_t* db_rx_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));  // Should be initialized to 0 by host
    // Use semaphore 0 to sync with SRC eth since SRC eth acks connected producer core's semaphore 0 to indicate it is ready to receive next command
    //  producers to SRC eth are: remote issue queue reader and remote command signaller
    volatile tt_l1_ptr uint32_t* tx_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be num command slots in the eth router

    uint32_t producer_noc_encoding = uint32_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y));
    uint32_t signaller_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));
    uint32_t eth_consumer_noc_encoding = uint32_t(NOC_XY_ENCODING(CONSUMER_NOC_X, CONSUMER_NOC_Y));

    constexpr bool db_rx_buf_switch = false;
    constexpr bool tx_buf_switch = false; //TODO: toggle db buf switch when adding double buffering on eth core
    while (true) {
        // Wait for dispatcher to supply a command
        // Received command is either finished running or is a host request to read from device buffer
        db_acquire(db_rx_semaphore_addr, ((uint64_t)signaller_noc_encoding << 32));

        uint32_t command_start_addr = get_command_slot_addr<cmd_base_addr, data_buffer_size>(db_rx_buf_switch);
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;
        header->fwd_path = 0; // hacky

        wait_consumer_space_available(tx_semaphore_addr);   // Check that there is space in the eth router

        uint32_t buffer_transfer_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t));
        volatile tt_l1_ptr uint32_t * buffer_transfer_command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_transfer_start_addr);
        uint32_t is_program = header->is_program_buffer;
        uint32_t num_pages = header->num_pages;
        const uint32_t dst_buf_type = buffer_transfer_command_ptr[5];
        bool reading_buffer = (!is_program) & (num_pages > 0 & (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY);

        uint32_t num_buffer_transfers = header->num_buffer_transfers;
        if (!reading_buffer) {
            // Received this command just to signal that it finished
            // This is hacky(!) but here we clear out cmd metadata so ethernet routers and completion queue write interface do not expect incoming data
            header->num_buffer_transfers = 0;
        }

        relay_command<cmd_base_addr, consumer_cmd_base_addr, consumer_data_buffer_size>(tx_buf_switch, ((uint64_t)eth_consumer_noc_encoding << 32));

        update_producer_consumer_sync_semaphores(((uint64_t)signaller_noc_encoding << 32), ((uint64_t)eth_consumer_noc_encoding << 32), tx_semaphore_addr, eth_get_semaphore(0));

        if (reading_buffer) {
            tt_l1_ptr db_cb_config_t *rx_db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, true);
            tt_l1_ptr db_cb_config_t *tx_db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, false);
            const tt_l1_ptr db_cb_config_t *dispatcher_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, false);
            const tt_l1_ptr db_cb_config_t *eth_db_cb_config = get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, false);

            uint32_t consumer_cb_num_pages = header->router_cb_num_pages;
            uint32_t page_size = header->page_size;
            uint32_t consumer_cb_size = header->router_cb_size;

            program_consumer_cb<cmd_base_addr, data_buffer_size, consumer_cmd_base_addr, consumer_data_buffer_size>(
                tx_db_cb_config,
                eth_db_cb_config,
                tx_buf_switch,
                ((uint64_t)eth_consumer_noc_encoding << 32),
                consumer_cb_num_pages,
                page_size,
                consumer_cb_size);

            // Use consumer_cb_size_idx because this kernel is on the return path but device command sets up producer/consumer from fwd path pov
            uint32_t producer_cb_size = header->consumer_cb_size;
            uint32_t consumer_router_transfer_num_pages = header->consumer_router_transfer_num_pages;
            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            // get_db_buf_addr is set up to get address of first CQ slot only because currently remote FD does not have any cmd double buffering
            transfer<false>(
                rx_db_cb_config,
                tx_db_cb_config,
                dispatcher_db_cb_config,
                eth_db_cb_config,
                command_ptr,
                num_buffer_transfers,
                page_size,
                producer_cb_size,
                (get_db_buf_addr<producer_cmd_base_addr, producer_data_buffer_size>(false) + producer_cb_size) >> 4,
                ((uint64_t)producer_noc_encoding << 32),
                consumer_cb_size,
                (get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(false) + consumer_cb_size) >> 4,
                ((uint64_t)eth_consumer_noc_encoding << 32),
                consumer_router_transfer_num_pages,
                (get_db_buf_addr<cmd_base_addr, data_buffer_size>(false) + producer_cb_size) >> 4);
        }

        // Notify to dispatcher that is has completed a command
        noc_semaphore_inc(((uint64_t)producer_noc_encoding << 32) | get_semaphore(1), 1);
        noc_async_write_barrier(); // Barrier for now
    }
}
