// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"

// TODO: commonize pieces with command_queue_producer
void kernel_main() {
    constexpr uint32_t host_issue_queue_read_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t issue_queue_start_addr = get_compile_time_arg_val(1);
    uint32_t issue_queue_size = get_compile_time_arg_val(2);  // not constexpr since can change
    constexpr uint32_t command_start_addr = get_compile_time_arg_val(3);
    constexpr uint32_t data_section_addr = get_compile_time_arg_val(4);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(5);
    constexpr uint32_t consumer_cmd_base_addr = get_compile_time_arg_val(6);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(7);

    setup_issue_queue_read_interface(issue_queue_start_addr, issue_queue_size);

    // Initialize the producer/consumer DB semaphore
    // This represents how many buffers the producer can write to.
    // At the beginning, it can write to two different buffers.
    uint32_t producer_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));
    uint32_t eth_consumer_noc_encoding = uint32_t(NOC_XY_ENCODING(CONSUMER_NOC_X, CONSUMER_NOC_Y));
    uint32_t pcie_core_noc_encoding = uint32_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));

    volatile tt_l1_ptr uint32_t* db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to num command slots by host (1 for remote cq)

    bool db_buf_switch = false; //TODO: toggle db buf switch when adding double buffering on eth core
    while (true) {

        issue_queue_wait_front();

        // Read in command
        uint32_t rd_ptr = (cq_read_interface.issue_fifo_rd_ptr << 4);
        uint64_t src_noc_addr = ((uint64_t)pcie_core_noc_encoding << 32) | rd_ptr;
        noc_async_read(src_noc_addr, command_start_addr, min(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, issue_queue_size - rd_ptr));
        noc_async_read_barrier();

        // Producer information
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;
        uint32_t data_size = header->issue_data_size;
        uint32_t num_buffer_transfers = header->num_buffer_transfers;
        uint32_t page_size = header->page_size;
        uint32_t producer_cb_size = header->producer_cb_size;
        uint32_t consumer_cb_size = header->router_cb_size;
        uint32_t producer_cb_num_pages = header->producer_cb_num_pages;
        uint32_t consumer_cb_num_pages = header->router_cb_num_pages; // AL: is this zero?
        uint32_t num_pages = header->num_pages;
        uint32_t producer_consumer_transfer_num_pages = header->producer_router_transfer_num_pages;
        uint32_t sharded_buffer_num_cores = header->sharded_buffer_num_cores;
        uint32_t wrap = header->wrap;

        if ((DeviceCommand::WrapRegion)wrap == DeviceCommand::WrapRegion::ISSUE) {
            // Basically popfront without the extra conditional
            cq_read_interface.issue_fifo_rd_ptr = cq_read_interface.issue_fifo_limit - cq_read_interface.issue_fifo_size;  // Head to beginning of command queue
            cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
            notify_host_of_issue_queue_read_pointer<host_issue_queue_read_ptr_addr>();
            continue;
        }

        db_cb_config_t* db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, false);
        const db_cb_config_t* eth_db_cb_config =
            get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, false);

        program_local_cb(data_section_addr, producer_cb_num_pages, page_size, producer_cb_size);
        wait_consumer_space_available(db_semaphore_addr);
        program_consumer_cb<command_start_addr, data_buffer_size, consumer_cmd_base_addr, consumer_data_buffer_size>(
            db_cb_config,
            eth_db_cb_config,
            db_buf_switch,
            ((uint64_t)eth_consumer_noc_encoding << 32),
            consumer_cb_num_pages,
            page_size,
            consumer_cb_size);
        relay_command<command_start_addr, consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, ((uint64_t)eth_consumer_noc_encoding << 32));

        update_producer_consumer_sync_semaphores(((uint64_t)producer_noc_encoding << 32), ((uint64_t)eth_consumer_noc_encoding << 32), db_semaphore_addr, uint32_t(eth_get_semaphore(0)));

        // Fetch data and send to the consumer
        produce_for_eth_src_router<true, consumer_cmd_base_addr, consumer_data_buffer_size>(
            command_ptr,
            num_buffer_transfers,
            sharded_buffer_num_cores,
            producer_cb_size,
            producer_cb_num_pages,
            ((uint64_t)eth_consumer_noc_encoding << 32),
            producer_consumer_transfer_num_pages,
            db_buf_switch,
            db_cb_config,
            eth_db_cb_config);

        issue_queue_pop_front<host_issue_queue_read_ptr_addr>(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size);
    }
}
