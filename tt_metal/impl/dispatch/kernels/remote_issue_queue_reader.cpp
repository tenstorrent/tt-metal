// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"
// #include "debug/dprint.h"

// TODO: commonize pieces with command_queue_producer
void kernel_main() {
    constexpr uint32_t host_issue_queue_read_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t issue_queue_start_addr = get_compile_time_arg_val(1);
    constexpr uint32_t command_start_addr = get_compile_time_arg_val(2);
    constexpr uint32_t data_section_addr = get_compile_time_arg_val(3);
    constexpr uint32_t consumer_cmd_base_addr = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(5);

    // Only the issue queue size is a runtime argument
    uint32_t issue_queue_size = get_arg_val<uint32_t>(0);

    setup_issue_queue_read_interface(issue_queue_start_addr, issue_queue_size);

    // Initialize the producer/consumer DB semaphore
    // This represents how many buffers the producer can write to.
    // At the beginning, it can write to two different buffers.
    uint32_t producer_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));
    uint32_t consumer_noc_encoding = uint32_t(NOC_XY_ENCODING(CONSUMER_NOC_X, CONSUMER_NOC_Y));
    uint32_t pcie_core_noc_encoding = uint32_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));

    volatile tt_l1_ptr uint32_t* db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to num command slots by host (1 for remote cq)

    bool db_buf_switch = false;
    while (true) {

        issue_queue_wait_front();

        // Read in command
        uint32_t rd_ptr = (cq_read_interface.issue_fifo_rd_ptr << 4);
        uint64_t src_noc_addr = ((uint64_t)pcie_core_noc_encoding << 32) | rd_ptr;
        noc_async_read(src_noc_addr, command_start_addr, min(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, issue_queue_size - rd_ptr));
        noc_async_read_barrier();

        // Producer information
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        uint32_t data_size = command_ptr[DeviceCommand::data_size_idx];
        uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];
        uint32_t page_size = command_ptr[DeviceCommand::page_size_idx];
        uint32_t producer_cb_size = command_ptr[DeviceCommand::producer_cb_size_idx];
        uint32_t consumer_cb_size = command_ptr[DeviceCommand::consumer_cb_size_idx];
        uint32_t producer_cb_num_pages = command_ptr[DeviceCommand::producer_cb_num_pages_idx];
        uint32_t consumer_cb_num_pages = command_ptr[DeviceCommand::consumer_cb_num_pages_idx];
        uint32_t num_pages = command_ptr[DeviceCommand::num_pages_idx];
        uint32_t producer_consumer_transfer_num_pages = command_ptr[DeviceCommand::producer_consumer_transfer_num_pages_idx];
        uint32_t sharded_buffer_num_cores = command_ptr[DeviceCommand::sharded_buffer_num_cores_idx];
        uint32_t wrap = command_ptr[DeviceCommand::wrap_idx];

        if ((DeviceCommand::WrapRegion)wrap == DeviceCommand::WrapRegion::ISSUE) {
            // Basically popfront without the extra conditional
            cq_read_interface.issue_fifo_rd_ptr = cq_read_interface.issue_fifo_limit - cq_read_interface.issue_fifo_size;  // Head to beginning of command queue
            cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
            notify_host_of_issue_queue_read_pointer<host_issue_queue_read_ptr_addr>();
            continue;
        }

        program_local_cb(data_section_addr, producer_cb_num_pages, page_size, producer_cb_size);
        while (db_semaphore_addr[0] == 0)
            ;  // Check that there is space in the consumer
        // program_consumer_cb<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, ((uint64_t)consumer_noc_encoding << 32), consumer_cb_num_pages, page_size, consumer_cb_size);
        // relay_command<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, ((uint64_t)consumer_noc_encoding << 32));

        // Decrement the semaphore value
        noc_semaphore_inc(((uint64_t)producer_noc_encoding << 32) | uint32_t(db_semaphore_addr), -1);  // Two's complement addition
        noc_async_write_barrier();

        // Notify the consumer
        // noc_semaphore_inc( ((uint64_t)consumer_noc_encoding << 32) | get_semaphore(0), 1);
        // noc_async_write_barrier();  // Barrier for now

        // Fetch data and send to the consumer
        // produce<consumer_cmd_base_addr, consumer_data_buffer_size>(
        //     command_ptr,
        //     num_buffer_transfers,
        //     sharded_buffer_num_cores,
        //     page_size,
        //     producer_cb_size,
        //     producer_cb_num_pages,
        //     consumer_cb_size,
        //     consumer_cb_num_pages,
        //     ((uint64_t)consumer_noc_encoding << 32),
        //     producer_consumer_transfer_num_pages,
        //     db_buf_switch);

        issue_queue_pop_front<host_issue_queue_read_ptr_addr>(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size);

        // db_buf_switch = not db_buf_switch; // only 1 command slot on consumer side
    }
}
