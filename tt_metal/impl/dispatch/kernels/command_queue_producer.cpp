// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"

static constexpr uint32_t COMMAND_START_ADDR =
    L1_UNRESERVED_BASE;  // Space between UNRESERVED_BASE -> data_start is for commands

FORCE_INLINE
void program_local_cb(uint32_t num_pages, uint32_t page_size, uint32_t cb_size) {
    uint32_t cb_id = 0;
    uint32_t fifo_addr = DeviceCommand::DATA_SECTION_ADDRESS >> 4;
    uint32_t fifo_limit = fifo_addr + (cb_size >> 4);
    cb_interface[cb_id].fifo_limit = fifo_limit;  // to check if we need to wrap
    cb_interface[cb_id].fifo_wr_ptr = fifo_addr;
    cb_interface[cb_id].fifo_rd_ptr = fifo_addr;
    cb_interface[cb_id].fifo_size = cb_size >> 4;
    cb_interface[cb_id].tiles_acked = 0;
    cb_interface[cb_id].tiles_received = 0;
    cb_interface[cb_id].fifo_num_pages = num_pages;
    cb_interface[cb_id].fifo_page_size = page_size >> 4;
}

FORCE_INLINE
void program_consumer_cb(bool db_buf_switch, uint64_t consumer_noc_encoding, uint32_t num_pages, uint32_t page_size, uint32_t cb_size) {
    /*
        This API programs the double-buffered CB space of the consumer. This API should be called
        before notifying the consumer that data is available.
    */
    uint32_t acked_addr = get_db_cb_ack_addr(db_buf_switch);
    uint32_t recv_addr = get_db_cb_recv_addr(db_buf_switch);
    uint32_t num_pages_addr = get_db_cb_num_pages_addr(db_buf_switch);
    uint32_t page_size_addr = get_db_cb_page_size_addr(db_buf_switch);
    uint32_t total_size_addr = get_db_cb_total_size_addr(db_buf_switch);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(acked_addr)[0] = 0;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_addr)[0] = 0;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num_pages_addr)[0] = num_pages;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_size_addr)[0] = page_size >> 4;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(total_size_addr)[0] = cb_size >> 4;

    uint32_t rd_ptr_addr = get_db_cb_rd_ptr_addr(db_buf_switch);
    uint32_t wr_ptr_addr = get_db_cb_wr_ptr_addr(db_buf_switch);
    uint32_t cb_start_addr = get_db_buf_addr(db_buf_switch);
    reinterpret_cast<volatile uint32_t*>(rd_ptr_addr)[0] = cb_start_addr >> 4;
    reinterpret_cast<volatile uint32_t*>(wr_ptr_addr)[0] = cb_start_addr >> 4;

    uint32_t cb_base = get_db_cb_l1_base(db_buf_switch);
    noc_async_write(cb_base, consumer_noc_encoding | cb_base, 7 * 16);
    noc_async_write_barrier();  // barrier for now
}

// Only the read interface is set up on the device... the write interface
// belongs to host
void setup_issue_queue_read_interface(const uint32_t issue_region_rd_ptr, const uint32_t issue_region_size) {
    cq_read_interface.issue_fifo_rd_ptr = issue_region_rd_ptr >> 4;
    cq_read_interface.issue_fifo_size = issue_region_size >> 4;
    cq_read_interface.issue_fifo_limit = (issue_region_rd_ptr + issue_region_size) >> 4;
    cq_read_interface.issue_fifo_rd_toggle = 0;
}

void kernel_main() {
    constexpr uint32_t host_issue_queue_read_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t issue_queue_start_addr = get_compile_time_arg_val(1);
    constexpr uint32_t issue_queue_size = get_compile_time_arg_val(2);

    setup_issue_queue_read_interface(issue_queue_start_addr, issue_queue_size);

    // Initialize the producer/consumer DB semaphore
    // This represents how many buffers the producer can write to.
    // At the beginning, it can write to two different buffers.
    uint64_t producer_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;
    uint64_t consumer_noc_encoding = uint64_t(NOC_XY_ENCODING(CONSUMER_NOC_X, CONSUMER_NOC_Y)) << 32;
    uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32;

    volatile tt_l1_ptr uint32_t* db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to 2 by host

    bool db_buf_switch = false;
    while (true) {

        issue_queue_wait_front();

        // Read in command
        uint32_t rd_ptr = (cq_read_interface.issue_fifo_rd_ptr << 4);
        uint64_t src_noc_addr = pcie_core_noc_encoding | rd_ptr;
        noc_async_read(src_noc_addr, COMMAND_START_ADDR, min(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, issue_queue_size - rd_ptr));
        noc_async_read_barrier();

        // Producer information
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(COMMAND_START_ADDR);
        uint32_t data_size = command_ptr[DeviceCommand::data_size_idx];
        uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];
        uint32_t stall = command_ptr[DeviceCommand::stall_idx];
        uint32_t page_size = command_ptr[DeviceCommand::page_size_idx];
        uint32_t producer_cb_size = command_ptr[DeviceCommand::producer_cb_size_idx];
        uint32_t consumer_cb_size = command_ptr[DeviceCommand::consumer_cb_size_idx];
        uint32_t producer_cb_num_pages = command_ptr[DeviceCommand::producer_cb_num_pages_idx];
        uint32_t consumer_cb_num_pages = command_ptr[DeviceCommand::consumer_cb_num_pages_idx];
        uint32_t num_pages = command_ptr[DeviceCommand::num_pages_idx];
        uint32_t wrap = command_ptr[DeviceCommand::wrap_idx];
        uint32_t producer_consumer_transfer_num_pages = command_ptr[DeviceCommand::producer_consumer_transfer_num_pages_idx];
        uint32_t sharded_buffer_num_cores = command_ptr[DeviceCommand::sharded_buffer_num_cores_idx];
        uint32_t finish = command_ptr[DeviceCommand::finish_idx];

        if ((DeviceCommand::WrapRegion)wrap == DeviceCommand::WrapRegion::ISSUE) {
            // Basically popfront without the extra conditional
            cq_read_interface.issue_fifo_rd_ptr = cq_read_interface.issue_fifo_limit - cq_read_interface.issue_fifo_size;  // Head to beginning of command queue
            cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
            notify_host_of_issue_queue_read_pointer();
            continue;
        }

        program_local_cb(producer_cb_num_pages, page_size, producer_cb_size);
        while (db_semaphore_addr[0] == 0)
            ;  // Check that there is space in the consumer
        program_consumer_cb(db_buf_switch, consumer_noc_encoding, consumer_cb_num_pages, page_size, consumer_cb_size);
        relay_command(db_buf_switch, consumer_noc_encoding);
        if (stall) {
            while (*db_semaphore_addr != 2)
                ;
        }
        // Decrement the semaphore value
        noc_semaphore_inc(producer_noc_encoding | uint32_t(db_semaphore_addr), -1);  // Two's complement addition
        noc_async_write_barrier();

        // Notify the consumer
        noc_semaphore_inc(consumer_noc_encoding | get_semaphore(0), 1);
        noc_async_write_barrier();  // Barrier for now

        // Fetch data and send to the consumer
        produce(
            command_ptr,
            num_buffer_transfers,
            sharded_buffer_num_cores,
            page_size,
            producer_cb_size,
            producer_cb_num_pages,
            consumer_cb_size,
            consumer_cb_num_pages,
            consumer_noc_encoding,
            producer_consumer_transfer_num_pages,
            db_buf_switch);

        issue_queue_pop_front(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + data_size);

        db_buf_switch = not db_buf_switch;
    }
}
