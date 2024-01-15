// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_consumer.hpp"

// The read interface for the issue region is set up on the device, the write interface belongs to host
// Opposite for completion region where device sets up the write interface and host owns read interface
void setup_completion_queue_write_interface(const uint32_t completion_region_wr_ptr, const uint32_t completion_region_size) {
    cq_write_interface.completion_fifo_wr_ptr = completion_region_wr_ptr >> 4;
    cq_write_interface.completion_fifo_size = completion_region_size >> 4;
    cq_write_interface.completion_fifo_limit = (completion_region_wr_ptr + completion_region_size) >> 4;
    cq_write_interface.completion_fifo_wr_toggle = 0;
}

void kernel_main() {
    bool db_buf_switch = false;

    constexpr uint32_t host_completion_queue_write_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t completion_queue_start_addr = get_compile_time_arg_val(1);
    constexpr uint32_t completion_queue_size = get_compile_time_arg_val(2);
    constexpr uint32_t host_finish_addr = get_compile_time_arg_val(3);
    constexpr uint32_t cmd_base_address = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(5);

    volatile uint32_t* db_semaphore_addr = reinterpret_cast<volatile uint32_t*>(SEMAPHORE_BASE);

    uint64_t producer_noc_encoding = uint64_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y)) << 32;
    uint64_t consumer_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;

    setup_completion_queue_write_interface(completion_queue_start_addr, completion_queue_size);

    while (true) {
        // Wait for producer to supply a command
        db_acquire(db_semaphore_addr, consumer_noc_encoding);

        // For each instruction, we need to jump to the relevant part of the device command
        uint32_t command_start_addr = get_command_slot_addr<cmd_base_address, consumer_data_buffer_size>(db_buf_switch);
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
        uint32_t sharded_buffer_num_cores = command_ptr[DeviceCommand::sharded_buffer_num_cores_idx];
        uint32_t wrap = command_ptr[DeviceCommand::wrap_idx];

        db_cb_config_t *db_cb_config = (db_cb_config_t *)(CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset));
        const db_cb_config_t *remote_db_cb_config =
            (db_cb_config_t *)(CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset));
        if ((DeviceCommand::WrapRegion)wrap == DeviceCommand::WrapRegion::COMPLETION) {
            cq_write_interface.completion_fifo_wr_ptr = completion_queue_start_addr >> 4;     // Head to the beginning of the completion region
            cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
            notify_host_of_completion_queue_write_pointer<host_completion_queue_write_ptr_addr>();
        } else if (is_program) {
            write_and_launch_program(
                db_cb_config,
                remote_db_cb_config,
                program_transfer_start_addr,
                num_pages,
                command_ptr,
                producer_noc_encoding,
                producer_consumer_transfer_num_pages);
            wait_for_program_completion(num_workers);
        } else {
            command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_transfer_start_addr);
            write_buffers<host_completion_queue_write_ptr_addr>(
                db_cb_config,
                remote_db_cb_config,
                command_ptr,
                completion_queue_start_addr,
                num_buffer_transfers,
                sharded_buffer_num_cores,
                producer_noc_encoding,
                producer_consumer_transfer_num_pages);
        }

        if (finish) {
            notify_host_complete<host_finish_addr>();
        }

        // notify producer that it has completed a command
        noc_semaphore_inc(producer_noc_encoding | get_semaphore(0), 1);
        db_buf_switch = not db_buf_switch;
        noc_async_write_barrier(); // Barrier for now
    }
}
