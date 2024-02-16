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
    constexpr bool db_buf_switch = false; // DST router writes to first slot

    constexpr uint32_t host_completion_queue_write_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t completion_queue_start_addr = get_compile_time_arg_val(1);
    uint32_t completion_queue_size = get_compile_time_arg_val(2);
    constexpr uint32_t host_finish_addr = get_compile_time_arg_val(3);
    constexpr uint32_t cmd_base_address = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(5);

    volatile tt_l1_ptr uint32_t* db_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));

    uint32_t producer_noc_encoding = uint32_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y));
    uint32_t consumer_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));

    setup_completion_queue_write_interface(completion_queue_start_addr, completion_queue_size);

    while (true) {
        // Wait for eth producer to supply a command
        db_acquire(db_semaphore_addr,  ((uint64_t)consumer_noc_encoding << 32));

        // For each instruction, we need to jump to the relevant part of the device command
        uint32_t command_start_addr = get_command_slot_addr<cmd_base_address, consumer_data_buffer_size>(db_buf_switch);

        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;
        uint32_t buffer_transfer_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t));
        volatile tt_l1_ptr uint32_t *buffer_transfer_command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_transfer_start_addr);
        uint32_t is_program = header->is_program_buffer;
        uint32_t num_pages = header->num_pages;
        const uint32_t dst_buf_type = buffer_transfer_command_ptr[5];
        bool reading_buffer = (!is_program) & (num_pages > 0 & (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY);

        uint32_t completion_data_size = header->completion_data_size;
        completion_queue_reserve_back(completion_data_size);
        write_event(uint32_t(&header->event));

        uint32_t wrap = header->wrap;
        if ((DeviceCommand::WrapRegion)wrap == DeviceCommand::WrapRegion::COMPLETION) {
            cq_write_interface.completion_fifo_wr_ptr = completion_queue_start_addr >> 4;     // Head to the beginning of the completion region
            cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
            notify_host_of_completion_queue_write_pointer<host_completion_queue_write_ptr_addr>();
            noc_async_write_barrier(); // Barrier for now
        } else if (reading_buffer) {
            uint32_t buffer_transfer_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t));
            volatile tt_l1_ptr uint32_t *buffer_transfer_command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_transfer_start_addr);
            db_cb_config_t* db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, true);
            const db_cb_config_t* eth_db_cb_config = get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, false);
            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            bool is_sharded = (bool) (header->buffer_type == (uint32_t)DeviceCommand::BufferType::SHARDED);
            uint32_t sharded_buffer_num_cores = header->sharded_buffer_num_cores;
            uint32_t producer_consumer_transfer_num_pages = header->consumer_router_transfer_num_pages;
            write_buffers(
                db_cb_config,
                eth_db_cb_config,
                buffer_transfer_command_ptr,
                num_buffer_transfers,
                is_sharded,
                sharded_buffer_num_cores,
                (uint64_t(producer_noc_encoding) << 32),
                producer_consumer_transfer_num_pages);
        }

        uint32_t finish = header->finish;
        if (finish) {
            notify_host_complete<host_finish_addr>();
        }

        completion_queue_push_back<completion_queue_start_addr, host_completion_queue_write_ptr_addr>(completion_data_size);

        // notify producer that it has completed a command
        noc_semaphore_inc((uint64_t(producer_noc_encoding) << 32) | eth_get_semaphore(0), 1);
        noc_async_write_barrier(); // Barrier for now
    }
}
