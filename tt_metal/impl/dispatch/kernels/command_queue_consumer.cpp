// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/cq_prefetcher.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_dispatcher.hpp"

void kernel_main() {
    bool db_buf_switch = false;

    constexpr uint32_t host_completion_queue_write_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t completion_queue_start_addr = get_compile_time_arg_val(1);
    uint32_t completion_queue_size = get_compile_time_arg_val(2);
    constexpr uint32_t host_finish_addr = get_compile_time_arg_val(3);
    constexpr uint32_t cmd_base_address = get_compile_time_arg_val(4);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(5);

    volatile uint32_t* db_semaphore_addr = reinterpret_cast<volatile uint32_t*>(SEMAPHORE_BASE);

    uint64_t producer_noc_encoding = uint64_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y)) << 32;
    uint64_t consumer_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;

    setup_completion_queue_write_interface(completion_queue_start_addr, completion_queue_size);

    while (true) {
        DeviceZoneScopedMainN("CQ-CONSUMER-MAIN");
        {
            DeviceZoneScopedMainChildN("CQ-CONSUMER-PROD-SEM-ACQ");
            // Wait for producer to supply a command
            db_acquire(db_semaphore_addr, consumer_noc_encoding);
        }

        // For each instruction, we need to jump to the relevant part of the device command
        uint32_t command_start_addr = get_command_slot_addr<cmd_base_address, consumer_data_buffer_size>(db_buf_switch);
        uint32_t buffer_transfer_start_addr = command_start_addr + (DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER * sizeof(uint32_t));
        uint32_t program_transfer_start_addr = buffer_transfer_start_addr + ((DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION * DeviceCommand::NUM_POSSIBLE_BUFFER_TRANSFERS) * sizeof(uint32_t));

        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;
        uint32_t num_workers = header->num_workers;
        uint32_t num_buffer_transfers = header->num_buffer_transfers;
        uint32_t is_program = header->is_program_buffer;
        uint32_t page_size = header->page_size;
        uint32_t consumer_cb_size = header->consumer_cb_size;
        uint32_t consumer_cb_num_pages = header->consumer_cb_num_pages;
        uint32_t num_pages = header->num_pages;
        uint32_t producer_consumer_transfer_num_pages = header->producer_consumer_transfer_num_pages;
        bool is_sharded = (bool) (header->buffer_type == (uint32_t)DeviceCommand::BufferType::SHARDED);
        uint32_t sharded_buffer_num_cores = header->sharded_buffer_num_cores;
        uint32_t wrap = header->wrap;
        bool is_event_sync = header->is_event_sync;

        db_cb_config_t* db_cb_config = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
        db_cb_config_t* remote_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
        uint32_t completion_data_size = header->completion_data_size;
        completion_queue_reserve_back(completion_data_size);
        write_event(uint32_t(&header->event));
        if ((DeviceCommand::WrapRegion)wrap == DeviceCommand::WrapRegion::COMPLETION) {
            cq_write_interface.completion_fifo_wr_ptr = completion_queue_start_addr >> 4;     // Head to the beginning of the completion region
            cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
            notify_host_of_completion_queue_write_pointer(host_completion_queue_write_ptr_addr);
            noc_async_write_barrier(); // Barrier for now
        } else if (is_program) {
            uint32_t l1_consumer_fifo_limit = (db_cb_config->rd_ptr_16B << 4) + (db_cb_config->total_size_16B << 4);
            reset_dispatch_message_addr();
            write_and_launch_program(
                db_cb_config,
                remote_db_cb_config,
                (CommandHeader*)command_ptr,
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(program_transfer_start_addr),
                producer_noc_encoding,
                producer_consumer_transfer_num_pages,
                l1_consumer_fifo_limit);
            wait_for_program_completion(num_workers);
        } else if (is_event_sync) {
            wait_for_event(header->event_sync_event_id, header->event_sync_core_x, header->event_sync_core_y);
        } else {
            command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer_transfer_start_addr);
            write_buffers(
                db_cb_config,
                remote_db_cb_config,
                command_ptr,
                num_buffer_transfers,
                is_sharded,
                sharded_buffer_num_cores,
                producer_noc_encoding,
                producer_consumer_transfer_num_pages);
        }

        completion_queue_push_back(completion_data_size, completion_queue_start_addr, host_completion_queue_write_ptr_addr);
        record_last_completed_event(header->event);
        {
            DeviceZoneScopedN("CQ-NOTIFY_PROC");
            // notify producer that it has completed a command
            noc_semaphore_inc(producer_noc_encoding | get_semaphore(0), 1);
            db_buf_switch = not db_buf_switch;
            noc_async_write_barrier(); // Barrier for now
            noc_async_atomic_barrier();
        }
    }
}
