// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/cq_prefetcher.hpp"
#include "debug/dprint.h"

template <PullAndRelayType pull_type, PullAndRelayType push_type>
FORCE_INLINE
uint32_t program_pull_and_push_config(
    volatile tt_l1_ptr uint32_t* buffer_transfer_ptr,
    PullAndRelayCfg& src_pr_cfg,
    PullAndRelayCfg& dst_pr_cfg,
    uint32_t page_size,
    uint32_t num_pages_to_read,
    uint32_t num_pages_to_write,
    uint64_t pull_noc_encoding = 0,
    uint64_t push_noc_encoding = 0,
    uint32_t remote_cb_size = 0,
    volatile db_cb_config_t* local_multicore_cb_cfg = nullptr,
    volatile db_cb_config_t* remote_multicore_cb_cfg = nullptr) {
    /*
        Set up src_pr_cfg and dst_pr_cfg
    */
    uint32_t src_bank_base_address = buffer_transfer_ptr[0];
    uint32_t dst_bank_base_address = buffer_transfer_ptr[1];
    uint32_t num_pages = buffer_transfer_ptr[2];
    uint32_t src_buf_type = buffer_transfer_ptr[4];
    uint32_t dst_buf_type = buffer_transfer_ptr[5];
    uint32_t src_page_index = buffer_transfer_ptr[6];
    uint32_t dst_page_index = buffer_transfer_ptr[7];

    static_assert(pull_type == PullAndRelayType::CIRCULAR_BUFFER or pull_type == PullAndRelayType::BUFFER);
    static_assert(push_type == PullAndRelayType::CIRCULAR_BUFFER or push_type == PullAndRelayType::BUFFER);

    if constexpr (pull_type == PullAndRelayType::BUFFER) {
        src_pr_cfg.buff_cfg.buffer.init((BufferType)src_buf_type, src_bank_base_address, page_size);
        src_pr_cfg.buff_cfg.page_id = src_page_index;
    } else {
        src_pr_cfg.cb_buff_cfg.remote_noc_encoding = pull_noc_encoding;
        src_pr_cfg.cb_buff_cfg.local_multicore_cb_cfg = local_multicore_cb_cfg;
        src_pr_cfg.cb_buff_cfg.remote_multicore_cb_cfg = remote_multicore_cb_cfg;
        // uint32_t l1_remote_fifo_limit_16B = (get_cb_start_address<true>() + remote_cb_size) >> 4;
        // src_pr_cfg.cb_buff_cfg.fifo_limit_16B = l1_remote_fifo_limit_16B;
    }
    src_pr_cfg.num_pages_to_read = num_pages_to_read;

    if constexpr (push_type == PullAndRelayType::BUFFER) {
        dst_pr_cfg.buff_cfg.buffer.init((BufferType)dst_buf_type, dst_bank_base_address, page_size);
        dst_pr_cfg.buff_cfg.page_id = dst_page_index;
    } else { // pushing data to circular buffer
        dst_pr_cfg.cb_buff_cfg.remote_noc_encoding = push_noc_encoding;
        dst_pr_cfg.cb_buff_cfg.local_multicore_cb_cfg = local_multicore_cb_cfg;
        dst_pr_cfg.cb_buff_cfg.remote_multicore_cb_cfg = remote_multicore_cb_cfg;
        // uint32_t l1_remote_fifo_limit_16B = (get_cb_start_address<true>() + remote_cb_size) >> 4;
        // dst_pr_cfg.cb_buff_cfg.fifo_limit_16B = l1_remote_fifo_limit_16B;
    }
    dst_pr_cfg.num_pages_to_write = num_pages_to_write;
    return num_pages;
}

void kernel_main() {
    bool db_buf_switch = false;
    constexpr uint32_t host_issue_queue_read_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t issue_queue_start_addr = get_compile_time_arg_val(1);
    constexpr uint32_t issue_queue_size = get_compile_time_arg_val(2);
    constexpr uint32_t host_completion_queue_write_ptr_addr = get_compile_time_arg_val(3);
    constexpr uint32_t completion_queue_start_addr = get_compile_time_arg_val(4);
    constexpr uint32_t completion_queue_size = get_compile_time_arg_val(5);
    constexpr uint32_t host_finish_addr = get_compile_time_arg_val(6);
    constexpr uint32_t command_start_addr = get_compile_time_arg_val(7);
    constexpr uint32_t data_section_addr = get_compile_time_arg_val(8);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(9);
    constexpr uint32_t consumer_cmd_base_addr = get_compile_time_arg_val(10);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(11);
    constexpr tt::PullAndPushConfig pull_and_push_config = (tt::PullAndPushConfig)get_compile_time_arg_val(12);

    static_assert(pull_and_push_config == tt::PullAndPushConfig::LOCAL or pull_and_push_config == tt::PullAndPushConfig::PUSH_TO_REMOTE or pull_and_push_config == tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH or pull_and_push_config == tt::PullAndPushConfig::PULL_FROM_REMOTE);

    constexpr bool read_from_issue_queue = (pull_and_push_config == tt::PullAndPushConfig::LOCAL or pull_and_push_config == tt::PullAndPushConfig::PUSH_TO_REMOTE);

    setup_issue_queue_read_interface(issue_queue_start_addr, issue_queue_size);
    setup_completion_queue_write_interface(completion_queue_start_addr, completion_queue_size);

    // Initialize the producer/consumer DB semaphore
    // This represents how many buffers the producer can write to.
    // At the beginning, it can write to two different buffers.
    uint64_t my_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;
    uint64_t pull_noc_encoding = uint64_t(NOC_XY_ENCODING(PULL_NOC_X, PULL_NOC_Y)) << 32;
    uint64_t push_noc_encoding = uint64_t(NOC_XY_ENCODING(PUSH_NOC_X, PUSH_NOC_Y)) << 32;
    uint64_t dispatch_noc_encoding = uint64_t(NOC_XY_ENCODING(DISPATCH_NOC_X, DISPATCH_NOC_Y)) << 32;
    uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32;

    volatile tt_l1_ptr uint32_t* push_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));   // Should be initialized to num commands slots on router core by host
    volatile tt_l1_ptr uint32_t *pull_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));   // Should be initialized to 0 by host
    volatile tt_l1_ptr uint32_t *dispatch_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));   // Should be initialized to num commands slots on dispatch core by host

    // DPRINT << "SEM INIT: " << push_semaphore_addr[0] << ENDL();
    volatile db_cb_config_t* local_multicore_cb_cfg = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
    volatile db_cb_config_t* dispatch_multicore_cb_cfg = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);

    constexpr uint32_t dispatch_cb_num_pages = (MEM_L1_SIZE - L1_UNRESERVED_BASE - DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND * 2) / DeviceCommand::PROGRAM_PAGE_SIZE;
    constexpr uint32_t dispatch_cb_size = dispatch_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

    // TODO: Need to support this for REMOTE_PULL_AND_PUSH to allow sending programs to dispatch core
    if constexpr (pull_and_push_config == tt::PullAndPushConfig::LOCAL) {
        // DPRINT << "Programming cb for programs on dispatch core" << ENDL();
        // Dispatch core has a constant CB
        program_remote_sync_cb<false>(
            local_multicore_cb_cfg,
            dispatch_multicore_cb_cfg,
            dispatch_noc_encoding,
            dispatch_cb_num_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            dispatch_cb_size);
    }

    // Set up dispatch core CB
    ProgramEventBuffer program_event_buffer(EVENT_PTR, completion_queue_start_addr, host_completion_queue_write_ptr_addr);
    PullAndRelayCfg src_pr_cfg(program_event_buffer);
    PullAndRelayCfg dst_pr_cfg(program_event_buffer);
    dst_pr_cfg.dispatch_synchronization_semaphore = dispatch_semaphore_addr;


    DPRINT << "MY_X: " << (uint32_t) my_x[0] << ", MY_Y: " << (uint32_t) my_y[0] << ENDL();
    while (true) {
        if constexpr (read_from_issue_queue) {
            // we will also need to poll the program event buffer
            DPRINT << "WAIT CMD" << ENDL();
            while (not issue_queue_space_available()) {
                if constexpr (pull_and_push_config == tt::PullAndPushConfig::LOCAL) {
                    if (consumer_is_idle<2>(dispatch_semaphore_addr)) {
                        program_event_buffer.write_events(); // write number of events in program event buffer
                    }
                }
            }
            DPRINT << "GOT CMD" << ENDL();

            uint32_t rd_ptr = (cq_read_interface.issue_fifo_rd_ptr << 4);
            uint64_t src_noc_addr = pcie_core_noc_encoding | rd_ptr;
            noc_async_read(src_noc_addr, command_start_addr, min(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, issue_queue_size - rd_ptr));
        } else { // need to split this up for "remote_pull_and_push" to poll the program event buffer
            DPRINT << "WAIT FOR DST TO GET CMD" << ENDL();
            db_acquire(pull_semaphore_addr, my_noc_encoding); // dst routers increments this semaphore when cmd is available in the dst router
            DPRINT << "DONE WAIT FOR DST TO GET CMD" << ENDL();
            uint64_t src_noc_addr = pull_noc_encoding | eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
            noc_async_read(src_noc_addr, command_start_addr, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND); // read in cmd header only
        }
        noc_async_read_barrier();

        // Producer information
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;

        uint32_t issue_data_size = header->issue_data_size;
        uint32_t num_buffer_transfers = header->num_buffer_transfers;
        uint32_t stall = header->stall;
        uint32_t page_size = header->page_size;
        uint32_t producer_cb_size = header->producer_cb_size;
        uint32_t consumer_cb_size = header->consumer_cb_size;
        uint32_t producer_cb_num_pages = header->producer_cb_num_pages;
        uint32_t consumer_cb_num_pages = header->consumer_cb_num_pages;
        uint32_t producer_consumer_transfer_num_pages = header->producer_consumer_transfer_num_pages;
        bool is_program = header->is_program_buffer;
        uint32_t completion_data_size = header->completion_data_size;
        DeviceCommand::WrapRegion wrap = (DeviceCommand::WrapRegion)header->wrap;
        uint32_t event = header->event;

        if constexpr (read_from_issue_queue) { // don't wrap issue queue on completion path
            if (wrap == DeviceCommand::WrapRegion::ISSUE) {
                // Basically popfront without the extra conditional
                cq_read_interface.issue_fifo_rd_ptr = cq_read_interface.issue_fifo_limit - cq_read_interface.issue_fifo_size;  // Head to beginning of command queue
                cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
                notify_host_of_issue_queue_read_pointer<host_issue_queue_read_ptr_addr>();
                continue; // issue wraps are not relayed forward
            }
        }

        // DPRINT << "PROGRAM LOCAL CB" << ENDL();
        program_local_cb(data_section_addr, producer_cb_num_pages, page_size, producer_cb_size);

        if constexpr (pull_and_push_config == tt::PullAndPushConfig::LOCAL) {
            wait_consumer_space_available(dispatch_semaphore_addr);

            if (is_program) {
                relay_command<command_start_addr, consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, dispatch_noc_encoding);
                db_buf_switch = not db_buf_switch;
            }
            if (stall) {
                wait_consumer_idle<2>(dispatch_semaphore_addr);
                program_event_buffer.write_events();
            }

            volatile tt_l1_ptr uint32_t* buffer_transfer_ptr = command_ptr + DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
            uint32_t num_pages_to_read = dispatch_cb_num_pages / 2;
            uint32_t num_pages_to_write = producer_consumer_transfer_num_pages;

            if (is_program) {
                program_event_buffer.push_event(event);
                update_producer_consumer_sync_semaphores(my_noc_encoding, dispatch_noc_encoding, dispatch_semaphore_addr, get_semaphore(0));
                for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                    uint32_t num_pages_in_transfer = program_pull_and_push_config<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(
                        buffer_transfer_ptr,
                        src_pr_cfg,
                        dst_pr_cfg,
                        page_size,
                        num_pages_to_read,
                        num_pages_to_write,
                        pull_noc_encoding,
                        dispatch_noc_encoding,
                        dispatch_cb_size,
                        local_multicore_cb_cfg,
                        dispatch_multicore_cb_cfg
                    );
                    pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages_in_transfer);
                    buffer_transfer_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
                    program_local_cb(data_section_addr, producer_cb_num_pages, page_size, producer_cb_size);
                }
            } else {
                completion_queue_reserve_back(completion_data_size);
                for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                    uint32_t num_pages_in_transfer = program_pull_and_push_config<PullAndRelayType::BUFFER, PullAndRelayType::BUFFER>(
                        buffer_transfer_ptr,
                        src_pr_cfg,
                        dst_pr_cfg,
                        page_size,
                        num_pages_to_read,
                        num_pages_to_write
                    );
                    pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages_in_transfer);
                    buffer_transfer_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
                }

                write_event((uint32_t)&header->event);

                if (wrap == DeviceCommand::WrapRegion::COMPLETION) {
                    cq_write_interface.completion_fifo_wr_ptr = completion_queue_start_addr >> 4;     // Head to the beginning of the completion region
                    cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
                    notify_host_of_completion_queue_write_pointer(host_completion_queue_write_ptr_addr);
                    noc_async_write_barrier(); // Barrier for now
                }

                completion_queue_push_back(completion_data_size, completion_queue_start_addr, host_completion_queue_write_ptr_addr);
            }

            issue_queue_pop_front<host_issue_queue_read_ptr_addr>(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + issue_data_size);

        } else if constexpr (pull_and_push_config == tt::PullAndPushConfig::PUSH_TO_REMOTE) {
            wait_consumer_space_available(push_semaphore_addr); // ensure SRC router has space to push command

            // Send command to SRC router
            relay_command<command_start_addr, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, consumer_data_buffer_size>(db_buf_switch, push_noc_encoding);
            update_producer_consumer_sync_semaphores(my_noc_encoding, push_noc_encoding, push_semaphore_addr, eth_get_semaphore(0)); // notify SRC router that command was pushed

            // Need to support program transfers from sysmem
            if (num_buffer_transfers == 1) {
                volatile tt_l1_ptr uint32_t* buffer_transfer_ptr = command_ptr + DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
                uint32_t src_buf_type = buffer_transfer_ptr[4];
                uint32_t dst_buf_type = buffer_transfer_ptr[5];

                if ((BufferType)src_buf_type == BufferType::SYSTEM_MEMORY) {
                    volatile db_cb_config_t* remote_multicore_cb_cfg = get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, db_buf_switch);

                    program_remote_sync_cb<true>(
                        local_multicore_cb_cfg,
                        remote_multicore_cb_cfg,
                        push_noc_encoding,
                        consumer_cb_num_pages,
                        page_size,
                        consumer_cb_size
                    );

                    uint32_t num_pages_in_transfer = program_pull_and_push_config<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(
                        buffer_transfer_ptr,
                        src_pr_cfg,
                        dst_pr_cfg,
                        page_size,
                        (producer_cb_num_pages / 2),
                        producer_consumer_transfer_num_pages,
                        pull_noc_encoding,
                        push_noc_encoding,
                        consumer_cb_size,
                        local_multicore_cb_cfg,
                        remote_multicore_cb_cfg
                    );

                    // Send data to SRC router
                    pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages_in_transfer);
                }
            }
            DPRINT << "DONE sending cmd and data to src router" << ENDL();

            issue_queue_pop_front<host_issue_queue_read_ptr_addr>(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + issue_data_size);

        } else if constexpr (pull_and_push_config == tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH) {
            wait_consumer_space_available(push_semaphore_addr); // ensure SRC router has space to push command

            volatile db_cb_config_t* remote_multicore_cb_cfg = get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, db_buf_switch);

            // Program the CB on the DST router because we may need to pull in data from DST
            program_remote_sync_cb<true>(
                local_multicore_cb_cfg,
                remote_multicore_cb_cfg,
                pull_noc_encoding,
                consumer_cb_num_pages,
                page_size,
                consumer_cb_size
            );

            // Signal to DST router that command has been read in
            noc_semaphore_inc(pull_noc_encoding | eth_get_semaphore(0), 1);
            noc_async_write_barrier(); // Barrier for now

            header->fwd_path = 0;

            // TODO: Need to handle programs
            if (num_buffer_transfers == 1) {
                volatile tt_l1_ptr uint32_t* buffer_transfer_ptr = command_ptr + DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
                uint32_t src_buf_type = buffer_transfer_ptr[4];
                uint32_t dst_buf_type = buffer_transfer_ptr[5];

                if ((BufferType)src_buf_type == BufferType::SYSTEM_MEMORY) {
                    // Writing data to buffer on R chip after getting data from DST router
                    volatile db_cb_config_t* remote_multicore_cb_cfg = get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, db_buf_switch);

                    // remote pull and relay
                    // doing a write so we pull from eth router cb and write to buffer
                    DPRINT << "Read from eth cb and write to dst buffer - pull sem is " << pull_semaphore_addr[0] << ENDL();

                    uint32_t num_pages_in_transfer = program_pull_and_push_config<PullAndRelayType::CIRCULAR_BUFFER, PullAndRelayType::BUFFER>(
                        buffer_transfer_ptr,
                        src_pr_cfg,
                        dst_pr_cfg,
                        page_size,
                        producer_consumer_transfer_num_pages,
                        producer_consumer_transfer_num_pages,
                        pull_noc_encoding,
                        push_noc_encoding,
                        consumer_cb_size,
                        local_multicore_cb_cfg,
                        remote_multicore_cb_cfg
                    );

                    pull_and_relay<PullAndRelayType::CIRCULAR_BUFFER, PullAndRelayType::BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages_in_transfer); // write all the data

                    DPRINT << "DONE WRITE R BUFFER" << ENDL();

                    // Doing doing the write now send the write buffer command back to the src router without any data
                    header->num_buffer_transfers = 0; // make sure src router doesn't expect any data incoming
                    relay_command<command_start_addr, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, consumer_data_buffer_size>(db_buf_switch, push_noc_encoding);
                    update_producer_consumer_sync_semaphores(my_noc_encoding, push_noc_encoding, push_semaphore_addr, eth_get_semaphore(0));

                } else if ((BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY) {
                    // Reading data from buffer on R chip and sending command + buffer data back to L chip
                    volatile db_cb_config_t* remote_multicore_cb_cfg = get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, db_buf_switch);

                    DPRINT << "Programming remote CB" << ENDL();
                    program_remote_sync_cb<true>(
                        local_multicore_cb_cfg,
                        remote_multicore_cb_cfg,
                        push_noc_encoding,
                        consumer_cb_num_pages,
                        page_size,
                        consumer_cb_size
                    );

                    uint32_t num_pages_in_transfer = program_pull_and_push_config<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(
                        buffer_transfer_ptr,
                        src_pr_cfg,
                        dst_pr_cfg,
                        page_size,
                        (producer_cb_num_pages / 2),
                        producer_consumer_transfer_num_pages,
                        pull_noc_encoding,
                        push_noc_encoding,
                        consumer_cb_size,
                        local_multicore_cb_cfg,
                        remote_multicore_cb_cfg
                    );

                    DPRINT << "Read from src buffer and write to cb" << ENDL();

                    relay_command<command_start_addr, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, consumer_data_buffer_size>(db_buf_switch, push_noc_encoding);
                    update_producer_consumer_sync_semaphores(my_noc_encoding, push_noc_encoding, push_semaphore_addr, eth_get_semaphore(0));
                    pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages_in_transfer);
                }
            } else {
                // Send command to src router
                relay_command<command_start_addr, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, consumer_data_buffer_size>(db_buf_switch, push_noc_encoding);
                update_producer_consumer_sync_semaphores(my_noc_encoding, push_noc_encoding, push_semaphore_addr, eth_get_semaphore(0));
            }


        } else if constexpr (pull_and_push_config == tt::PullAndPushConfig::PULL_FROM_REMOTE) {
            // Reads data from DST router and writes to completion queue
            completion_queue_reserve_back(completion_data_size);
            write_event(uint32_t(&header->event));

            if (wrap == DeviceCommand::WrapRegion::COMPLETION) {
                cq_write_interface.completion_fifo_wr_ptr = completion_queue_start_addr >> 4;     // Head to the beginning of the completion region
                cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
                notify_host_of_completion_queue_write_pointer(host_completion_queue_write_ptr_addr);
                noc_async_write_barrier(); // Barrier for now
            }

            volatile db_cb_config_t* remote_multicore_cb_cfg = get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, db_buf_switch);

            // Program the CB on the DST router because we may need to pull in data from DST
            program_remote_sync_cb<true>(
                local_multicore_cb_cfg,
                remote_multicore_cb_cfg,
                pull_noc_encoding,
                consumer_cb_num_pages,
                page_size,
                consumer_cb_size
            );

            // Signal to DST router that command has been read in
            noc_semaphore_inc(pull_noc_encoding | eth_get_semaphore(0), 1);
            noc_async_write_barrier(); // Barrier for now

            // Read data from DST router
            if (num_buffer_transfers == 1) {
                volatile tt_l1_ptr uint32_t* buffer_transfer_ptr = command_ptr + DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

                uint32_t num_pages_in_transfer = program_pull_and_push_config<PullAndRelayType::CIRCULAR_BUFFER, PullAndRelayType::BUFFER>(
                    buffer_transfer_ptr,
                    src_pr_cfg,
                    dst_pr_cfg,
                    page_size,
                    producer_consumer_transfer_num_pages,
                    producer_consumer_transfer_num_pages,
                    pull_noc_encoding,
                    push_noc_encoding,
                    consumer_cb_size,
                    local_multicore_cb_cfg,
                    remote_multicore_cb_cfg
                );

                pull_and_relay<PullAndRelayType::CIRCULAR_BUFFER, PullAndRelayType::BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages_in_transfer); // write data to sysmem buffer
            }

            completion_queue_push_back(completion_data_size, completion_queue_start_addr, host_completion_queue_write_ptr_addr);
            continue;
        }
    }
}
