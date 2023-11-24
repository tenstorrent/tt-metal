// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

static constexpr uint32_t PROGRAM_CB_ID = 0;

FORCE_INLINE
void multicore_cb_wait_front(bool db_buf_switch, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint32_t pages_acked = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_ack_addr(db_buf_switch));
    volatile tt_l1_ptr uint32_t* pages_received_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_recv_addr(db_buf_switch));

    uint16_t pages_received;
    do {
        pages_received = uint16_t(*pages_received_ptr) - pages_acked;
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'R', 'B', 'D');
}

void multicore_cb_pop_front(
    uint64_t producer_noc_encoding,
    bool db_buf_switch,
    uint32_t fifo_limit,
    uint32_t fifo_size,
    uint32_t num_pages,
    uint32_t page_size) {
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_ACK_PTR = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_ack_addr(db_buf_switch));
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_READ_PTR =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_rd_ptr_addr(db_buf_switch));

    *CQ_CONSUMER_CB_ACK_PTR += num_pages;
    *CQ_CONSUMER_CB_READ_PTR += (page_size * num_pages) >> 4;

    if ((*CQ_CONSUMER_CB_READ_PTR << 4) > fifo_limit) {
        *CQ_CONSUMER_CB_READ_PTR -= fifo_size >> 4;
    }

    uint32_t pages_ack_addr = get_db_cb_ack_addr(db_buf_switch);
    noc_semaphore_set_remote(uint32_t(CQ_CONSUMER_CB_ACK_PTR), producer_noc_encoding | pages_ack_addr);
}

FORCE_INLINE
uint32_t get_read_ptr(bool db_buf_switch) {
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_rd_ptr_addr(db_buf_switch)) << 4;
}

//inline uint32_t min(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

FORCE_INLINE void write_buffers(
    volatile tt_l1_ptr uint32_t* command_ptr,
    uint32_t num_destinations,
    uint32_t num_cores,
    uint32_t consumer_cb_size,
    uint32_t consumer_cb_num_pages,
    uint64_t producer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages,
    bool db_buf_switch) {

    bool sharded = num_cores > 1;
    for (uint32_t i = 0; i < num_destinations; i++) {
        const uint32_t bank_base_address = command_ptr[1];
        const uint32_t num_pages = command_ptr[2];
        const uint32_t page_size = command_ptr[3];
        const uint32_t dst_buf_type = command_ptr[5];

        uint32_t num_to_write;
        uint32_t src_addr = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_rd_ptr_addr(db_buf_switch)) << 4;
        uint32_t l1_consumer_fifo_limit = src_addr + consumer_cb_size - 1;

        for (uint32_t id = 0; id < num_pages;) {
            num_to_write = min(num_pages - id, producer_consumer_transfer_num_pages);
            multicore_cb_wait_front(db_buf_switch, num_to_write);
            uint32_t src_addr = get_read_ptr(db_buf_switch);
            if((BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY || !sharded){
                Buffer buffer((BufferType)dst_buf_type, bank_base_address, page_size);
                buffer.noc_async_write_buffer(src_addr, id, num_to_write, 0);
            }
            else{
                ShardedBuffer src_buffer(page_size, num_cores, bank_base_address, command_ptr+6);
                src_buffer.noc_async_write_buffer(src_addr, id, num_to_write);
            }
            noc_async_write_barrier();
            multicore_cb_pop_front(
                producer_noc_encoding,
                db_buf_switch,
                l1_consumer_fifo_limit,
                consumer_cb_size,
                num_to_write,
                page_size);
            noc_async_write_barrier();
            id += num_to_write;
        }
    }
}

template <bool multicast>
FORCE_INLINE void write_program_page(uint32_t page_addr, volatile tt_l1_ptr uint32_t*& command_ptr, bool last_page) {
    uint32_t num_transfers = command_ptr[0];
    command_ptr++;
    uint32_t src = page_addr;

    for (uint32_t i = 0; i < num_transfers; i++) {
        uint32_t num_bytes = command_ptr[0];
        uint32_t dst = command_ptr[1];
        uint32_t dst_noc = command_ptr[2];
        uint32_t num_recv = command_ptr[3];
        bool last_transfer_in_group = command_ptr[4];
        bool linked = (not (last_page & last_transfer_in_group)) & command_ptr[5];

        uint64_t dst_noc_addr = (uint64_t(dst_noc) << 32) | dst;

        if constexpr (multicast) {
            noc_async_write_multicast(src, dst_noc_addr, num_bytes, num_recv, linked);
        } else {
            noc_async_write_one_packet(src, dst_noc_addr, num_bytes);
        }

        command_ptr += 6;
        if (last_transfer_in_group) {
            src = align(src + num_bytes, 16);
        }
    }
}

template <bool multicast>
FORCE_INLINE void program_page_transfer(
    volatile tt_l1_ptr uint32_t*& command_ptr,
    uint64_t producer_noc_encoding,
    uint32_t consumer_cb_size,
    uint32_t consumer_cb_num_pages,
    uint32_t producer_consumer_transfer_num_pages,
    bool db_buf_switch,
    uint32_t num_pages_in_transfer) {

    uint32_t l1_consumer_fifo_limit = get_read_ptr(db_buf_switch) + consumer_cb_size - 1;
    for (uint32_t page_idx = 0; page_idx < num_pages_in_transfer;) {
        uint32_t num_to_write = min(num_pages_in_transfer - page_idx, producer_consumer_transfer_num_pages);
        multicore_cb_wait_front(db_buf_switch, num_to_write);
        uint32_t src_addr = get_read_ptr(db_buf_switch);
        for (uint32_t i = 0; i < num_to_write; i++) {
            write_program_page<multicast>(src_addr, command_ptr, i == num_to_write - 1);
            src_addr += DeviceCommand::PROGRAM_PAGE_SIZE;
        }
        page_idx += num_to_write;
        noc_async_write_barrier();
        multicore_cb_pop_front(
            producer_noc_encoding,
            db_buf_switch,
            l1_consumer_fifo_limit,
            consumer_cb_size,
            num_to_write,
            DeviceCommand::PROGRAM_PAGE_SIZE);
        noc_async_write_barrier();  // Flush barrier, not an ack barrier
    }
}

FORCE_INLINE
void write_and_launch_program(
    uint32_t program_transfer_start_addr,
    uint32_t num_pages,
    volatile tt_l1_ptr uint32_t*& command_ptr,
    uint64_t producer_noc_encoding,
    uint32_t consumer_cb_size,
    uint32_t consumer_cb_num_pages,
    uint32_t producer_consumer_transfer_num_pages,
    bool db_buf_switch) {

    if (not num_pages) {
        return;
    }

    // GO signals are just data within pages, so we need to set
    // our local 'recv' address value to 0 before we initiate
    // any transfers
    volatile tt_l1_ptr uint32_t* message_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(DISPATCH_MESSAGE_ADDR);
    *message_addr_ptr = 0;

    volatile tt_l1_ptr uint32_t* command_ptr_fixed = command_ptr;
    command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(program_transfer_start_addr);
    for (uint32_t transfer_type_idx = 0; transfer_type_idx < (uint32_t) DeviceCommand::TransferType::NUM_TRANSFER_TYPES; transfer_type_idx++) {
        uint32_t num_pages_in_transfer;
        bool multicast = true;
        switch (transfer_type_idx) {
            DeviceCommand::TransferType transfer_type;
            case (uint32_t) DeviceCommand::TransferType::RUNTIME_ARGS:
                multicast = false;
                num_pages_in_transfer = command_ptr_fixed[DeviceCommand::num_runtime_arg_pages_idx];
                break;
            case (uint32_t) DeviceCommand::TransferType::CB_CONFIGS:
                num_pages_in_transfer = command_ptr_fixed[DeviceCommand::num_cb_config_pages_idx];
                break;
            case (uint32_t) DeviceCommand::TransferType::PROGRAM_PAGES:
                num_pages_in_transfer = command_ptr_fixed[DeviceCommand::num_program_pages_idx];
                break;
            case (uint32_t) DeviceCommand::TransferType::GO_SIGNALS:
                num_pages_in_transfer = command_ptr_fixed[DeviceCommand::num_go_signal_pages_idx];
                break;
        }

        if (multicast) {
            program_page_transfer<true>(command_ptr, producer_noc_encoding, consumer_cb_size, consumer_cb_num_pages, producer_consumer_transfer_num_pages, db_buf_switch, num_pages_in_transfer);
        } else {
            program_page_transfer<false>(command_ptr, producer_noc_encoding, consumer_cb_size, consumer_cb_num_pages, producer_consumer_transfer_num_pages, db_buf_switch, num_pages_in_transfer);
        }
    }
}

FORCE_INLINE void wait_for_program_completion(
    uint32_t num_workers, uint32_t tensix_soft_reset_addr) {
    if (not num_workers)
        return;

    // Wait on worker cores to notify me that they have completed
    DEBUG_STATUS('Q', 'W');

    volatile tt_l1_ptr uint32_t* message_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(DISPATCH_MESSAGE_ADDR);

    while (*message_addr_ptr != num_workers)
        ;

    DEBUG_STATUS('Q', 'D');
}

FORCE_INLINE void notify_host_complete() {
    volatile tt_l1_ptr uint32_t* finish_ptr = get_cq_finish_ptr();
    finish_ptr[0] = 1;
    constexpr static uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32;
    uint64_t finish_noc_addr = pcie_core_noc_encoding | HOST_CQ_FINISH_PTR;
    noc_async_write(uint32_t(finish_ptr), finish_noc_addr, 4);
    noc_async_write_barrier();
    finish_ptr[0] = 0;
}
