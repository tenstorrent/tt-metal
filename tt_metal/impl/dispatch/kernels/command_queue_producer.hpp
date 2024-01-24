// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "risc_attribs.h"

CQReadInterface cq_read_interface;

inline __attribute__((always_inline)) volatile uint32_t* get_cq_issue_read_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_ISSUE_READ_PTR);
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_issue_write_ptr() {
    return reinterpret_cast<volatile uint32_t*>(CQ_ISSUE_WRITE_PTR);
}

// Only the read interface is set up on the device... the write interface
// belongs to host
FORCE_INLINE
void setup_issue_queue_read_interface(const uint32_t issue_region_rd_ptr, const uint32_t issue_region_size) {
    cq_read_interface.issue_fifo_rd_ptr = issue_region_rd_ptr >> 4;
    cq_read_interface.issue_fifo_size = issue_region_size >> 4;
    cq_read_interface.issue_fifo_limit = (issue_region_rd_ptr + issue_region_size) >> 4;
    cq_read_interface.issue_fifo_rd_toggle = 0;
}

FORCE_INLINE
void issue_queue_wait_front() {
    DEBUG_STATUS('N', 'Q', 'W');
    uint32_t issue_write_ptr_and_toggle;
    uint32_t issue_write_ptr;
    uint32_t issue_write_toggle;
    do {
        issue_write_ptr_and_toggle = *get_cq_issue_write_ptr();
        issue_write_ptr = issue_write_ptr_and_toggle & 0x7fffffff;
        issue_write_toggle = issue_write_ptr_and_toggle >> 31;
    } while (cq_read_interface.issue_fifo_rd_ptr == issue_write_ptr and cq_read_interface.issue_fifo_rd_toggle == issue_write_toggle);
    DEBUG_STATUS('N', 'Q', 'D');
}

template <uint32_t host_issue_queue_read_ptr_addr>
FORCE_INLINE
void notify_host_of_issue_queue_read_pointer() {
    // These are the PCIE core coordinates
    constexpr static uint64_t pcie_address = (uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32) | host_issue_queue_read_ptr_addr;
    uint32_t issue_rd_ptr_and_toggle = cq_read_interface.issue_fifo_rd_ptr | (cq_read_interface.issue_fifo_rd_toggle << 31);
    volatile tt_l1_ptr uint32_t* issue_rd_ptr_addr = get_cq_issue_read_ptr();
    issue_rd_ptr_addr[0] = issue_rd_ptr_and_toggle;
    noc_async_write(CQ_ISSUE_READ_PTR, pcie_address, 4);
    noc_async_write_barrier();
}

template <uint32_t host_issue_queue_read_ptr_addr>
FORCE_INLINE
void issue_queue_pop_front(uint32_t cmd_size_B) {
    // First part of equation aligns to nearest multiple of 32, and then we shift to make it a 16B addr. Both
    // host and device are consistent in updating their pointers in this way, so they won't get out of sync. The
    // alignment is necessary because we can only read/write from/to 32B aligned addrs in host<->dev communication.
    uint32_t cmd_size_16B = align(cmd_size_B, 32) >> 4;
    cq_read_interface.issue_fifo_rd_ptr += cmd_size_16B;

    if (cq_read_interface.issue_fifo_rd_ptr >= cq_read_interface.issue_fifo_limit) {
        cq_read_interface.issue_fifo_rd_ptr -= cq_read_interface.issue_fifo_size;
        cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
    }

    notify_host_of_issue_queue_read_pointer<host_issue_queue_read_ptr_addr>();
}

FORCE_INLINE
void program_local_cb(uint32_t data_section_addr, uint32_t num_pages, uint32_t page_size, uint32_t cb_size) {
    uint32_t cb_id = 0;
    uint32_t fifo_addr = data_section_addr >> 4;
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

template <uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
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
    uint32_t cb_start_addr = get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch);
    reinterpret_cast<volatile uint32_t*>(rd_ptr_addr)[0] = cb_start_addr >> 4;
    reinterpret_cast<volatile uint32_t*>(wr_ptr_addr)[0] = cb_start_addr >> 4;

    uint32_t cb_base = get_db_cb_l1_base(db_buf_switch);
    noc_async_write(cb_base, consumer_noc_encoding | cb_base, 7 * 16);
    noc_async_write_barrier();  // barrier for now
}

FORCE_INLINE
bool cb_producer_space_available(int32_t num_pages) {
    uint32_t operand = 0;
    uint32_t pages_acked_ptr = (uint32_t) get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    DEBUG_STATUS('C', 'R', 'B', 'W');

    // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
    // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
    uint16_t pages_acked = (uint16_t)reg_read(pages_acked_ptr);
    uint16_t free_space_pages_wrap =
        cb_interface[operand].fifo_num_pages - (pages_received - pages_acked);
    free_space_pages = (int32_t)free_space_pages_wrap;
    return free_space_pages >= num_pages;
}

//FORCE_INLINE
//uint32_t min(uint32_t a, uint32_t b) { return (a < b) ? a: b; }

FORCE_INLINE
bool cb_consumer_space_available(bool db_buf_switch, int32_t num_pages) {

    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_acked = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_ack_addr(db_buf_switch));
    uint16_t pages_recv = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_recv_addr(db_buf_switch));
    uint32_t num_pages_consumer = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_num_pages_addr(db_buf_switch));

    uint16_t free_space_pages_wrap = num_pages_consumer - (pages_recv - pages_acked);
    int32_t free_space_pages = (int32_t)free_space_pages_wrap;
    DEBUG_STATUS('C', 'R', 'B', 'D');

    return free_space_pages >= num_pages;
}

FORCE_INLINE
void multicore_cb_push_back(uint64_t consumer_noc_encoding, uint32_t consumer_fifo_limit, uint32_t consumer_fifo_size, bool db_buf_switch, uint32_t page_size, uint32_t num_to_write) {
    // TODO(agrebenisan): Should create a multi-core CB interface... struct in L1
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_RECV_PTR = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_recv_addr(db_buf_switch));
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_WRITE_PTR = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_wr_ptr_addr(db_buf_switch));

    *CQ_CONSUMER_CB_RECV_PTR += num_to_write;
    *CQ_CONSUMER_CB_WRITE_PTR += (page_size * num_to_write) >> 4;

    if ((*CQ_CONSUMER_CB_WRITE_PTR << 4) >= consumer_fifo_limit) {
        *CQ_CONSUMER_CB_WRITE_PTR -= consumer_fifo_size >> 4;
    }

    uint32_t pages_recv_addr = get_db_cb_recv_addr(db_buf_switch);
    noc_semaphore_set_remote(uint32_t(CQ_CONSUMER_CB_RECV_PTR), consumer_noc_encoding | pages_recv_addr);
}

template <uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
FORCE_INLINE
void relay_command(bool db_buf_switch, uint64_t consumer_noc_encoding) {
    /*
        Relays the current command to the consumer.
    */

    uint64_t consumer_command_slot_addr = consumer_noc_encoding | get_command_slot_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch);
    noc_async_write(L1_UNRESERVED_BASE, consumer_command_slot_addr, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
    noc_async_write_barrier();
}

template <uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size>
void produce(
    volatile tt_l1_ptr uint32_t* command_ptr, uint32_t num_srcs, uint32_t sharded_buffer_num_cores, uint32_t page_size, uint32_t producer_cb_size, uint32_t producer_cb_num_pages,
    uint32_t consumer_cb_size, uint32_t consumer_cb_num_pages, uint64_t consumer_noc_encoding, uint32_t producer_consumer_transfer_num_pages, bool db_buf_switch) {
    /*
        This API prefetches data from host memory and writes data to the consumer core. On the consumer,
        we partition the data space into 2 via double-buffering. There are two command slots, and two
        data slots. The producer reads in data into its local buffer and checks whether it can write to
        the consumer. It continues like this in a loop, context switching between pulling in data and
        writing to the consumer.
    */

    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
    uint32_t l1_consumer_fifo_limit = get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + consumer_cb_size;

    bool sharded = sharded_buffer_num_cores > 1;

    for (uint32_t i = 0; i < num_srcs; i++) {
        const uint32_t bank_base_address = command_ptr[0];
        const uint32_t num_pages = command_ptr[2];
        const uint32_t page_size = command_ptr[3];
        const uint32_t src_buf_type = command_ptr[4];
        const uint32_t src_page_index = command_ptr[6];


        uint32_t fraction_of_producer_cb_num_pages = consumer_cb_num_pages / 2;

        uint32_t num_to_read = min(num_pages, fraction_of_producer_cb_num_pages);
        uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages); // This must be a bigger number for perf.
        uint32_t num_reads_issued = 0;
        uint32_t num_reads_completed = 0;
        uint32_t num_writes_completed = 0;
        uint32_t src_page_id = src_page_index;

        Buffer buffer;
        if ((BufferType)src_buf_type == BufferType::SYSTEM_MEMORY or not(sharded)) {
            buffer.init((BufferType)src_buf_type, bank_base_address, page_size);
        }
        else{
            buffer.init_sharded(page_size, sharded_buffer_num_cores, bank_base_address,
                            command_ptr + COMMAND_PTR_SHARD_IDX);
        }

        while (num_writes_completed != num_pages) {
            // Context switch between reading in pages and sending them to the consumer.
            // These APIs are non-blocking to allow for context switching.
            if (cb_producer_space_available(num_to_read) and num_reads_issued < num_pages) {
                uint32_t l1_write_ptr = get_write_ptr(0);
                buffer.noc_async_read_buffer(l1_write_ptr, src_page_id, num_to_read);
                cb_push_back(0, num_to_read);
                num_reads_issued += num_to_read;
                src_page_id += num_to_read;

                uint32_t num_pages_left = num_pages - num_reads_issued;
                num_to_read = min(num_pages_left, fraction_of_producer_cb_num_pages);
            }

            if (num_reads_issued > num_writes_completed and cb_consumer_space_available(db_buf_switch, num_to_write)) {
                if (num_writes_completed == num_reads_completed) {
                    noc_async_read_barrier();
                    num_reads_completed = num_reads_issued;
                }

                uint32_t dst_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_wr_ptr_addr(db_buf_switch))[0] << 4;
                uint64_t dst_noc_addr = consumer_noc_encoding | dst_addr;
                uint32_t l1_read_ptr = get_read_ptr(0);
                noc_async_write(l1_read_ptr, dst_noc_addr, page_size * num_to_write);
                multicore_cb_push_back(consumer_noc_encoding, l1_consumer_fifo_limit, consumer_cb_size, db_buf_switch, page_size, num_to_write);
                noc_async_write_barrier();
                cb_pop_front(0, num_to_write);
                num_writes_completed += num_to_write;
                num_to_write = min(num_pages - num_writes_completed, producer_consumer_transfer_num_pages);
            }
        }
        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
    }
}
