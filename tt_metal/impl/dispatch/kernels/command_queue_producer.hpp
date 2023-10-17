#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/src/firmware/riscv/common/risc_attribs.h"

/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */


FORCE_INLINE
bool cb_producer_space_available(i32 num_pages) {
    u32 operand = 0;
    uint32_t pages_acked_ptr = (uint32_t) get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    DEBUG_STATUS('C', 'R', 'B', 'W');

    // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
    // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
    uint16_t pages_acked = (uint16_t)reg_read_barrier(pages_acked_ptr);
    uint16_t free_space_pages_wrap =
        cb_interface[operand].fifo_num_pages - (pages_received - pages_acked);
    free_space_pages = (int32_t)free_space_pages_wrap;
    return free_space_pages >= num_pages;
}

FORCE_INLINE
u32 min(u32 a, u32 b) { return (a < b) ? a: b; }

FORCE_INLINE
bool cb_consumer_space_available(bool db_buf_switch, int32_t num_pages) {

    DEBUG_STATUS('C', 'R', 'B', 'W');

    u16 pages_acked = *reinterpret_cast<volatile u32*>(get_db_cb_ack_addr(db_buf_switch));
    u16 pages_recv = *reinterpret_cast<volatile u32*>(get_db_cb_recv_addr(db_buf_switch));
    u32 num_pages_consumer = *reinterpret_cast<volatile u32*>(get_db_cb_num_pages_addr(db_buf_switch));

    u16 free_space_pages_wrap = num_pages_consumer - (pages_recv - pages_acked);
    i32 free_space_pages = (i32)free_space_pages_wrap;
    DEBUG_STATUS('C', 'R', 'B', 'D');

    return free_space_pages >= num_pages;
}

FORCE_INLINE
void multicore_cb_push_back(u64 consumer_noc_encoding, u32 consumer_fifo_limit, u32 consumer_fifo_size, bool db_buf_switch, u32 page_size, u32 num_to_write) {
    // TODO(agrebenisan): Should create a multi-core CB interface... struct in L1
    volatile u32* CQ_CONSUMER_CB_RECV_PTR = reinterpret_cast<volatile u32*>(get_db_cb_recv_addr(db_buf_switch));
    volatile u32* CQ_CONSUMER_CB_WRITE_PTR = reinterpret_cast<volatile u32*>(get_db_cb_wr_ptr_addr(db_buf_switch));

    *CQ_CONSUMER_CB_RECV_PTR += num_to_write;
    *CQ_CONSUMER_CB_WRITE_PTR += (page_size * num_to_write) >> 4;

    if ((*CQ_CONSUMER_CB_WRITE_PTR << 4) > consumer_fifo_limit) {
        *CQ_CONSUMER_CB_WRITE_PTR -= consumer_fifo_size >> 4;
    }

    u32 pages_recv_addr = get_db_cb_recv_addr(db_buf_switch);
    noc_semaphore_set_remote(u32(CQ_CONSUMER_CB_RECV_PTR), consumer_noc_encoding | pages_recv_addr);
}

FORCE_INLINE
void relay_command(bool db_buf_switch, u64 consumer_noc_encoding) {
    /*
        Relays the current command to the consumer.
    */

    u64 consumer_command_slot_addr = consumer_noc_encoding | get_command_slot_addr(db_buf_switch);
    noc_async_write(L1_UNRESERVED_BASE, consumer_command_slot_addr, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
    noc_async_write_barrier();
}

void produce(
    volatile tt_l1_ptr u32* command_ptr, u32 num_srcs, u32 page_size, u32 producer_cb_size, u32 producer_cb_num_pages,
    u32 consumer_cb_size, u32 consumer_cb_num_pages, u64 consumer_noc_encoding, u32 producer_consumer_transfer_num_pages, bool db_buf_switch) {
    /*
        This API prefetches data from host memory and writes data to the consumer core. On the consumer,
        we partition the data space into 2 via double-buffering. There are two command slots, and two
        data slots. The producer reads in data into its local buffer and checks whether it can write to
        the consumer. It continues like this in a loop, context switching between pulling in data and
        writing to the consumer.
    */
    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
    u32 l1_consumer_fifo_limit = get_db_buf_addr(db_buf_switch) + consumer_cb_size - 1;

    for (u32 i = 0; i < num_srcs; i++) {
        const u32 bank_base_address = command_ptr[0];
        const u32 num_pages = command_ptr[2];
        const u32 page_size = command_ptr[3];
        const u32 src_buf_type = command_ptr[4];

        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;

        Buffer src_buffer((BufferType)src_buf_type, bank_base_address, page_size);
        u32 fraction_of_producer_cb_num_pages = consumer_cb_num_pages / 2;

        u32 num_to_read = min(num_pages, fraction_of_producer_cb_num_pages);
        u32 num_to_write = min(num_pages, producer_consumer_transfer_num_pages); // This must be a bigger number for perf.
        u32 num_reads_issued = 0;
        u32 num_reads_completed = 0;
        u32 num_writes_completed = 0;

        while (num_writes_completed != num_pages) {
            // Context switch between reading in pages and sending them to the consumer.
            // These APIs are non-blocking to allow for context switching.
            if (cb_producer_space_available(num_to_read) and num_reads_issued < num_pages) {
                u32 l1_write_ptr = get_write_ptr(0);
                src_buffer.noc_async_read_buffer(l1_write_ptr, num_reads_issued, num_to_read, 0);
                cb_push_back(0, num_to_read);
                num_reads_issued += num_to_read;

                u32 num_pages_left = num_pages - num_reads_issued;
                num_to_read = min(num_pages_left, fraction_of_producer_cb_num_pages);
            }

            if (num_reads_issued > num_writes_completed and cb_consumer_space_available(db_buf_switch, num_to_write)) {
                if (num_writes_completed == num_reads_completed) {
                    noc_async_read_barrier();
                    num_reads_completed = num_reads_issued;
                }

                u32 dst_addr = reinterpret_cast<volatile u32*>(get_db_cb_wr_ptr_addr(db_buf_switch))[0] << 4;
                u64 dst_noc_addr = consumer_noc_encoding | dst_addr;
                u32 l1_read_ptr = get_read_ptr(0);
                noc_async_write(l1_read_ptr, dst_noc_addr, page_size * num_to_write);
                multicore_cb_push_back(consumer_noc_encoding, l1_consumer_fifo_limit, consumer_cb_size, db_buf_switch, page_size, num_to_write);
                noc_async_write_barrier();
                cb_pop_front(0, num_to_write);
                num_writes_completed += num_to_write;
                num_to_write = min(num_pages - num_writes_completed, producer_consumer_transfer_num_pages);
            }
        }
    }
}
