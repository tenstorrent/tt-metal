#include "dataflow_api.h"
// #include "tt_metal/impl/dispatch/device_command.hpp"

static constexpr u32 PROGRAM_CB_ID = 0;

FORCE_INLINE
void multicore_cb_wait_front(bool db_buf_switch, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    u32 pages_acked = *reinterpret_cast<volatile u32*>(get_db_cb_ack_addr(db_buf_switch));
    volatile u32* pages_received_ptr = reinterpret_cast<volatile u32*>(get_db_cb_recv_addr(db_buf_switch));

    u16 pages_received;
    do {
        pages_received = u16(*pages_received_ptr) - pages_acked;
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'R', 'B', 'D');
}

void multicore_cb_pop_front(
    u64 producer_noc_encoding, bool db_buf_switch, u32 fifo_limit, u32 fifo_size, u32 num_pages, u32 page_size) {
    volatile u32* CQ_CONSUMER_CB_ACK_PTR = reinterpret_cast<volatile u32*>(get_db_cb_ack_addr(db_buf_switch));
    volatile u32* CQ_CONSUMER_CB_READ_PTR = reinterpret_cast<volatile u32*>(get_db_cb_rd_ptr_addr(db_buf_switch));

    *CQ_CONSUMER_CB_ACK_PTR += num_pages;
    *CQ_CONSUMER_CB_READ_PTR += (page_size * num_pages) >> 4;

    if ((*CQ_CONSUMER_CB_READ_PTR << 4) > fifo_limit) {
        *CQ_CONSUMER_CB_READ_PTR -= fifo_size >> 4;
    }

    u32 pages_ack_addr = get_db_cb_ack_addr(db_buf_switch);
    noc_semaphore_set_remote(u32(CQ_CONSUMER_CB_ACK_PTR), producer_noc_encoding | pages_ack_addr);
}

FORCE_INLINE
u32 get_read_ptr(bool db_buf_switch) {
    return *reinterpret_cast<volatile u32*>(get_db_cb_rd_ptr_addr(db_buf_switch)) << 4;
}

inline u32 min(u32 a, u32 b) { return (a < b) ? a : b; }

FORCE_INLINE void write_buffers(
    volatile tt_l1_ptr u32* command_ptr,
    u32 num_destinations,
    u32 consumer_cb_size,
    u64 producer_noc_encoding,
    bool db_buf_switch) {
    for (u32 i = 0; i < num_destinations; i++) {
        const u32 bank_base_address = command_ptr[1];
        const u32 num_pages = command_ptr[2];
        const u32 page_size = command_ptr[3];
        const u32 dst_buf_type = command_ptr[5];
        Buffer buffer((BufferType)dst_buf_type, bank_base_address, page_size);

        u32 num_to_write = 1;

        u32 src_addr = *reinterpret_cast<volatile u32*>(get_db_cb_rd_ptr_addr(db_buf_switch)) << 4;
        u32 l1_consumer_fifo_limit = src_addr + consumer_cb_size - 1;
        for (u32 id = 0; id < num_pages; id += num_to_write) {
            multicore_cb_wait_front(db_buf_switch, num_to_write);
            u32 src_addr = get_read_ptr(db_buf_switch);
            buffer.noc_async_write_buffer(src_addr, id, num_to_write, 0);
            noc_async_write_barrier();
            multicore_cb_pop_front(
                producer_noc_encoding, db_buf_switch, l1_consumer_fifo_limit, consumer_cb_size, num_to_write, page_size);
            noc_async_write_barrier();
        }
    }
}

FORCE_INLINE
void write_program_page(u32 page_addr, volatile u32*& command_ptr) {
    u32 num_transfers = command_ptr[0];
    command_ptr++;
    u32 src = page_addr;

    for (u32 i = 0; i < num_transfers; i++) {
        u32 num_bytes = command_ptr[0];
        u32 dst = command_ptr[1];
        u32 dst_noc = command_ptr[2];
        u32 num_recv = command_ptr[3];

        // advance is false if we are sending the same data to different rectangles of workers
        bool last_transfer_in_group = command_ptr[4];

        noc_async_write_multicast(src, (u64(dst_noc) << 32) | dst, num_bytes, num_recv);
        command_ptr += 5;
        if (last_transfer_in_group) {
            src = align(src + num_bytes, 16);
        }
    }
}

FORCE_INLINE
void write_and_launch_program(
    u32 num_pages, volatile u32*& command_ptr, u64 producer_noc_encoding, u32 consumer_cb_size, bool db_buf_switch) {
    u32 l1_consumer_fifo_limit = get_read_ptr(db_buf_switch) + consumer_cb_size - 1;

    if (not num_pages) {
        return;
    }

    // GO signals are just data within pages, so we need to set
    // our local 'recv' address value to 0 before we initiate
    // any transfers
    volatile tt_l1_ptr uint32_t* message_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(DISPATCH_MESSAGE_ADDR);
    *message_addr_ptr = 0;

    for (u32 page_idx = 0; page_idx < num_pages; page_idx++) {
        multicore_cb_wait_front(db_buf_switch, 1);
        u32 src_addr = get_read_ptr(db_buf_switch);
        write_program_page(src_addr, command_ptr);
        noc_async_write_barrier();
        multicore_cb_pop_front(
            producer_noc_encoding,
            db_buf_switch,
            l1_consumer_fifo_limit,
            consumer_cb_size,
            1,
            DeviceCommand::PROGRAM_PAGE_SIZE);
        noc_async_write_barrier();  // Flush barrier, not an ack barrier
    }
}

FORCE_INLINE void wait_for_program_completion(
    u32 num_workers, volatile tt_l1_ptr u32*& command_ptr, u32 tensix_soft_reset_addr) {
    if (not num_workers)
        return;

    // Wait on worker cores to notify me that they have completed
    DEBUG_STATUS('Q', 'W');

    volatile tt_l1_ptr uint32_t* message_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(DISPATCH_MESSAGE_ADDR);
    while (*message_addr_ptr != num_workers);

    DEBUG_STATUS('Q', 'D');
}

FORCE_INLINE void notify_host_complete() {
    volatile tt_l1_ptr u32* finish_ptr = get_cq_finish_ptr();
    finish_ptr[0] = 1;
    constexpr static u64 pcie_core_noc_encoding = u64(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32;
    u64 finish_noc_addr = pcie_core_noc_encoding | HOST_CQ_FINISH_PTR;
    noc_async_write(u32(finish_ptr), finish_noc_addr, 4);
    noc_async_write_barrier();
    finish_ptr[0] = 0;
}
