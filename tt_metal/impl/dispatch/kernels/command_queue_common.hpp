/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/device_command.hpp"

// TODO(pgk) move all this to host/device interface
// static uint32_t go_packet[4] __attribute__((section("l1_data"))) __attribute__((aligned(16))) = {
//     RUN_MESSAGE_GO,     // brisc
//     true,               // enable ncrisc (TODO(pgk))
//     true,               // enable trisc (TODO(pgk))
//     0,                  // ncrisc fw size (TODO(pgk))
// };

static launch_msg_t launch_msg __attribute__((section("l1_data"))) __attribute__((aligned(16))) = {
    .kernel_group_id = 0,
    .ncrisc_fw_size = 0,
    .mode = DISPATCH_MODE_DEV,
    .enable_brisc = true,
    .enable_ncrisc = true,
    .enable_triscs = true,
    .run = RUN_MSG_GO
};

static constexpr u32 l1_db_cb_addr_offset = 7 * 16;

FORCE_INLINE
u32 get_db_cb_command_slot_addr(bool db_buf_switch) {
    constexpr static u32 second_buffer_offset = (MEM_L1_SIZE - L1_UNRESERVED_BASE) / 2;
    return L1_UNRESERVED_BASE + (db_buf_switch * second_buffer_offset);
}

FORCE_INLINE
u32 get_db_cb_l1_base(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset);
}

FORCE_INLINE
u32 get_db_cb_ack_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset);
}

FORCE_INLINE
u32 get_db_cb_recv_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 16);
}

FORCE_INLINE
u32 get_db_cb_num_pages_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 32);
}

FORCE_INLINE
u32 get_db_cb_page_size_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 48);
}

FORCE_INLINE
u32 get_db_cb_total_size_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 64);
}

FORCE_INLINE
u32 get_db_cb_rd_ptr_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 80);

}

FORCE_INLINE
u32 get_db_cb_wr_ptr_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 96);
}


FORCE_INLINE
u32 get_command_slot_addr(bool db_buf_switch) {
    static constexpr u32 command0_start = L1_UNRESERVED_BASE;
    static constexpr u32 command1_start = command0_start + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + DeviceCommand::CONSUMER_DATA_BUFFER_SIZE;
    return (db_buf_switch) ? command0_start : command1_start;
}

FORCE_INLINE
u32 get_db_buf_addr(bool db_buf_switch) {
    static constexpr u32 buf0_start = L1_UNRESERVED_BASE + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    static constexpr u32 buf1_start = buf0_start + DeviceCommand::CONSUMER_DATA_BUFFER_SIZE + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    return (not db_buf_switch) ? buf0_start : buf1_start;
}


FORCE_INLINE
void db_acquire(volatile u32* semaphore, u64 noc_encoding) {
    while (semaphore[0] == 0);
    noc_semaphore_inc(noc_encoding | u32(semaphore), -1); // Two's complement addition
    noc_async_write_barrier();
}
