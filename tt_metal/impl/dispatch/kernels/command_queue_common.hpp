// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/device_command.hpp"

static constexpr uint32_t l1_db_cb_addr_offset = 7 * 16;

FORCE_INLINE
uint32_t get_db_cb_command_slot_addr(bool db_buf_switch) {
    constexpr static uint32_t second_buffer_offset = (MEM_L1_SIZE - L1_UNRESERVED_BASE) / 2;
    return L1_UNRESERVED_BASE + (db_buf_switch * second_buffer_offset);
}

FORCE_INLINE
uint32_t get_db_cb_l1_base(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset);
}

FORCE_INLINE
uint32_t get_db_cb_ack_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset);
}

FORCE_INLINE
uint32_t get_db_cb_recv_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 16);
}

FORCE_INLINE
uint32_t get_db_cb_num_pages_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 32);
}

FORCE_INLINE
uint32_t get_db_cb_page_size_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 48);
}

FORCE_INLINE
uint32_t get_db_cb_total_size_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 64);
}

FORCE_INLINE
uint32_t get_db_cb_rd_ptr_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + 80);

}

FORCE_INLINE
uint32_t get_db_cb_wr_ptr_addr(bool db_buf_switch) {
    return CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset + CQ_START);
}


FORCE_INLINE
uint32_t get_command_slot_addr(bool db_buf_switch) {
    static constexpr uint32_t command0_start = L1_UNRESERVED_BASE;
    static constexpr uint32_t command1_start = command0_start + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + DeviceCommand::CONSUMER_DATA_BUFFER_SIZE;
    return (db_buf_switch) ? command0_start : command1_start;
}

FORCE_INLINE
uint32_t get_db_buf_addr(bool db_buf_switch) {
    static constexpr uint32_t buf0_start = L1_UNRESERVED_BASE + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    static constexpr uint32_t buf1_start = buf0_start + DeviceCommand::CONSUMER_DATA_BUFFER_SIZE + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    return (not db_buf_switch) ? buf0_start : buf1_start;
}


FORCE_INLINE
void db_acquire(volatile uint32_t* semaphore, uint64_t noc_encoding) {
    while (semaphore[0] == 0);
    noc_semaphore_inc(noc_encoding | uint32_t(semaphore), -1); // Two's complement addition
    noc_async_write_barrier();
}
