// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/device_command.hpp"

#define ATTR_ALIGNL1 __attribute__((aligned(L1_ALIGNMENT)))
struct db_cb_config_t {
    volatile uint32_t ack ATTR_ALIGNL1;
    volatile uint32_t recv ATTR_ALIGNL1;
    volatile uint32_t num_pages ATTR_ALIGNL1;
    volatile uint32_t page_size ATTR_ALIGNL1;   // 16B
    volatile uint32_t total_size ATTR_ALIGNL1;  // 16B
    volatile uint32_t rd_ptr ATTR_ALIGNL1;      // 16B
    volatile uint32_t wr_ptr ATTR_ALIGNL1;      // 16B
};
static constexpr uint32_t l1_db_cb_addr_offset = sizeof(db_cb_config_t);

template <uint32_t cmd_base_address, uint32_t consumer_data_buffer_size>
FORCE_INLINE uint32_t get_command_slot_addr(bool db_buf_switch) {
    static constexpr uint32_t command0_start = cmd_base_address;
    static constexpr uint32_t command1_start = command0_start + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + consumer_data_buffer_size;
    return (not db_buf_switch) ? command0_start : command1_start;
}

template <uint32_t cmd_base_address, uint32_t consumer_data_buffer_size>
FORCE_INLINE uint32_t get_db_buf_addr(bool db_buf_switch) {
    static constexpr uint32_t buf0_start = cmd_base_address + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    static constexpr uint32_t buf1_start = buf0_start + consumer_data_buffer_size + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    return (not db_buf_switch) ? buf0_start : buf1_start;
}


FORCE_INLINE
void db_acquire(volatile uint32_t* semaphore, uint64_t noc_encoding) {
    while (semaphore[0] == 0);
    noc_semaphore_inc(noc_encoding | uint32_t(semaphore), -1); // Two's complement addition
    noc_async_write_barrier();
}
