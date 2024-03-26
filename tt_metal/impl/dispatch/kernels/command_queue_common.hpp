// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "debug/assert.h"

#define ATTR_ALIGNL1 __attribute__((aligned(L1_ALIGNMENT)))
struct db_cb_config_t {
    volatile uint32_t ack ATTR_ALIGNL1;
    volatile uint32_t recv ATTR_ALIGNL1;
    volatile uint32_t num_pages ATTR_ALIGNL1;
    volatile uint32_t page_size_16B ATTR_ALIGNL1;   // 16B
    volatile uint32_t total_size_16B ATTR_ALIGNL1;  // 16B
    volatile uint32_t rd_ptr_16B ATTR_ALIGNL1;      // 16B
    volatile uint32_t wr_ptr_16B ATTR_ALIGNL1;      // 16B
    volatile uint32_t fifo_limit_16B ATTR_ALIGNL1;  // 16B
};
static constexpr uint32_t l1_db_cb_addr_offset = sizeof(db_cb_config_t);

static constexpr uint32_t CQ_DISPATCHER_CB_CONFIG_BASE = CQ_CONSUMER_CB_BASE + l1_db_cb_addr_offset;

template <uint32_t cmd_base_address, uint32_t data_buffer_size>
FORCE_INLINE uint32_t get_command_slot_addr(bool db_buf_switch) {
    static constexpr uint32_t command0_start = cmd_base_address;
    static constexpr uint32_t command1_start = cmd_base_address + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND  + data_buffer_size;
    return (not db_buf_switch) ? command0_start : command1_start;
}

FORCE_INLINE
void db_acquire(volatile uint32_t* semaphore, uint64_t noc_encoding) {
#if defined(COMPILE_FOR_IDLE_ERISC)
    uint32_t heartbeat = 0;
#endif
    while (semaphore[0] == 0) {
#if defined(COMPILE_FOR_IDLE_ERISC)
        RISC_POST_HEARTBEAT(heartbeat);
#endif
    };
    noc_semaphore_inc(noc_encoding | uint32_t(semaphore), -1); // Two's complement addition
    noc_async_write_barrier();
}

// Local refers to the core that is calling this function
tt_l1_ptr db_cb_config_t* get_local_db_cb_config(uint32_t base_addr, bool db_buf_switch = false) {
    // TODO: remove multiply here
    db_cb_config_t* db_cb_config = (db_cb_config_t*)(base_addr + (db_buf_switch * l1_db_cb_addr_offset));
    return db_cb_config;
}

// Remote refers to any other core on the same chip
tt_l1_ptr db_cb_config_t* get_remote_db_cb_config(uint32_t base_addr, bool db_buf_switch = false) {
    // TODO: remove multiply here
    db_cb_config_t* db_cb_config = (db_cb_config_t*)(base_addr + (db_buf_switch * l1_db_cb_addr_offset));
    return db_cb_config;
}

// For FD v1.3 there is only one circular buffer for data
template <SyncCBConfigRegion cb_config_region, uint32_t data_buffer_size = 0>
FORCE_INLINE uint32_t get_cb_start_address(bool db_buf_switch) {
    if constexpr (cb_config_region == SyncCBConfigRegion::ROUTER_ISSUE) {
        return eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    } else if constexpr (cb_config_region == SyncCBConfigRegion::ROUTER_COMPLETION) {
        return eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE +  eth_l1_mem::address_map::ERISC_L1_TUNNEL_BUFFER_SIZE + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    } else if constexpr (cb_config_region == SyncCBConfigRegion::DB_TENSIX) {
        static constexpr uint32_t buf0_start = L1_UNRESERVED_BASE + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
        static constexpr uint32_t buf1_start = buf0_start + data_buffer_size + DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
        return (not db_buf_switch) ? buf0_start : buf1_start;
    } else {
        return L1_UNRESERVED_BASE + 2 * DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND;
    }
}


FORCE_INLINE
int32_t multicore_cb_pages_left(volatile tt_l1_ptr db_cb_config_t* db_cb_config) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t free_space_pages_wrap = db_cb_config->num_pages - (db_cb_config->recv - db_cb_config->ack);
    int32_t free_space_pages = (int32_t)free_space_pages_wrap;
    DEBUG_STATUS('C', 'R', 'B', 'D');
    return free_space_pages;
}

FORCE_INLINE
bool multicore_cb_space_available(volatile tt_l1_ptr db_cb_config_t* db_cb_config, int32_t num_pages) {
    // TODO: delete cb_consumer_space_available and use this one
    return multicore_cb_pages_left(db_cb_config) >= num_pages;
}

FORCE_INLINE
void multicore_cb_push_back(
    volatile db_cb_config_t* db_cb_config,
    const volatile db_cb_config_t* remote_db_cb_config,
    uint64_t consumer_noc_encoding,
    uint32_t consumer_fifo_16b_limit,
    uint32_t num_to_write) {
    db_cb_config->recv += num_to_write;
    db_cb_config->wr_ptr_16B += db_cb_config->page_size_16B * num_to_write;

    ASSERT(db_cb_config->wr_ptr_16B <= consumer_fifo_16b_limit);
    if (db_cb_config->wr_ptr_16B == consumer_fifo_16b_limit) {
        db_cb_config->wr_ptr_16B -= db_cb_config->total_size_16B;
    }

    uint32_t remote_pages_recv_addr = (uint32_t)(&(remote_db_cb_config->recv));
    noc_semaphore_set_remote((uint32_t)(&(db_cb_config->recv)), consumer_noc_encoding | remote_pages_recv_addr);
}

FORCE_INLINE
void multicore_cb_wait_front(volatile db_cb_config_t* db_cb_config, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_received;
    do {
        pages_received = uint16_t(db_cb_config->recv - db_cb_config->ack);
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'R', 'B', 'D');
}

void multicore_cb_pop_front(
    volatile db_cb_config_t* db_cb_config,
    const volatile db_cb_config_t* remote_db_cb_config,
    uint64_t producer_noc_encoding,
    uint32_t consumer_fifo_limit_16B,
    uint32_t num_to_write,
    uint32_t page_size_16B) {
    db_cb_config->ack += num_to_write;
    db_cb_config->rd_ptr_16B += page_size_16B * num_to_write;

    ASSERT(db_cb_config->rd_ptr_16B <= consumer_fifo_limit_16B);
    if (db_cb_config->rd_ptr_16B == consumer_fifo_limit_16B) {
        db_cb_config->rd_ptr_16B -= db_cb_config->total_size_16B;
    }

    uint32_t pages_ack_addr = (uint32_t)(&(remote_db_cb_config->ack));
    noc_semaphore_set_remote((uint32_t)(&(db_cb_config->ack)), producer_noc_encoding | pages_ack_addr);
}
