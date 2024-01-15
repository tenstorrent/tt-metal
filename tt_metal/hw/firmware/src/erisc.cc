// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "firmware_common.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ApplicationHandler(void) __attribute__((__section__(".init")));

#ifdef __cplusplus
}
#endif

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

tt_l1_ptr mailboxes_t * const mailboxes = (tt_l1_ptr mailboxes_t *)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);

uint8_t noc_index = 0;  // TODO: remove hardcoding
uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));

FORCE_INLINE
void multicore_eth_cb_wait_front(db_cb_config_t *eth_db_cb_config, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_received;
    do {
        pages_received = uint16_t(eth_db_cb_config->recv - eth_db_cb_config->ack);
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'R', 'B', 'D');
}

FORCE_INLINE
void multicore_eth_cb_pop_front(
    db_cb_config_t *eth_db_cb_config,
    const db_cb_config_t *remote_db_cb_config,
    uint64_t producer_noc_encoding,
    uint32_t num_pages) {
    eth_db_cb_config->ack += num_pages;

    uint32_t pages_ack_addr = (uint32_t)(&(remote_db_cb_config->ack));
    noc_semaphore_set_remote((uint32_t)(&(eth_db_cb_config->ack)), producer_noc_encoding | pages_ack_addr);
}

FORCE_INLINE
uint64_t get_dispatch_core_noc_encoding() {
    uint32_t dispatch_noc_x = (uint32_t)((routing_info->dispatch_core_xy >> 16) & 0xFFFF);
    uint32_t dispatch_noc_y = (uint32_t)(routing_info->dispatch_core_xy & 0xFFFF);
    return uint64_t(NOC_XY_ENCODING(dispatch_noc_x, dispatch_noc_y)) << 32;
}

FORCE_INLINE
void eth_db_acquire(volatile uint32_t *semaphore, uint64_t noc_encoding) {
    while (semaphore[0] == 0 and routing_info->routing_enabled) {
    }
}

void __attribute__((section("code_l1"))) risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}
void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    kernel_profiler::init_profiler();
    kernel_profiler::mark_time(CC_MAIN_START);
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    risc_init();
    noc_init();

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    while (routing_info->routing_enabled != 1) {
        internal_::risc_context_switch();
    }
    uint64_t dispatch_core_noc_encoding = get_dispatch_core_noc_encoding();
    uint64_t eth_router_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;

    volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));

    static constexpr uint32_t command_start_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;

    bool db_buf_switch = false;
    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        if (erisc_info->launch_user_kernel == 1) {
            kernel_init();
        } else if (routing_info->routing_mode == EthRouterMode::FD_SRC) {
            eth_db_acquire(eth_db_semaphore_addr, eth_router_noc_encoding);
            if (!routing_info->routing_enabled) {
                break;
            }
            noc_semaphore_inc(
                eth_router_noc_encoding | uint32_t(eth_db_semaphore_addr), -1);  // Two's complement addition
            noc_async_write_barrier();

            db_cb_config_t *eth_db_cb_config =
                (db_cb_config_t
                     *)(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset));
            const db_cb_config_t *remote_db_cb_config =
                (db_cb_config_t *)(CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset));
            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
            uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];
            uint32_t producer_consumer_transfer_num_pages =
                command_ptr[DeviceCommand::producer_consumer_transfer_num_pages_idx];
            command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

            for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                const uint32_t num_pages = command_ptr[2];
                uint32_t num_writes_completed = 0;
                while (num_writes_completed != num_pages) {
                    uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages);
                    multicore_eth_cb_wait_front(eth_db_cb_config, num_to_write);
                    // contains device command, maybe just send pages, and send cmd once at the start
                    internal_::send_fd_packets();
                    multicore_eth_cb_pop_front(
                        eth_db_cb_config, remote_db_cb_config, dispatch_core_noc_encoding, num_to_write);
                    num_writes_completed += num_to_write;
                }
            }
            noc_semaphore_inc(dispatch_core_noc_encoding | get_semaphore(0), 1);
            noc_async_write_barrier();  // Barrier for now
        } else if (routing_info->routing_mode == EthRouterMode::FD_DST) {
            internal_::receive_fd_packets();
            // TODO: add sync with remote processor
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
    kernel_profiler::mark_time(CC_MAIN_END);
}
