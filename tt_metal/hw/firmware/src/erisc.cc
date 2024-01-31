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
#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"

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

    // In FD_SRC mode dispatch core is the remote issue queue reader which sends FD packets to eth router
    // In FD_DST mode dispatch core is the remote command processor which receives FD packets from eth router
    uint64_t dispatch_core_noc_encoding = get_dispatch_core_noc_encoding();
    uint64_t eth_router_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;

    volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));

    static constexpr uint32_t command_start_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
    static constexpr uint32_t data_buffer_size = MEM_ETH_SIZE - (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t)) - eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;

    bool db_buf_switch = false;
    db_cb_config_t *eth_db_cb_config =
        (db_cb_config_t *)(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset));
    const db_cb_config_t *remote_db_cb_config =
        (db_cb_config_t *)(CQ_CONSUMER_CB_BASE + (db_buf_switch * l1_db_cb_addr_offset));

    uint32_t num_pages_transferred = 0;

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

            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
            uint32_t num_buffer_transfers = command_ptr[DeviceCommand::num_buffer_transfers_idx];
            uint32_t producer_consumer_transfer_num_pages =
                command_ptr[DeviceCommand::producer_router_transfer_num_pages_idx];

            command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

            for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                const uint32_t num_pages = command_ptr[2];
                uint32_t num_pages_tunneled = 0;
                while (num_pages_tunneled != num_pages) {
                    uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages);
                    multicore_eth_cb_wait_front(eth_db_cb_config, num_to_write);
                    // contains device command, maybe just send pages, and send cmd once at the start
                    internal_::send_fd_packets();
                    multicore_eth_cb_pop_front(
                        eth_db_cb_config, remote_db_cb_config, dispatch_core_noc_encoding, num_to_write);
                    num_pages_tunneled += num_to_write;
                }
            }
            noc_semaphore_inc(dispatch_core_noc_encoding | get_semaphore(0), 1);
            noc_async_write_barrier();  // Barrier for now
        } else if (routing_info->routing_mode == EthRouterMode::FD_DST) {
            // Poll until FD_SRC router sends FD packet
            // Each FD packet comprises of command header followed by command data
            internal_::wait_for_fd_packet();

            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);

            // Block until consumer can accept new command
            // `num_pages_transferred` tracks whether the remote command processor received all the data
            while (eth_db_semaphore_addr[0] == 0 and num_pages_transferred == 0) {
                internal_::risc_context_switch();
            } // Check that there is space in consumer to send command

            // Send the full command header
            constexpr uint32_t consumer_cmd_base_addr = L1_UNRESERVED_BASE;
            constexpr uint32_t consumer_data_buffer_size = (MEM_L1_SIZE - (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t)) - L1_UNRESERVED_BASE);
            uint32_t page_size = command_ptr[DeviceCommand::page_size_idx];
            // The consumer is the remote command processor which acts like a producer from the perspective of the dispatch core
            uint32_t consumer_cb_num_pages = command_ptr[DeviceCommand::producer_cb_num_pages_idx];
            uint32_t consumer_cb_size = command_ptr[DeviceCommand::producer_cb_size_idx];
            program_consumer_cb<command_start_addr, data_buffer_size, consumer_cmd_base_addr, consumer_data_buffer_size>(
                eth_db_cb_config,
                remote_db_cb_config,
                db_buf_switch,
                dispatch_core_noc_encoding,
                consumer_cb_num_pages,
                page_size,
                consumer_cb_size);
            relay_command<command_start_addr, consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, dispatch_core_noc_encoding);

            // Decrement the semaphore value
            noc_semaphore_inc(eth_router_noc_encoding | uint32_t(eth_db_semaphore_addr), -1);  // Two's complement addition
            noc_async_write_barrier();

            // Notify the consumer
            noc_semaphore_inc(dispatch_core_noc_encoding | get_semaphore(0), 1);
            noc_async_write_barrier();  // Barrier for now

            // Send the data that was in this packet
            uint32_t producer_consumer_transfer_num_pages = command_ptr[DeviceCommand::producer_router_transfer_num_pages_idx];
            uint32_t total_num_pages = command_ptr[DeviceCommand::num_pages_idx];
            uint32_t num_pages_to_tx = 0;
            if (num_pages_transferred < total_num_pages) {
                uint32_t src_addr = eth_db_cb_config->rd_ptr << 4;
                uint64_t dst_noc_addr = dispatch_core_noc_encoding | (eth_db_cb_config->wr_ptr << 4);

                command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER; // jump to buffer transfer region
                const uint32_t num_pages = command_ptr[2];
                // producer_consumer_transfer_num_pages is the total num of data pages that could fit in a FD packet
                num_pages_to_tx = min(num_pages, producer_consumer_transfer_num_pages);

                noc_async_write(src_addr, dst_noc_addr, page_size * num_pages_to_tx);
                uint32_t l1_consumer_fifo_limit_16B =
                    (get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + consumer_cb_size) >> 4;
                multicore_cb_push_back(
                    eth_db_cb_config,
                    remote_db_cb_config,
                    dispatch_core_noc_encoding,
                    l1_consumer_fifo_limit_16B,
                    num_pages_to_tx
                );
            }
            num_pages_transferred += num_pages_to_tx;
            if (num_pages_transferred == total_num_pages) {
                // Done sending all data associated with this command, reset `num_pages_transferred` for next command
                num_pages_transferred = 0;
            }

            // Signal to FD_SRC router that packet has been processed
            internal_::ack_fd_packet();
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
    kernel_profiler::mark_time(CC_MAIN_END);
}
