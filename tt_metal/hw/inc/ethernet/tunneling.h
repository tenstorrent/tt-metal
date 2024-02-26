
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//#include "../dataflow_api.h"
#include "eth_l1_address_map.h"
#include "noc_nonblocking_api.h"
#include "erisc.h"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_prefetcher.hpp"

//#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"

#define FORCE_INLINE inline __attribute__((always_inline))

inline void RISC_POST_STATUS(uint32_t status) {
    volatile uint32_t *ptr = (volatile uint32_t *)(NOC_CFG(ROUTER_CFG_2));
    ptr[0] = status;
}

struct erisc_info_t {
    volatile uint32_t launch_user_kernel;
    volatile uint32_t unused_arg0;
    volatile uint32_t unused_arg1;
    volatile uint32_t unused_arg2;
    volatile uint32_t user_buffer_bytes_sent;
    uint32_t reserved_0_;
    uint32_t reserved_1_;
    uint32_t reserved_2_;
};

// Routing info, initialized by erisc.cc
// TODO: turn into extern

erisc_info_t *erisc_info = (erisc_info_t *)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
routing_info_t *routing_info = (routing_info_t *)(eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE);

// Context Switch Config
tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);

extern uint32_t __erisc_jump_table;
volatile uint32_t *RtosTable =
    (volatile uint32_t *)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;

// FD configs
// Sempahore(0) syncing on src e.g. remote issue q reader, remote signaller
// Sempahore(1) syncing on dst e.g. remote command processor, remote completion writer

static constexpr uint32_t data_buffer_size = eth_l1_mem::address_map::ERISC_L1_TUNNEL_BUFFER_SIZE -
                                             (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t));

namespace internal_ {

FORCE_INLINE
void eth_send_packet(uint32_t q_num, uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words) {
    while (eth_txq_reg_read(q_num, ETH_TXQ_CMD) != 0) {
        risc_context_switch();
    }
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_START_ADDR, src_word_addr << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, dest_word_addr << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_SIZE_BYTES, num_words << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_DATA);
}

FORCE_INLINE
void eth_write_remote_reg(uint32_t q_num, uint32_t reg_addr, uint32_t val) {
    while (eth_txq_reg_read(q_num, ETH_TXQ_CMD) != 0) {
        risc_context_switch();
    }
    eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, reg_addr);
    eth_txq_reg_write(q_num, ETH_TXQ_REMOTE_REG_DATA, val);
    eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_REG);
}

void check_and_context_switch() {
    uint32_t start_time = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t end_time = start_time;
    while (end_time - start_time < 100000) {
        RISC_POST_STATUS(0xdeadCAFE);
        internal_::risc_context_switch();
        RISC_POST_STATUS(0xdeadFEAD);
        end_time = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    }
    // proceed
}

FORCE_INLINE
void notify_dispatch_core_done(uint64_t dispatch_addr) {
    //  flush both nocs because ethernet kernels could be using different nocs to try to atomically increment semaphore
    //  in dispatch core
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        while (!noc_cmd_buf_ready(n, NCRISC_AT_CMD_BUF))
            ;
    }
    noc_fast_atomic_increment(noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, NOC_UNICAST_WRITE_VC, 1, 31 /*wrap*/, false /*linked*/);
}

FORCE_INLINE
void send_fd_packets(uint8_t buffer_id) {
    internal_::eth_send_packet(
        0,
        (eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + buffer_id * eth_l1_mem::address_map::ERISC_L1_TUNNEL_BUFFER_SIZE) >> 4,
        (eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + buffer_id * eth_l1_mem::address_map::ERISC_L1_TUNNEL_BUFFER_SIZE) >> 4,
        (eth_l1_mem::address_map::ERISC_L1_TUNNEL_BUFFER_SIZE) >> 4);
    routing_info->fd_buffer_msgs[buffer_id].bytes_sent = 1;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(routing_info->fd_buffer_msgs[buffer_id].bytes_sent))) >> 4,
        ((uint32_t)(&(routing_info->fd_buffer_msgs[buffer_id].bytes_sent))) >> 4,
        1);
    // There should always be a valid cmd here, since eth_db_acquire completed
    while (routing_info->fd_buffer_msgs[buffer_id].bytes_sent != 0) {
        // routing_info->routing_enabled && erisc_info->launch_user_kernel == 0){
        // TODO: add timer to restrict this
        risc_context_switch();
    }
}

FORCE_INLINE
void wait_for_fd_packet(uint8_t buffer_id) {
    // There may not be a valid cmd here, since DST router is always polling
    // This should only happen on cluster close
    while (routing_info->fd_buffer_msgs[buffer_id].bytes_sent != 1 && routing_info->routing_enabled) {
        // TODO: add timer to restrict this
        risc_context_switch();
    }
}

FORCE_INLINE
void ack_fd_packet(uint8_t buffer_id) {
    routing_info->fd_buffer_msgs[buffer_id].bytes_sent = 0;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(routing_info->fd_buffer_msgs[buffer_id].bytes_sent))) >> 4,
        ((uint32_t)(&(routing_info->fd_buffer_msgs[buffer_id].bytes_sent))) >> 4,
        1);
}

}  // namespace internal_

FORCE_INLINE
void multicore_eth_cb_wait_front(db_cb_config_t *eth_db_cb_config, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_received;
    do {
        pages_received = uint16_t(eth_db_cb_config->recv - eth_db_cb_config->ack);
        internal_::risc_context_switch(); // AL: hopefully we can remove this...
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
void eth_db_acquire(volatile uint32_t *semaphore, uint64_t noc_encoding) {
    while (semaphore[0] == 0 and routing_info->routing_enabled) {
        // Without this context switch a src router on R chip of N300 may get configured for FD
        //  and block L chip of N300 from sending config for dst router on R because the path to the dst router is through the src router
        internal_::risc_context_switch();
    }
}

// Implement yielding if SENDER is not ISSUE, this may help with devices getting commands first
template<uint8_t buffer_id, uint8_t other_buffer_id, bool sender_is_issue_path>
FORCE_INLINE
void eth_tunnel_src_forward_one_cmd(db_cb_config_t *eth_db_cb_config, uint32_t relay_noc_encoding) {
    volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));

    constexpr uint32_t command_start_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + buffer_id * eth_l1_mem::address_map::ERISC_L1_TUNNEL_BUFFER_SIZE;
    static constexpr uint32_t data_buffer_size = eth_l1_mem::address_map::ERISC_L1_TUNNEL_BUFFER_SIZE - (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t));
     uint32_t eth_router_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));

    bool db_buf_switch = false;
    const db_cb_config_t *remote_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE);


            noc_semaphore_inc(
                ((uint64_t)eth_router_noc_encoding << 32) | uint32_t(eth_db_semaphore_addr),
                -1);  // Two's complement addition
            noc_async_write_barrier();

            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
            volatile tt_l1_ptr CommandHeader *header = (CommandHeader *)command_ptr;
            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            uint32_t router_transfer_num_pages = header->router_transfer_num_pages;
            bool is_program = header->is_program_buffer;
            bool fwd_path = header->fwd_path;
            command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

            // DPRINT << "SRC got" << ENDL();

            // send cmd even if there is no data associated
            erisc_info->unused_arg2 = 66666666;
            internal_::send_fd_packets(buffer_id); // TODO: AL, is this right?
            erisc_info->unused_arg2 = 66666667;
            // DPRINT << "SRC2DST" << ENDL();

            for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                // erisc_info->unused_arg0 = 109;
                const uint32_t num_pages = command_ptr[2];
                const uint32_t src_buf_type = command_ptr[4];
                const uint32_t dst_buf_type = command_ptr[5];

                bool read_from_sysmem = (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY;
                bool write_to_sysmem = (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY;
                bool tunnel_data = (read_from_sysmem) | (write_to_sysmem & !fwd_path & !is_program);

                if (!tunnel_data) {
                    // erisc_info->unused_arg0 = 500;
                    continue;
                }

                uint32_t num_to_write = min(num_pages, router_transfer_num_pages);

                uint32_t num_pages_tunneled = 0;
                while (num_pages_tunneled != num_pages) {
                    // DPRINT << "SRC waiting for " << num_to_write << ENDL();
                    erisc_info->unused_arg2 = 66666668;
                    multicore_eth_cb_wait_front(eth_db_cb_config, num_to_write);
                    erisc_info->unused_arg2 = 66666669;
                    internal_::send_fd_packets(buffer_id); // AL: increment since idx to msg sent
                    erisc_info->unused_arg2 = 66666670;
                    multicore_eth_cb_pop_front(
                        eth_db_cb_config, remote_db_cb_config, ((uint64_t)relay_noc_encoding << 32), num_to_write);
                    num_pages_tunneled += num_to_write;
                    num_to_write = min(num_pages - num_pages_tunneled, router_transfer_num_pages);
                }
                command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
            }

            // DPRINT << "SRCD" << ENDL();
            noc_semaphore_inc(((uint64_t)relay_noc_encoding << 32) | get_semaphore(0), 1);
            noc_async_write_barrier();  // Barrier for now
            // erisc_info->unused_arg1 = routing_info->relay_src_x;
            // erisc_info->unused_arg2 = routing_info->relay_src_y;
}

// Implement yielding if SENDER is not ISSUE, this may help with devices getting commands first
template <uint8_t buffer_id, uint8_t other_buffer_id, bool sender_is_issue_path>
FORCE_INLINE
void eth_tunnel_dst_forward_one_cmd(db_cb_config_t *eth_db_cb_config, uint32_t relay_noc_encoding) {
    /*
    // Debug: stall forever if no valid fd cmd
    // TODO: turn into watcher assert
    if (routing_info->fd_buffer_msgs_sent == 0) {
        while(true) {
            RISC_POST_STATUS(0x02140000);
        }
    }*/
    volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(1));

    bool db_buf_switch = false;
    const db_cb_config_t *remote_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE);
    uint32_t eth_router_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));
    constexpr uint32_t command_start_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + buffer_id * eth_l1_mem::address_map::ERISC_L1_TUNNEL_BUFFER_SIZE;

            // Poll until FD_SRC router sends FD packet
            // Each FD packet comprises of command header followed by command data
            internal_::wait_for_fd_packet(buffer_id);
            // DPRINT << "DST routing enabled" << ENDL();

            // tell pull_and_relay core that command is available

            // DPRINT << "DST GOT" << ENDL();


            // DPRINT << "DST INFORMED" << ENDL();
            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
            volatile tt_l1_ptr CommandHeader *header = (CommandHeader *)command_ptr;
            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            uint32_t router_transfer_num_pages = header->router_transfer_num_pages;
            uint32_t page_size = header->page_size;
            bool is_program = header->is_program_buffer;
            bool fwd_path = header->fwd_path;
            command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

            // Initially 1
            // After update_producer_consumer_sync_semaphores goes to 0
            // At some point pull_and_relay will set it to 1 once it reads in the command
            update_producer_consumer_sync_semaphores(((uint64_t)eth_router_noc_encoding << 32), ((uint64_t)relay_noc_encoding << 32), eth_db_semaphore_addr, get_semaphore(1));

            // Wait until push and pull kernel has read in the command
            // Before pull and push kernel signal to DST router that it has read in a command, it also programs the DST router CB
            while (eth_db_semaphore_addr[0] == 0) {
                internal_::risc_context_switch();
            }

            // Ack because pull and push kernel signalled that it got the command
            internal_::ack_fd_packet(buffer_id);

            for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                const uint32_t num_pages = command_ptr[2];
                const uint32_t src_buf_type = command_ptr[4];
                const uint32_t dst_buf_type = command_ptr[5];

                bool read_from_sysmem = (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY;
                bool write_to_sysmem = (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY;
                bool tunnel_data = (read_from_sysmem) | (write_to_sysmem & !is_program & !fwd_path);

                if (!tunnel_data) {
                    continue;
                }

                // router_transfer_num_pages is the total num of data pages that could fit in a FD packet
                uint32_t num_pages_to_signal = min(num_pages, router_transfer_num_pages);
                uint32_t num_pages_transferred = 0;

                while (num_pages_transferred != num_pages) {
                    while (routing_info->fd_buffer_msgs[buffer_id].bytes_sent != 1) { // wait for SRC router to send data
                        internal_::risc_context_switch();
                    }

                    erisc_info->unused_arg1 = 12340000 + num_pages_to_signal;
                    while (not cb_consumer_space_available(eth_db_cb_config, num_pages_to_signal)) {
                      internal_::risc_context_switch(); // noc stall so don't need to switch to base FW --> do we need this?
                    }
                    erisc_info->unused_arg1 = 12350000 + num_pages_to_signal;
                    multicore_cb_push_back( // signal to pull and push kernel that data is ready
                        eth_db_cb_config,
                        remote_db_cb_config,
                        ((uint64_t)relay_noc_encoding << 32),
                        eth_db_cb_config->fifo_limit_16B,
                        num_pages_to_signal);
                    // wait for pull and push kernel to pick up the data
                    //erisc_info->unused_arg2 = (eth_db_cb_config->ack << 16) | (eth_db_cb_config->recv);
                    while (eth_db_cb_config->ack != eth_db_cb_config->recv) {
                        internal_::risc_context_switch();
                    }
                    erisc_info->unused_arg1 = 12360000 + num_pages_to_signal;

                    internal_::ack_fd_packet(buffer_id); // signal to SRC router that more data can be sent

                    num_pages_transferred += num_pages_to_signal;
                    num_pages_to_signal = min(num_pages - num_pages_transferred, router_transfer_num_pages);
                }
                command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;  // jump to buffer transfer region
            }

}

FORCE_INLINE
void run_routing() {
    //router_init();
    // TODO: maybe split into two FWs? or this may be better to sometimes allow each eth core to do both send and
    // receive of fd packets
        internal_::risc_context_switch();
}
