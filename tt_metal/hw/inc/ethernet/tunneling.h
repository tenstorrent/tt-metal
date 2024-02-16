
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../dataflow_api.h"
#include "eth_l1_address_map.h"
#include "noc_nonblocking_api.h"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_producer.hpp"

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
uint32_t relay_src_noc_encoding;
uint32_t relay_dst_noc_encoding;
uint32_t eth_router_noc_encoding;
EthRouterMode my_routing_mode;

erisc_info_t *erisc_info = (erisc_info_t *)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
routing_info_t *routing_info = (routing_info_t *)(eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE);
volatile uint32_t *flag_disable = (uint32_t *)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);

// Context Switch Config
tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);

extern uint32_t __erisc_jump_table;
volatile uint32_t *RtosTable =
    (volatile uint32_t *)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;

void (*rtos_context_switch_ptr)();

// FD configs
bool db_buf_switch = false;
db_cb_config_t *eth_db_cb_config = get_local_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, false);
const db_cb_config_t *remote_src_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, false);
const db_cb_config_t *remote_dst_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, true);

volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));

static constexpr uint32_t command_start_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
static constexpr uint32_t data_buffer_size = eth_l1_mem::address_map::ERISC_APP_RESERVED_SIZE -
                                             (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t));

void __attribute__((section("code_l1"))) router_init() {
    relay_src_noc_encoding = uint32_t(NOC_XY_ENCODING(routing_info->relay_src_x, routing_info->relay_src_y));
    relay_dst_noc_encoding = uint32_t(NOC_XY_ENCODING(routing_info->relay_dst_x, routing_info->relay_dst_y));

    eth_router_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));
    my_routing_mode = (EthRouterMode)routing_info->routing_mode;
}

namespace internal_ {
FORCE_INLINE
void __attribute__((section("code_l1"))) risc_context_switch() {
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
}

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
    noc_fast_atomic_increment_l1(noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, 1, 31 /*wrap*/, false /*linked*/);
}

FORCE_INLINE
void disable_erisc_app() { flag_disable[0] = 0; }

FORCE_INLINE
void send_fd_packets() {
    internal_::eth_send_packet(
        0,
        (eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE) >> 4,
        ((eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE)) >> 4,
        (eth_l1_mem::address_map::ERISC_APP_RESERVED_SIZE) >> 4);
    routing_info->fd_buffer_msgs_sent = 1;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(routing_info->fd_buffer_msgs_sent))) >> 4,
        ((uint32_t)(&(routing_info->fd_buffer_msgs_sent))) >> 4,
        1);
    // There should always be a valid cmd here, since eth_db_acquire completed
    while (routing_info->fd_buffer_msgs_sent != 0) {
        // routing_info->routing_enabled && erisc_info->launch_user_kernel == 0){
        // TODO: add timer to restrict this
        risc_context_switch();
    }
}

FORCE_INLINE
void wait_for_fd_packet() {
    // There may not be a valid cmd here, since DST router is always polling
    // This should only happen on cluster close
    while (routing_info->fd_buffer_msgs_sent != 1 && routing_info->routing_enabled &&
           erisc_info->launch_user_kernel == 0) {
        // TODO: add timer to restrict this
        risc_context_switch();
    }
}

FORCE_INLINE
void ack_fd_packet() {
    routing_info->fd_buffer_msgs_sent = 0;
    internal_::eth_send_packet(
        0,
        ((uint32_t)(&(routing_info->fd_buffer_msgs_sent))) >> 4,
        ((uint32_t)(&(routing_info->fd_buffer_msgs_sent))) >> 4,
        1);
}

}  // namespace internal_

FORCE_INLINE
void multicore_eth_cb_wait_front(db_cb_config_t *eth_db_cb_config, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_received;
    do {
        pages_received = uint16_t(eth_db_cb_config->recv - eth_db_cb_config->ack);
        internal_::risc_context_switch();  // AL: hopefully we can remove this...
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
    while (semaphore[0] == 0 and routing_info->routing_enabled and erisc_info->launch_user_kernel == 0) {
        // Without this context switch a src router on R chip of N300 may get configured for FD
        //  and block L chip of N300 from sending config for dst router on R because the path to the dst router is
        //  through the src router
        internal_::risc_context_switch();
    }
}

template <
    uint32_t producer_cmd_base_addr,
    uint32_t producer_data_buffer_size,
    uint32_t consumer_cmd_base_addr,
    uint32_t consumer_data_buffer_size>
FORCE_INLINE void eth_program_consumer_cb(
    db_cb_config_t *db_cb_config,
    const db_cb_config_t *remote_db_cb_config,
    bool db_buf_switch,
    uint64_t consumer_noc_encoding,
    uint32_t num_pages,
    uint32_t page_size,
    uint32_t cb_size) {
    // This sets up multi-core CB where the writer is an eth core and consumer is tensix core
    // The data is at different L1 addresses in the producer and consumer
    //  but both producer and consumer use the read pointer to determine location of data in their local L1
    // Producer uses the read pointer when sending to consumer and uses the write pointer to determine L1 address of
    // data in consumer Consumer uses the read pointer to read in data To account for differences in data location, the
    // consumer is sent cb config with the "correct" rd pointer from its POV
    //  and after sending, producer sets it back to its local L1 address

    uint32_t cb_start_producer_addr = get_db_buf_addr<producer_cmd_base_addr, producer_data_buffer_size>(db_buf_switch);
    uint32_t cb_start_consumer_addr = get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch);

    db_cb_config->ack = 0;
    db_cb_config->recv = 0;
    db_cb_config->num_pages = num_pages;
    db_cb_config->page_size_16B = page_size >> 4;
    db_cb_config->total_size_16B = cb_size >> 4;
    db_cb_config->rd_ptr_16B = cb_start_consumer_addr >> 4;  // first set the rd_ptr to value that conumer needs to see
    db_cb_config->wr_ptr_16B = cb_start_consumer_addr >> 4;

    noc_async_write(
        (uint32_t)(db_cb_config), consumer_noc_encoding | (uint32_t)(remote_db_cb_config), sizeof(db_cb_config_t));
    noc_async_write_barrier();  // barrier for now
    // db cb config has been sent to consumer, now read address can be set to expected value for eth producer
    db_cb_config->rd_ptr_16B = cb_start_producer_addr >> 4;
    noc_async_write_barrier();  // barrier for now
}

FORCE_INLINE
void eth_tunnel_src_forward_one_cmd() {
    volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));

    /*
    // Debug: stall forever if no valid fd cmd
    // TODO: turn into watcher assert
    if (eth_db_semaphore_addr[0] == 0) {
        while(true) {
            RISC_POST_STATUS(0x02130000);
        }
    }*/
    bool db_buf_switch = false;
    db_cb_config_t *eth_db_cb_config = get_local_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, false);
    const db_cb_config_t *remote_src_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, false);
    const db_cb_config_t *remote_dst_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, true);

    noc_semaphore_inc(
        ((uint64_t)eth_router_noc_encoding << 32) | uint32_t(eth_db_semaphore_addr),
        -1);  // Two's complement addition
    noc_async_write_barrier();

    volatile tt_l1_ptr uint32_t *command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
    volatile tt_l1_ptr CommandHeader *header = (CommandHeader *)command_ptr;
    uint32_t num_buffer_transfers = header->num_buffer_transfers;
    uint32_t producer_consumer_transfer_num_pages = header->producer_router_transfer_num_pages;
    bool is_program = header->is_program_buffer;
    bool fwd_path = header->fwd_path;
    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

    // if (num_buffer_transfers == 0) { removed this when sending fd packet unconditionally even if there is data to tx
    //                                      removed this when sending fd packet unconditionally even if there is data to
    //                                      tx because there are cases where programs only have one buffer tx (from
    //                                      dram) and in that case we weren't sending the cmd at all (because of
    //                                      is_program continue below)
    // send cmd even if there is no data associated
    internal_::send_fd_packets();  // TODO: AL, is this right?
    // }

    for (uint32_t i = 0; i < num_buffer_transfers; i++) {
        const uint32_t num_pages = command_ptr[2];
        const uint32_t src_buf_type = command_ptr[4];
        const uint32_t dst_buf_type = command_ptr[5];

        bool read_from_sysmem = (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY;
        bool write_to_sysmem = (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY;
        bool tunnel_data = (read_from_sysmem) | (write_to_sysmem & !fwd_path & !is_program);

        if (!tunnel_data) {
            erisc_info->unused_arg2 = 109;
            continue;
        }
        // if (is_program & ((BufferType)src_buf_type == BufferType::DRAM)) {
        //     continue;
        // }

        uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages);

        // erisc_info->unused_arg2 = 800 + num_pages *10 + num_buffer_transfers;
        // erisc_info->unused_arg2 = (uint32_t) (&command_ptr[2]);
        uint32_t num_pages_tunneled = 0;
        while (num_pages_tunneled != num_pages) {
            multicore_eth_cb_wait_front(eth_db_cb_config, num_to_write);
            internal_::send_fd_packets();  // AL: increment since idx to msg sent
            // DPRINT << "src sent data" << ENDL();
            erisc_info->unused_arg1 = 500 + i;
            multicore_eth_cb_pop_front(
                eth_db_cb_config, remote_src_db_cb_config, ((uint64_t)relay_src_noc_encoding << 32), num_to_write);
            num_pages_tunneled += num_to_write;
            // DPRINT << "src-num-remaining: " << (uint32_t)(num_pages - num_pages_tunneled) << ENDL();
            num_to_write = min(num_pages - num_pages_tunneled, producer_consumer_transfer_num_pages);
        }
        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
    }
    noc_semaphore_inc(((uint64_t)relay_src_noc_encoding << 32) | get_semaphore(0), 1);
    noc_async_write_barrier();  // Barrier for now
}

FORCE_INLINE
void eth_tunnel_dst_forward_one_cmd() {
    /*
    // Debug: stall forever if no valid fd cmd
    // TODO: turn into watcher assert
    if (routing_info->fd_buffer_msgs_sent == 0) {
        while(true) {
            RISC_POST_STATUS(0x02140000);
        }
    }*/
    volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));

    bool db_buf_switch = false;
    db_cb_config_t *eth_db_cb_config = get_local_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, false);
    const db_cb_config_t *remote_src_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, false);
    const db_cb_config_t *remote_dst_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, true);
    volatile tt_l1_ptr uint32_t *command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
    volatile tt_l1_ptr CommandHeader *header = (CommandHeader *)command_ptr;

    // Block until consumer can accept new command
    // `num_pages_transferred` tracks whether the remote command processor received all the data
    while (eth_db_semaphore_addr[0] == 0) {
        internal_::risc_context_switch();
    }  // Check that there is space in consumer to send command

    // Send the full command header
    constexpr uint32_t consumer_cmd_base_addr = L1_UNRESERVED_BASE;
    constexpr uint32_t consumer_data_buffer_size =
        (MEM_L1_SIZE - (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t)) - L1_UNRESERVED_BASE);
    uint32_t page_size = header->page_size;
    uint32_t consumer_cb_num_pages = header->producer_cb_num_pages;
    uint32_t consumer_cb_size = header->producer_cb_size;
    uint32_t num_buffer_transfers = header->num_buffer_transfers;
    // if (num_pages_transferred == 0) {   // new command
    eth_program_consumer_cb<command_start_addr, data_buffer_size, consumer_cmd_base_addr, consumer_data_buffer_size>(
        eth_db_cb_config,
        remote_dst_db_cb_config,
        db_buf_switch,
        ((uint64_t)relay_dst_noc_encoding << 32),
        consumer_cb_num_pages,
        page_size,
        consumer_cb_size);
    relay_command<command_start_addr, consumer_cmd_base_addr, consumer_data_buffer_size>(
        db_buf_switch, ((uint64_t)relay_dst_noc_encoding << 32));

    update_producer_consumer_sync_semaphores(
        ((uint64_t)eth_router_noc_encoding << 32),
        ((uint64_t)relay_dst_noc_encoding << 32),
        eth_db_semaphore_addr,
        get_semaphore(0));
    //   }

    // Send the data that was in this packet
    bool is_program = header->is_program_buffer;
    bool fwd_path = header->fwd_path;
    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;  // jump to buffer transfer region
    // if (num_buffer_transfers == 0 | is_program) { removed this when sending fd packet unconditionally from src even
    // if there is data to tx
    internal_::ack_fd_packet();
    // }
    uint32_t producer_consumer_transfer_num_pages = header->producer_router_transfer_num_pages;
    uint32_t l1_consumer_fifo_limit_16B =
        (get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + consumer_cb_size) >> 4;
    uint32_t local_cb_size_16B = header->router_cb_size >> 4;
    uint32_t local_fifo_limit_16B =
        (get_db_buf_addr<command_start_addr, data_buffer_size>(db_buf_switch) >> 4) + local_cb_size_16B;
    for (uint32_t i = 0; i < num_buffer_transfers; i++) {
        const uint32_t num_pages = command_ptr[2];
        const uint32_t src_buf_type = command_ptr[4];
        const uint32_t dst_buf_type = command_ptr[5];

        bool read_from_sysmem = (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY;
        bool write_to_sysmem = (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY;
        bool tunnel_data = (read_from_sysmem) | (write_to_sysmem & !is_program & !fwd_path);

        if (!tunnel_data) {
            // erisc_info->unused_arg2 = 109;
            continue;
        }

        // producer_consumer_transfer_num_pages is the total num of data pages that could fit in a FD packet
        uint32_t num_pages_to_tx = min(num_pages, producer_consumer_transfer_num_pages);
        uint32_t num_pages_transferred = 0;

        // erisc_info->unused_arg2 = 800 + num_pages *10 + num_buffer_transfers;
        // erisc_info->unused_arg2 = (uint32_t) (&command_ptr[2]);
        while (num_pages_transferred != num_pages) {
            // if (num_pages_transferred != 0) { removed this when sending fd packet unconditionally from src even if
            // there is data to tx
            while (routing_info->fd_buffer_msgs_sent != 1) {
                // maybe contesxt switch
                internal_::risc_context_switch();
            }
            // }
            uint32_t src_addr = eth_db_cb_config->rd_ptr_16B << 4;
            uint64_t dst_noc_addr = ((uint64_t)relay_dst_noc_encoding << 32) | (eth_db_cb_config->wr_ptr_16B << 4);
            while (!cb_consumer_space_available(eth_db_cb_config, num_pages_to_tx))
                ;
            noc_async_write(src_addr, dst_noc_addr, page_size * num_pages_to_tx);
            internal_::ack_fd_packet();
            multicore_cb_push_back(
                eth_db_cb_config,
                remote_dst_db_cb_config,
                ((uint64_t)relay_dst_noc_encoding << 32),
                l1_consumer_fifo_limit_16B,
                num_pages_to_tx);
            eth_db_cb_config->rd_ptr_16B += eth_db_cb_config->page_size_16B * num_pages_to_tx;
            if (eth_db_cb_config->rd_ptr_16B >= local_fifo_limit_16B) {
                eth_db_cb_config->rd_ptr_16B -= local_cb_size_16B;
            }
            num_pages_transferred += num_pages_to_tx;
            num_pages_to_tx = min(num_pages - num_pages_transferred, producer_consumer_transfer_num_pages);
        }
        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;  // jump to buffer transfer region
    }
}

FORCE_INLINE
void run_routing() {
    router_init();
    // TODO: maybe split into two FWs? or this may be better to sometimes allow each eth core to do both send and
    // receive of fd packets
    if (my_routing_mode == EthRouterMode::FD_SRC) {
        volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));
        if (eth_db_semaphore_addr[0] != 0) {
            eth_tunnel_src_forward_one_cmd();
        }

    } else if (my_routing_mode == EthRouterMode::FD_DST) {
        if (routing_info->fd_buffer_msgs_sent == 1) {
            eth_tunnel_dst_forward_one_cmd();
        }
    } else if (my_routing_mode == EthRouterMode::SD) {
        // slow dispatch mode
        internal_::risc_context_switch();
    } else {
        while (true) {
            // Debug: stall forever if routing mode is invalid
            // TODO: turn into watcher assert
            RISC_POST_STATUS(0x02130213);
        }
    }
}
