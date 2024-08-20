
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "erisc.h"
#include "eth_l1_address_map.h"
#include "noc_nonblocking_api.h"

inline void RISC_POST_STATUS(uint32_t status) {
    volatile uint32_t *ptr = (volatile uint32_t *)(NOC_CFG(ROUTER_CFG_2));
    ptr[0] = status;
}

struct eth_channel_sync_t {
    // Do not reorder fields without also updating the corresponding APIs that use
    // any of them

    // Notifies how many bytes were sent by the sender. Receiver resets this to 0
    // and sends the change to sender to signal second level ack, that the
    // receiver buffer can be written into
    volatile uint32_t bytes_sent;

    // First level ack that signals to sender that the payload was received by receiver,
    // indicating that sender can reuse the sender side buffer safely.
    volatile uint32_t receiver_ack;
    uint32_t reserved_1;
    uint32_t reserved_2;
};

struct erisc_info_t {
    volatile uint32_t launch_user_kernel;
    volatile uint32_t unused_arg0;
    volatile uint32_t unused_arg1;
    volatile uint32_t unused_arg2;
    volatile eth_channel_sync_t channels[eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS]; // user_buffer_bytes_sent
};

erisc_info_t *erisc_info = (erisc_info_t *)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
routing_info_t *routing_info = (routing_info_t *)(eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE);

// Context Switch Config
tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);

extern uint32_t __erisc_jump_table;
volatile uint32_t *RtosTable =
    (volatile uint32_t *)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;

namespace internal_ {

FORCE_INLINE bool eth_txq_is_busy(uint32_t q_num) {
    return eth_txq_reg_read(q_num, ETH_TXQ_CMD) != 0;
}

FORCE_INLINE
void eth_send_packet(uint32_t q_num, uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words) {
    while (eth_txq_reg_read(q_num, ETH_TXQ_CMD) != 0) {
        // Note, this is overly eager... Kills perf on allgather
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
    DEBUG_SANITIZE_NOC_ADDR(noc_index, dispatch_addr, 4);
    noc_fast_atomic_increment(
        noc_index, NCRISC_AT_CMD_BUF, dispatch_addr, NOC_UNICAST_WRITE_VC, 1, 31 /*wrap*/, false /*linked*/);
}

}  // namespace internal_

FORCE_INLINE
void run_routing() {
    // router_init();
    // TODO: maybe split into two FWs? or this may be better to sometimes allow each eth core to do both send and
    // receive of fd packets
    internal_::risc_context_switch();
}
