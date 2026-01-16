
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/ethernet/erisc.h"
#include "eth_l1_address_map.h"
#include "noc_nonblocking_api.h"
#include "api/debug/eth_link_status.h"
#include "tt_eth_ss_regs.h"
#include "internal/ethernet/tt_eth_api.h"
inline void RISC_POST_STATUS(uint32_t status) {
    volatile uint32_t* ptr = (volatile uint32_t*)(NOC_CFG(ROUTER_CFG_2));
    ptr[0] = status;
}

static volatile uint32_t* const fabric_postcode_ptr =
    reinterpret_cast<volatile uint32_t*>(eth_l1_mem::address_map::AERISC_FABRIC_POSTCODES_BASE);

#define POSTCODE(status) (*fabric_postcode_ptr = static_cast<uint32_t>(status))

static volatile uint32_t* const fabric_scratch_ptr =
    reinterpret_cast<volatile uint32_t*>(eth_l1_mem::address_map::AERISC_FABRIC_SCRATCH_BASE);

#define ROUTER_SCRATCH_WRITE(id, val) (fabric_scratch_ptr[id]) = val;

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

    // Logical channel ID tagged by the sender. Not required when channels
    // are connected 1:1 (single producer - single consumer)
    volatile uint32_t src_id;

    uint32_t reserved_2;
};

struct erisc_info_t {
    volatile uint32_t launch_user_kernel;
    volatile uint32_t unused_arg0;
    volatile uint32_t unused_arg1;
    volatile uint32_t unused_arg2;
    volatile eth_channel_sync_t
        channels[eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS];  // user_buffer_bytes_sent
};

erisc_info_t* erisc_info = (erisc_info_t*)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
routing_info_t* routing_info = (routing_info_t*)(eth_l1_mem::address_map::ERISC_APP_ROUTING_INFO_BASE);

// Context Switch Config
tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);

extern uint32_t __erisc_jump_table;
volatile uint32_t* RtosTable =
    (volatile uint32_t*)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;

namespace internal_ {

FORCE_INLINE bool eth_txq_is_busy(uint32_t q_num) {
#ifdef ARCH_WORMHOLE
    return eth_txq_reg_read(q_num, ETH_TXQ_CMD) != 0;
#else
    // Due to https://tenstorrent.atlassian.net/browse/BH-55 we don't want to poll STATUS.cmd_ongoing bit too soon after
    // a previous TX. Workaround is to perform any register operation on the same TX queue to slow down successive polls
    eth_txq_reg_read(q_num, ETH_TXQ_CMD);
    return ((eth_txq_reg_read(q_num, ETH_TXQ_STATUS) >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 0x1) != 0;
#endif
}

template <bool ctx_switch = true>
FORCE_INLINE void eth_send_packet(uint32_t q_num, uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words) {
    DEBUG_SANITIZE_ETH(src_word_addr << 4, dest_word_addr << 4, num_words << 4);
    WATCHER_CHECK_ETH_LINK_STATUS();
    while (eth_txq_is_busy(q_num)) {
        // Note, this is overly eager... Kills perf on allgather
        if constexpr (ctx_switch) {
            risc_context_switch();
        }
    }
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_START_ADDR, src_word_addr << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, dest_word_addr << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_SIZE_BYTES, num_words << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_DATA);
}

FORCE_INLINE
void eth_send_packet_unsafe(uint32_t q_num, uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words) {
    DEBUG_SANITIZE_ETH(src_word_addr << 4, dest_word_addr << 4, num_words << 4);
    WATCHER_CHECK_ETH_LINK_STATUS();
    ASSERT(!eth_txq_is_busy(q_num));
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_START_ADDR, src_word_addr << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, dest_word_addr << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_SIZE_BYTES, num_words << 4);
    eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_DATA);
}

FORCE_INLINE
void eth_send_packet_bytes_unsafe(uint32_t q_num, uint32_t src_addr, uint32_t dest_addr, uint32_t num_bytes) {
    DEBUG_SANITIZE_ETH(src_addr, dest_addr, num_bytes);
    WATCHER_CHECK_ETH_LINK_STATUS();
    ASSERT(eth_txq_reg_read(q_num, ETH_TXQ_CMD) == 0);
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_START_ADDR, src_addr);
    eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, dest_addr);
    eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_SIZE_BYTES, num_bytes);
    eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_DATA);
}

template <bool ctx_switch = true>
FORCE_INLINE void eth_write_remote_reg(uint32_t q_num, uint32_t reg_addr, uint32_t val) {
    WATCHER_CHECK_ETH_LINK_STATUS();
    while (eth_txq_is_busy(q_num)) {
        if constexpr (ctx_switch) {
            risc_context_switch();
        }
    }
    eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, reg_addr);
    eth_txq_reg_write(q_num, ETH_TXQ_REMOTE_REG_DATA, val);
    eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_REG);
}
FORCE_INLINE
void eth_write_remote_reg_no_txq_check(uint32_t q_num, uint32_t reg_addr, uint32_t val) {
    WATCHER_CHECK_ETH_LINK_STATUS();
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
        while (!noc_cmd_buf_ready(n, NCRISC_AT_CMD_BUF));
    }
    noc_fast_write_dw_inline<DM_DEDICATED_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        1 << REMOTE_DEST_BUF_WORDS_FREE_INC,
        dispatch_addr,
        0xF,  // byte-enable
        NOC_UNICAST_WRITE_VC,
        false,  // mcast
        true    // posted
    );
}

}  // namespace internal_

FORCE_INLINE
void run_routing() {
    // router_init();
    // TODO: maybe split into two FWs? or this may be better to sometimes allow each eth core to do both send and
    // receive of fd packets
    internal_::risc_context_switch();
}

FORCE_INLINE
void run_routing_without_noc_sync() { internal_::risc_context_switch_without_noc_sync(); }
