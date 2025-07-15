// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define MEM_SYSENG_ETH_MSG_STATUS_MASK 0xFFFF0000
#define MEM_SYSENG_ETH_MSG_CALL 0xCA110000
#define MEM_SYSENG_ETH_MSG_DONE 0xD0E50000
#define MEM_SYSENG_ETH_MSG_TYPE_MASK 0x0000FFFF
#define MEM_SYSENG_ETH_MSG_LINK_STATUS_CHECK 0x0001
#define MEM_SYSENG_ETH_MSG_RELEASE_CORE 0x0002
#define MEM_SYSENG_ETH_MAILBOX_ADDR 0x7D000
#define MEM_SYSENG_ETH_MAILBOX_NUM_ARGS 3
#define MEM_SYSENG_ETH_HEARTBEAT 0x7CC70
#define MEM_SYSENG_ETH_RETRAIN_COUNT 0x7CE00
#define MEM_SYSENG_ETH_API_TABLE 0x7CF00

enum eth_mailbox_e : uint32_t {
    MAILBOX_HOST,
    MAILBOX_RISC1,
    MAILBOX_CMFW,
    MAILBOX_OTHER,
    NUM_ETH_MAILBOX,
};

struct eth_api_table_t {
    uint32_t* send_eth_msg_ptr;           // 0 - Pointer to the send eth msg function
    uint32_t* service_eth_msg_ptr;        // 1 - Pointer to the service eth msg function
    uint32_t* eth_link_status_check_ptr;  // 2 - Pointer to the eth link status check function

    uint32_t spare[16 - 3];  // 3-15
};

struct eth_mailbox_t {
    uint32_t msg;     // 0 - Message type. Defined with MEM_SYSENG_ETH_MSG_* macros
    uint32_t arg[3];  // 1-3 - Arguments to the message (not all need to be used)
};

struct all_eth_mailbox_t {
    eth_mailbox_t mailbox[4];  // 0-16 - 4 mailbox entries, 0 - Host, 1 - RSIC1, 2 - CMFW, 3 - Other
};

struct eth_live_status_t {
    uint32_t retrain_count;  // 0
    uint32_t rx_link_up;     // 1 - MAC/PCS RX Link Up

    uint32_t spare[8 - 2];  // 2-7

    // Snapshot registers
    uint64_t frames_txd;          // 8,9 - Cumulative TX Packets Transmitted count
    uint64_t frames_txd_ok;       // 10,11 - Cumulative TX Packets Transmitted OK count
    uint64_t frames_txd_badfcs;   // 12,13 - Cumulative TX Packets Transmitted with BAD FCS count
    uint64_t bytes_txd;           // 14,15 - Cumulative TX Bytes Transmitted count
    uint64_t bytes_txd_ok;        // 16,17 - Cumulative TX Bytes Transmitted OK count
    uint64_t bytes_txd_badfcs;    // 18,19 - Cumulative TX Bytes Transmitted with BAD FCS count
    uint64_t frames_rxd;          // 20,21 - Cumulative Packets Received count
    uint64_t frames_rxd_ok;       // 22,23 - Cumulative Packets Received OK count
    uint64_t frames_rxd_badfcs;   // 24,25 - Cumulative Packets Received with BAD FCS count
    uint64_t frames_rxd_dropped;  // 26,27 - Cumulative Dropped Packets Received count
    uint64_t bytes_rxd;           // 28,29 - Cumulative Bytes received count
    uint64_t bytes_rxd_ok;        // 30,31 - Cumulative Bytes received OK count
    uint64_t bytes_rxd_badfcs;    // 32,33 - Cumulative Bytes received with BAD FCS count
    uint64_t bytes_rxd_dropped;   // 34,35 - Cumulative Bytes received and dropped count
    uint64_t corr_cw;             // 36,37 - Cumulative Corrected Codeword count
    uint64_t uncorr_cw;           // 38,39 - Cumulative Uncorrected Codeword count

    uint32_t spare2[64 - 40];  // 40-63
};
