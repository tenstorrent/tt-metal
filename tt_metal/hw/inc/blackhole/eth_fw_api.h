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
