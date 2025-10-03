// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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
#define MEM_SYSENG_ETH_API_TABLE 0x7CF00
#define MEM_SYSENG_BOOT_RESULTS_BASE 0x7CC00
#define NUM_SERDES_LANES 8

#define ETH_RISC_CTRL_A_INTERRUPT_MODE_0__REG_ADDR 0xFFB14020
#define ETH_RISC_NUM_INTERRUPT_VECS 5
#define ETH_CORE_A_ETH_CTRL_A_PTP_TIMER_A_CFR_TIMER_LO_REG_ADDR 0xFFB98850
#define ETH_CORE_A_ETH_CTRL_A_PTP_TIMER_A_CFR_TIMER_HI_REG_ADDR 0xFFB98854
#define ETH_CLOCK_CYCLE_1MS 1000000
#define ETH_UPDATE_LINK_STATUS_INTERVAL_MS 1000

enum link_train_status_e : uint32_t {
    LINK_TRAIN_TRAINING,
    LINK_TRAIN_SKIP,
    LINK_TRAIN_PASS,
    LINK_TRAIN_INT_LB,
    LINK_TRAIN_EXT_LB,
    LINK_TRAIN_TIMEOUT_MANUAL_EQ,
    LINK_TRAIN_TIMEOUT_ANLT,
    LINK_TRAIN_TIMEOUT_CDR_LOCK,
    LINK_TRAIN_TIMEOUT_BIST_LOCK,
    LINK_TRAIN_TIMEOUT_LINK_UP,
    LINK_TRAIN_TIMEOUT_CHIP_INFO,
};

enum port_status_e : uint32_t {
    PORT_UNKNOWN,
    PORT_UP,
    PORT_DOWN,
    PORT_UNUSED,
};

struct fw_version_t {
    uint32_t patch : 8;
    uint32_t minor : 8;
    uint32_t major : 8;
    uint32_t unused : 8;
};

struct chip_info_t {
    uint8_t pcb_type;
    uint8_t asic_location;
    uint8_t eth_id;
    uint8_t logical_eth_id;
    uint32_t board_id_hi;
    uint32_t board_id_lo;
    uint32_t mac_addr_org;
    uint32_t mac_addr_id;
    uint32_t spare[2];
    uint32_t ack;
};

struct serdes_rx_bist_results_t {
    uint32_t bist_mode;
    uint32_t test_time; // Test time in cycles for bist mode 0 and ms for bist mode 1
    uint32_t error_cnt_nt[NUM_SERDES_LANES];
    uint32_t error_cnt_55t32_nt[NUM_SERDES_LANES];
    uint32_t error_cnt_overflow_nt[NUM_SERDES_LANES];
};

struct eth_status_t {
    // Basic status
    uint32_t postcode;
    port_status_e port_status;
    link_train_status_e train_status;
    uint32_t train_speed;   // Actual resulting speed from training

    uint32_t spare[28 - 4];

    // Heartbeat
    uint32_t heartbeat[4];
};

struct serdes_results_t {
    uint32_t postcode;
    uint32_t serdes_inst;
    uint32_t serdes_lane_mask;
    uint32_t target_speed;       // Target speed from the boot params
    uint32_t data_rate;
    uint32_t data_width;
    uint32_t spare_main[8 - 6];

    // Training retries
    uint32_t anlt_retry_cnt;
    uint32_t spare[16 - 9];

    // BIST
    uint32_t bist_mode;
    uint32_t bist_test_time; // Test time in cycles for bist mode 0 and ms for bist mode 1
    uint32_t bist_err_cnt_nt[NUM_SERDES_LANES];
    uint32_t bist_err_cnt_55t32_nt[NUM_SERDES_LANES];
    uint32_t bist_err_cnt_overflow_nt[NUM_SERDES_LANES];

    uint32_t cdr_unlocked_cnt;
    uint32_t cdr_unlock_transitions;

    uint32_t spare2[48 - 44];

    // Training times
    uint32_t man_eq_cmn_pstate_time;
    uint32_t man_eq_tx_ack_time;
    uint32_t man_eq_rx_ack_time;
    uint32_t man_eq_rx_eq_assert_time;
    uint32_t man_eq_rx_eq_deassert_time;
    uint32_t anlt_auto_neg_time;
    uint32_t anlt_link_train_time;
    uint32_t anlt_retrain_time;
    uint32_t cdr_lock_time;
    uint32_t bist_lock_time;

    uint32_t spare_time[64 - 58];
};

struct macpcs_results_t {
    uint32_t postcode;

    uint32_t macpcs_retry_cnt;

    uint32_t spare[24 - 2];

    // Training times
    uint32_t link_up_time;
    uint32_t chip_info_time;

    uint32_t spare_time[32 - 26];
};

struct eth_live_status_t {
    uint32_t retrain_count;
    uint32_t rx_link_up;     // MAC/PCS RX Link Up

    uint32_t spare[8 - 2];

    // Snapshot registers
    uint64_t frames_txd;          // Cumulative TX Packets Transmitted count
    uint64_t frames_txd_ok;       // Cumulative TX Packets Transmitted OK count
    uint64_t frames_txd_badfcs;   // Cumulative TX Packets Transmitted with BAD FCS count
    uint64_t bytes_txd;           // Cumulative TX Bytes Transmitted count
    uint64_t bytes_txd_ok;        // Cumulative TX Bytes Transmitted OK count
    uint64_t bytes_txd_badfcs;    // Cumulative TX Bytes Transmitted with BAD FCS count
    uint64_t frames_rxd;          // Cumulative Packets Received count
    uint64_t frames_rxd_ok;       // Cumulative Packets Received OK count
    uint64_t frames_rxd_badfcs;   // Cumulative Packets Received with BAD FCS count
    uint64_t frames_rxd_dropped;  // Cumulative Dropped Packets Received count
    uint64_t bytes_rxd;           // Cumulative Bytes received count
    uint64_t bytes_rxd_ok;        // Cumulative Bytes received OK count
    uint64_t bytes_rxd_badfcs;    // Cumulative Bytes received with BAD FCS count
    uint64_t bytes_rxd_dropped;   // Cumulative Bytes received and dropped count
    uint64_t corr_cw;             // Cumulative Corrected Codeword count
    uint64_t uncorr_cw;           // Cumulative Uncorrected Codeword count

    uint32_t spare2[64 - 40];
};

struct eth_api_table_t {
    uint32_t* send_eth_msg_ptr;           // Pointer to the send eth msg function
    uint32_t* service_eth_msg_ptr;        // Pointer to the service eth msg function
    uint32_t* eth_link_status_check_ptr;  // Pointer to the eth link status check function
    uint32_t* eth_flush_icache_ptr;       // Pointer to the eth flush icache function
    uint32_t spare[16 - 4];
};

enum eth_mailbox_e : uint32_t {
    MAILBOX_HOST,
    MAILBOX_RISC1,
    MAILBOX_CMFW,
    MAILBOX_OTHER,
    NUM_ETH_MAILBOX,
};

struct eth_mailbox_t {
    uint32_t msg;     // Message type. Defined with MEM_SYSENG_ETH_MSG_* macros
    uint32_t arg[3];  // Arguments to the message (not all need to be used)
};

struct all_eth_mailbox_t {
    eth_mailbox_t mailbox[4];  // 4 mailbox entries, 0 - Host, 1 - RSIC1, 2 - CMFW, 3 - Other
};

struct boot_results_t {
    eth_status_t eth_status;
    serdes_results_t serdes_results;
    macpcs_results_t macpcs_results;

    eth_live_status_t eth_live_status;
    eth_api_table_t eth_api_table;

    uint32_t spare[238 - 208];

    fw_version_t serdes_fw_ver;
    fw_version_t eth_fw_ver;
    chip_info_t local_info;
    chip_info_t remote_info;
};

#define MEM_SYSENG_ETH_STATUS (MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, eth_status))
#define MEM_SYSENG_ETH_LIVE_STATUS (MEM_SYSENG_BOOT_RESULTS_BASE + offsetof(boot_results_t, eth_live_status))

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "tt_metal/hw/inc/risc_common.h"
#include "tt_metal/hw/inc/ethernet/tt_eth_api.h"
#include "dev_msgs.h"

uint64_t get_next_link_status_check_timestamp() {
    return *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(GET_MAILBOX_ADDRESS_DEV(link_status_check_timestamp));
}

void update_next_link_status_check_timestamp() {
#if defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 0 && defined(ENABLE_2_ERISC_MODE)
    uint64_t timestamp = eth_read_wall_clock() + (ETH_CLOCK_CYCLE_1MS * ETH_UPDATE_LINK_STATUS_INTERVAL_MS);
    *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(GET_MAILBOX_ADDRESS_DEV(link_status_check_timestamp)) = timestamp;
#endif
}

void eth_set_interrupt_mode(uint32_t interrupt_number, uint32_t mode_val) {
#if defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 0 && defined(ENABLE_2_ERISC_MODE)
    auto reg_ptr = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        ETH_RISC_CTRL_A_INTERRUPT_MODE_0__REG_ADDR + (4 * interrupt_number));
    *reg_ptr = mode_val;
#endif
}

void disable_interrupts() {
#if defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 0 && defined(ENABLE_2_ERISC_MODE)
    for (uint32_t i = 0; i < ETH_RISC_NUM_INTERRUPT_VECS; i++) {
        eth_set_interrupt_mode(i, 0);
    }
#endif
}

FORCE_INLINE bool is_link_up() {
#if defined(COMPILE_FOR_AERISC) && (COMPILE_FOR_AERISC == 0) && !defined(ENABLE_2_ERISC_MODE)
    // Collect current link states
    // TODO: Until erisc0 is enabled, use MAILBOX_RISC1 for link status check. When both riscs are enabled, assign one
    // to use MAILBOX_OTHER Sending msgs to mailbox described in:
    // https://tenstorrent.atlassian.net/wiki/spaces/syseng/pages/904626206/ETH+SW+APIs
    constexpr uint32_t risc1_mailbox_addr = MEM_SYSENG_ETH_MAILBOX_ADDR + (MAILBOX_RISC1 * sizeof(eth_mailbox_t));

    volatile tt_l1_ptr uint32_t* risc1_mailbox_msg_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(risc1_mailbox_addr + offsetof(eth_mailbox_t, msg));
    volatile tt_l1_ptr uint32_t* risc1_mailbox_arg0_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(risc1_mailbox_addr + offsetof(eth_mailbox_t, arg[0]));
    uint32_t risc1_mailbox_val = *risc1_mailbox_msg_ptr;
    // Make sure mailbox is free to accept a new message
    do {
        invalidate_l1_cache();
        risc1_mailbox_val = *risc1_mailbox_msg_ptr;
    } while (((risc1_mailbox_val & MEM_SYSENG_ETH_MSG_STATUS_MASK) != MEM_SYSENG_ETH_MSG_DONE) &&
             risc1_mailbox_val != 0);

    // This tells link status check to avoid copying results into another location in L1
    *risc1_mailbox_arg0_ptr = 0xFFFFFFFF;

    // Send msg to get the link status
    *risc1_mailbox_msg_ptr = MEM_SYSENG_ETH_MSG_CALL | MEM_SYSENG_ETH_MSG_LINK_STATUS_CHECK;

    // Wait until the msg was serviced
    do {
        invalidate_l1_cache();
        risc1_mailbox_val = *risc1_mailbox_msg_ptr;
    } while ((risc1_mailbox_val & MEM_SYSENG_ETH_MSG_STATUS_MASK) == MEM_SYSENG_ETH_MSG_CALL);

    auto link_status = (volatile eth_live_status_t*)(MEM_SYSENG_ETH_LIVE_STATUS);
    return link_status->rx_link_up == 1;
#else
    auto link_status = (volatile eth_live_status_t*)(MEM_SYSENG_ETH_LIVE_STATUS);
    if (link_status->rx_link_up != 1) {
        // erisc0 checks link status and does retraining.  If erisc1 detects link down, wait a bit and check again
        eth_wait_cycles(3 << 30);
        eth_wait_cycles(2 << 30);
    }
    return link_status->rx_link_up == 1;
#endif
}

FORCE_INLINE bool is_port_up() {
    invalidate_l1_cache();
    return ((eth_status_t*)(MEM_SYSENG_ETH_STATUS))->port_status == port_status_e::PORT_UP;
}

static void service_eth_msg() {
#if defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 0 && defined(ENABLE_2_ERISC_MODE)
    invalidate_l1_cache();
    reinterpret_cast<void (*)()>((uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->service_eth_msg_ptr))();
#endif
}

static void update_boot_results_eth_link_status_check() {
#if defined(COMPILE_FOR_AERISC) && COMPILE_FOR_AERISC == 0 && defined(ENABLE_2_ERISC_MODE)
    uint64_t curr_timestamp = eth_read_wall_clock();
    uint64_t next_timestamp = get_next_link_status_check_timestamp();
    // Debounce to only be called at every interval
    // wrap-around safe comparison. calling this too many times can result in link
    // instability
    if ((curr_timestamp - next_timestamp) < (UINT64_MAX / 2)) {
        invalidate_l1_cache();
        reinterpret_cast<void (*)(uint32_t)>(
            (uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->eth_link_status_check_ptr))(0xFFFFFFFF);

        update_next_link_status_check_timestamp();
    }
#endif
}

#endif
