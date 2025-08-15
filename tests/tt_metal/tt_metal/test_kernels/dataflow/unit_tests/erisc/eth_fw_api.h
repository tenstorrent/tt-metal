#pragma once

#include <cstdint>

constexpr uint32_t NUM_SERDES_LANES = 8;

typedef enum {
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
} link_train_status_e;

typedef enum {
    PORT_UNKNOWN,
    PORT_UP,
    PORT_DOWN,
    PORT_UNUSED,
} port_status_e;

typedef struct {
    uint32_t patch : 8;
    uint32_t minor : 8;
    uint32_t major : 8;
    uint32_t unused : 8;
} fw_version_t;

typedef struct {
    uint8_t pcb_type;  // 0
    uint8_t asic_location;
    uint8_t eth_id;
    uint8_t logical_eth_id;
    uint32_t board_id_hi;   // 1
    uint32_t board_id_lo;   // 2
    uint32_t mac_addr_org;  // 3
    uint32_t mac_addr_id;   // 4
    uint32_t spare[2];      // 5-6
    uint32_t ack;           // 7
} chip_info_t;

typedef struct {
    uint32_t bist_mode;  // 0
    uint32_t test_time;  // 1
    // test_time in cycles for bist mode 0 and ms for bist mode 1
    uint32_t error_cnt_nt[NUM_SERDES_LANES];           // 2-9
    uint32_t error_cnt_55t32_nt[NUM_SERDES_LANES];     // 10-17
    uint32_t error_cnt_overflow_nt[NUM_SERDES_LANES];  // 18-25
} serdes_rx_bist_results_t;

typedef struct {
    // Basic status
    uint32_t postcode;                 // 0
    port_status_e port_status;         // 1
    link_train_status_e train_status;  // 2
    uint32_t train_speed;              // 3 - Actual resulting speed from training

    uint32_t spare[28 - 4];  // 4-27

    // Heartbeat
    uint32_t heartbeat[4];  // 28-31
} eth_status_t;

typedef struct {
    uint32_t postcode;           // 0
    uint32_t serdes_inst;        // 1
    uint32_t serdes_lane_mask;   // 2
    uint32_t target_speed;       // 3 - Target speed from the boot params
    uint32_t data_rate;          // 4
    uint32_t data_width;         // 5
    uint32_t spare_main[8 - 6];  // 6-7

    // Training retries
    uint32_t anlt_retry_cnt;  // 8
    uint32_t spare[16 - 9];   // 9-15

    // BIST
    uint32_t bist_mode;       // 16
    uint32_t bist_test_time;  // 17
    // test_time in cycles for bist mode 0 and ms for bist mode 1
    uint32_t bist_err_cnt_nt[NUM_SERDES_LANES];           // 18-25
    uint32_t bist_err_cnt_55t32_nt[NUM_SERDES_LANES];     // 26-33
    uint32_t bist_err_cnt_overflow_nt[NUM_SERDES_LANES];  // 34-41

    uint32_t cdr_unlocked_cnt;        // 42
    uint32_t cdr_unlock_transitions;  // 43

    uint32_t spare2[48 - 44];  // 44-47

    // Training times
    uint32_t man_eq_cmn_pstate_time;      // 48
    uint32_t man_eq_tx_ack_time;          // 49
    uint32_t man_eq_rx_ack_time;          // 50
    uint32_t man_eq_rx_eq_assert_time;    // 51
    uint32_t man_eq_rx_eq_deassert_time;  // 52
    uint32_t anlt_auto_neg_time;          // 53
    uint32_t anlt_link_train_time;        // 54
    uint32_t anlt_retrain_time;           // 55
    uint32_t cdr_lock_time;               // 56
    uint32_t bist_lock_time;              // 57

    uint32_t spare_time[64 - 58];  // 58-63
} serdes_results_t;

typedef struct {
    uint32_t postcode;  // 0

    uint32_t macpcs_retry_cnt;  // 1

    uint32_t spare[24 - 2];  // 2-23

    // Training times
    uint32_t link_up_time;    // 24
    uint32_t chip_info_time;  // 25

    uint32_t spare_time[32 - 26];  // 26-31
} macpcs_results_t;

typedef struct {
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
} eth_live_status_t;

typedef struct {
    uint32_t* send_eth_msg_ptr;           // 0 - Pointer to the send eth msg function
    uint32_t* service_eth_msg_ptr;        // 1 - Pointer to the service eth msg function
    uint32_t* eth_link_status_check_ptr;  // 2 - Pointer to the eth link status check function
    uint32_t spare[16 - 3];               // 3-15
} eth_api_table_t;

typedef struct {
    eth_status_t eth_status;          // 0-31
    serdes_results_t serdes_results;  // 32 - 95
    macpcs_results_t macpcs_results;  // 96 - 127

    eth_live_status_t eth_live_status;  // 128 - 191
    eth_api_table_t eth_api_table;      // 192 - 207

    uint32_t spare[238 - 208];  // 208 - 237

    fw_version_t serdes_fw_ver;  // 238
    fw_version_t eth_fw_ver;     // 239
    chip_info_t local_info;      // 240 - 247
    chip_info_t remote_info;     // 248 - 255
} boot_results_t;

constexpr uint32_t ETH_RESULTS_BASE_ADDR = 0x7CC00;
constexpr uint32_t ETH_RESULTS_BASE_ADDR_LIVE_STATUS =
    ETH_RESULTS_BASE_ADDR + sizeof(eth_status_t) + sizeof(serdes_results_t) + sizeof(macpcs_results_t);
constexpr uint32_t MEM_SYSENG_ETH_API_TABLE = ETH_RESULTS_BASE_ADDR + sizeof(eth_status_t) + sizeof(serdes_results_t) +
                                              sizeof(macpcs_results_t) + sizeof(eth_live_status_t);
