// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// For MEM_AERISC_RESUME_PHASE_BASE (the reserved L1 word for the resume-phase debug code).
// dev_mem_map.h is pure preprocessor and host/device/linker safe, so including it here is fine for
// both the device kernel build and the host HAL translation units that include this header.
#include "dev_mem_map.h"

#if defined(COMPILE_FOR_AERISC)
// For WATCHER_RING_BUFFER_PUSH(), used by fabric_dbg_ringbuf_push_txrx_counts(). The macro is a no-op
// unless the watcher is enabled (TT_METAL_WATCHER), so this costs nothing in production. Guarded to device builds
// because host translation units (e.g. the HAL) also include this header and lack the ring-buffer
// include path / build flags.
#include "api/debug/ring_buffer.h"
// For eth_enable_packet_mode(), called in the CONFIG-RESTORE block to re-apply full packet-mode
// config after a retrain. Guarded for the same reason as ring_buffer.h above.
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h"
#endif

#define MEM_SYSENG_ETH_MSG_STATUS_MASK 0xFFFF0000
#define MEM_SYSENG_ETH_MSG_CALL 0xCA110000
#define MEM_SYSENG_ETH_MSG_DONE 0xD0E50000
#define MEM_SYSENG_ETH_MSG_TYPE_MASK 0x0000FFFF
#define MEM_SYSENG_ETH_MSG_LINK_STATUS_CHECK 0x0001
#define MEM_SYSENG_ETH_MSG_RELEASE_CORE 0x0002
#define MEM_SYSENG_ETH_MSG_PORT_REINIT_MACPCS 0x0006
#define MEM_SYSENG_ETH_MSG_PORT_ACTION 0x0009
#define MEM_SYSENG_ETH_MSG_DYNAMIC_NOC_INIT 0x000F
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
#define ETH_CORE_A_ETH_CTRL_A_PCS_STATUS_REG_ADDR 0xFFB9800C
#define ETH_CORE_A_ETH_CTRL_A_ERR_STAT_REG_ADDR 0xFFB980D8
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
    LINK_TRAIN_PRBS,
    LINK_TRAIN_REQUESTED_DOWN,
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
    uint32_t asic_id_hi;
    uint32_t asic_id_lo;
    uint32_t req_ack;
};

static_assert(sizeof(chip_info_t) == 32, "chip_info_t size is not 32 bytes");

struct eth_status_t {
    // Basic status
    uint32_t postcode;
    port_status_e port_status;
    link_train_status_e train_status;
    uint32_t train_speed;  // Actual resulting speed from training

    uint32_t spare[28 - 4];

    // Heartbeat
    uint32_t heartbeat[4];
};

static_assert(sizeof(eth_status_t) == 128, "eth_status_t size is not 128 bytes");

struct serdes_results_t {
    uint32_t postcode;
    uint32_t serdes_inst;
    uint32_t serdes_lane_mask;
    uint32_t target_speed;  // Target speed from the boot params
    uint32_t data_rate;
    uint32_t data_width;
    uint32_t spare_main[8 - 6];

    // Training retries
    uint32_t anlt_retry_cnt;
    uint32_t manual_eq_retry_cnt;

    // LCPLL
    uint32_t lcpll_lock_fail_cnt;
    uint32_t spare[16 - 11];

    // BIST
    uint32_t bist_mode;
    uint32_t bist_test_time;  // Test time in cycles for bist mode 0 and ms for bist mode 1
    uint32_t bist_err_cnt_lo[NUM_SERDES_LANES];
    uint32_t bist_err_cnt_hi[NUM_SERDES_LANES];
    uint32_t bist_err_cnt_overflow_nt[NUM_SERDES_LANES];

    uint32_t cdr_unlocked_cnt;
    uint32_t cdr_unlock_transitions;

    uint32_t initial_serdes_init;
    uint32_t serdes_reset_status;
    uint32_t serdes_lane_reset_status;

    uint32_t host_msg;  // Communication field for host/firmware handshake

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
    uint32_t man_eq_sigdet_time;
    uint32_t lcpll_check_time;

    uint32_t spare_time[62 - 60];
    uint32_t serdes_reset_deassert_timestamp_hi;
    uint32_t serdes_reset_deassert_timestamp_lo;
};

static_assert(sizeof(serdes_results_t) == 256, "serdes_results_t size is not 256 bytes");

struct macpcs_results_t {
    uint32_t postcode;

    uint32_t macpcs_retry_cnt;
    uint32_t eth_cntrl_int;

    uint32_t spare[24 - 3];

    // Training times
    uint32_t link_up_time;
    uint32_t chip_info_time;

    uint32_t spare_time[30 - 26];
    uint32_t macpcs_reset_deassert_timestamp_hi;
    uint32_t macpcs_reset_deassert_timestamp_lo;
};

static_assert(sizeof(macpcs_results_t) == 128, "macpcs_results_t size is not 128 bytes");

struct eth_live_status_t {
    uint32_t retrain_count;
    uint32_t rx_link_up;  // MAC/PCS RX Link Up
    uint32_t link_flap_count;        // Link Flap Count
    uint32_t link_poll_alive_count;  // Link Poll Alive Count
    uint32_t spare[8 - 4];

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

    // TX/RX Queue registers
    uint64_t txq0_resend_cnt;  // Cumulative Packet Reset count on TXQ0
    uint64_t txq1_resend_cnt;  // Cumulative Packet Reset count on TXQ1
    uint64_t txq2_resend_cnt;  // Cumulative Packet Reset count on TXQ2
    uint64_t rxq0_pkt_drop;    // Cumulative Packet Drop count on RXQ0
    uint64_t rxq1_pkt_drop;    // Cumulative Packet Drop count on RXQ1
    uint64_t rxq2_pkt_drop;    // Cumulative Packet Drop count on RXQ2

    uint32_t spare2[64 - 52];  // 52-63
};

struct eth_api_table_t {
    uint32_t* send_eth_msg_ptr;           // Pointer to the send eth msg function
    uint32_t* service_eth_msg_ptr;        // Pointer to the service eth msg function
    uint32_t* eth_link_status_check_ptr;  // Pointer to the eth link status check function
    uint32_t* eth_dynamic_noc_init_ptr;   // Pointer to the eth dynamic noc init function
    uint32_t* eth_link_recovery_ptr;      // Pointer to the eth link recovery function

    uint32_t spare[16 - 5];  // 5-15
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

static_assert(sizeof(all_eth_mailbox_t) == 64, "all_eth_mailbox_t size is not 64 bytes");

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
#include "internal/tt-1xx/risc_common.h"
#include "internal/ethernet/tt_eth_api.h"
#include "hostdev/dev_msgs.h"

uint64_t get_next_link_status_check_timestamp() {
    return *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(GET_MAILBOX_ADDRESS_DEV(link_status_check_timestamp));
}

void update_next_link_status_check_timestamp() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    uint64_t timestamp = eth_read_wall_clock() + (ETH_CLOCK_CYCLE_1MS * ETH_UPDATE_LINK_STATUS_INTERVAL_MS);
    *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(GET_MAILBOX_ADDRESS_DEV(link_status_check_timestamp)) = timestamp;
#endif
}

void eth_set_interrupt_mode(uint32_t interrupt_number, uint32_t mode_val) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    auto reg_ptr = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        ETH_RISC_CTRL_A_INTERRUPT_MODE_0__REG_ADDR + (4 * interrupt_number));
    *reg_ptr = mode_val;
#endif
}

void disable_interrupts() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    for (uint32_t i = 0; i < ETH_RISC_NUM_INTERRUPT_VECS; i++) {
        eth_set_interrupt_mode(i, 0);
    }
#endif
}

FORCE_INLINE bool is_link_up() {
    auto pcs_status_ptr = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(ETH_CORE_A_ETH_CTRL_A_PCS_STATUS_REG_ADDR);
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 1)
    return *pcs_status_ptr == 1;
#else
    if (*pcs_status_ptr != 1) {
        // erisc0 checks link status and does retraining. If link down is detected, wait a bit and check again
        eth_wait_cycles(3 << 30);
        eth_wait_cycles(2 << 30);
    }
    return *pcs_status_ptr == 1;
#endif
}

FORCE_INLINE void base_fw_dynamic_noc_local_state_init() {
    // Reinitialize the dynamic NOC counters in base firmware
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 1)
    constexpr uint32_t risc1_mailbox_addr = MEM_SYSENG_ETH_MAILBOX_ADDR + (MAILBOX_RISC1 * sizeof(eth_mailbox_t));

    volatile tt_l1_ptr uint32_t* risc1_mailbox_msg_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(risc1_mailbox_addr + offsetof(eth_mailbox_t, msg));
    uint32_t risc1_mailbox_val = *risc1_mailbox_msg_ptr;

    // Make sure mailbox is free to accept a new message
    do {
        invalidate_l1_cache();
        risc1_mailbox_val = *risc1_mailbox_msg_ptr;
    } while (((risc1_mailbox_val & MEM_SYSENG_ETH_MSG_STATUS_MASK) != MEM_SYSENG_ETH_MSG_DONE) &&
             risc1_mailbox_val != 0);

    *risc1_mailbox_msg_ptr = MEM_SYSENG_ETH_MSG_CALL | MEM_SYSENG_ETH_MSG_DYNAMIC_NOC_INIT;

    do {
        invalidate_l1_cache();
        risc1_mailbox_val = *risc1_mailbox_msg_ptr;
    } while ((risc1_mailbox_val & MEM_SYSENG_ETH_MSG_STATUS_MASK) == MEM_SYSENG_ETH_MSG_CALL);
#else
    // Directly call the function on ERISC0. No need to switch to base firmware.
    reinterpret_cast<void (*)()>(
        (uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->eth_dynamic_noc_init_ptr))();
#endif
}

FORCE_INLINE bool is_port_up() {
    invalidate_l1_cache();
    return ((eth_status_t*)(MEM_SYSENG_ETH_STATUS))->port_status == port_status_e::PORT_UP;
}

static void service_eth_msg() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    invalidate_l1_cache();
    reinterpret_cast<void (*)()>((uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->service_eth_msg_ptr))();
#endif
}

static void update_boot_results_eth_link_status_check() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
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

// NOTE: the recovery / link-status ring-buffer marker codes (formerly 0xD09D/0x600D/0xCA11ED/0xDEAD)
// were removed. The watcher ring buffer is now used to log the live TX packet count on every context
// switch instead (see fabric_dbg_ringbuf_push_txrx_counts below), so we can watch TX/RX during a run.

// ---- Resume-phase debug word ----------------------------------------------------------------------
// A single uint32 in a reserved L1 slot (MEM_AERISC_RESUME_PHASE_BASE) that active ERISC0 stamps as it
// moves through the link-down -> recover -> resume-traffic sequence. Read it back from L1 post-mortem
// (or live) to see how far a recovery got:
//   stuck at RETRAIN_ENTER -> wedged inside the FW recovery call;
//   stuck at RETRAIN_DONE  -> recovered but the router never sent again (TX did not resume);
//   stuck at FIRST_TX      -> sent but never received a packet back (RX did not resume);
//   reaches FIRST_RX       -> traffic resumed both directions.
// Values are monotonic within one recovery cycle; a fresh link-down restamps RETRAIN_ENTER, so the
// word always reflects the most recent recovery attempt. Relies on TT_METAL_CLEAR_L1=1 for a 0 start.
constexpr uint32_t RESUME_PHASE_RETRAIN_ENTER = 0x5E5E0001;  // about to call the FW link-recovery entry point
constexpr uint32_t RESUME_PHASE_RETRAIN_DONE = 0x5E5E0002;   // FW recovery returned (link retrained)
constexpr uint32_t RESUME_PHASE_FIRST_TX = 0x5E5E0003;       // first packet sent after retrain
constexpr uint32_t RESUME_PHASE_FIRST_RX = 0x5E5E0004;       // first packet received after the first post-retrain TX

// [FREEZE-PROBE] Dense post-retrain checkpoints. Unlike the FIRST_TX/RX advances above, these are
// UNCONDITIONAL set_resume_phase overwrites, so the word always holds the LAST checkpoint the single
// allowed post-retrain iteration reached -- a hang parks the word at the stuck point.
//   TX (sender speedy step): 0x...11-18   RX (receiver speedy step): 0x...21-26   main loop: 0x...31-33
constexpr uint32_t RESUME_PHASE_TX_STEP_ENTER = 0x5E5E0011;      // entered sender step
constexpr uint32_t RESUME_PHASE_TX_ISSUE = 0x5E5E0014;           // about to issue eth_send_packet (payload)
constexpr uint32_t RESUME_PHASE_TX_POSTSEND_DRAIN = 0x5E5E0015;  // about to spin on eth_txq drain (prime suspect)
constexpr uint32_t RESUME_PHASE_TX_SEND_DONE = 0x5E5E0017;       // payload out + remote sent-count bumped
constexpr uint32_t RESUME_PHASE_RX_STEP_ENTER = 0x5E5E0021;      // entered receiver step
constexpr uint32_t RESUME_PHASE_RX_LOCAL_WRITE = 0x5E5E0023;     // about to NoC-write packet to local chip
constexpr uint32_t RESUME_PHASE_RX_FLUSH_POLL = 0x5E5E0024;      // about to poll per-trid NoC-write flush
constexpr uint32_t RESUME_PHASE_LOOP_TOP = 0x5E5E0031;           // top of a main-loop iteration
constexpr uint32_t RESUME_PHASE_CTX_SWITCH = 0x5E5E0032;         // about to enter coordinated context switch
constexpr uint32_t RESUME_PHASE_LOOP_BOTTOM = 0x5E5E0033;        // reached bottom of a main-loop iteration

// [WAS_RETRAINED] Edge-triggered freeze flag (per-core; only ERISC0's recovery sets it).
//   0 = normal running; 1 = a retrain just succeeded (link came UP after being DOWN);
//   the main loop advances 1 -> 2 after exactly one iteration, then freezes -- snapshotting the
//   resume-phase word right after retrain so we can see where the loop is stuck.
// volatile so the main loop reliably observes the recovery-set value.
inline volatile uint32_t was_retrained = 0;

// Number of post-retrain main-loop iterations to allow before the freeze gate stops the loop.
// was_retrained is 1 on the retrain edge and ++ each iteration bottom, so it takes values 1..N across
// the N allowed iterations; the gate freezes once it exceeds N (i.e. at N+1).
// 0xFFFFFFFF == effectively NEVER freeze -> free-run after retrain so we can watch (over a long run,
// via tx_count / the ring buffer) whether the router ever resumes sending. Lower it (e.g. 1 or 5) to
// re-enable the freeze snapshot of where the loop is right after retrain.
constexpr uint32_t WAS_RETRAINED_FREEZE_AFTER_N_ITERS = 0xFFFFFFFF;

// [TX-COUNT] Free-running 32-bit count of successful packets sent by ERISC0 over the eth link, stored
// in word[1] of the debug slot (MEM_AERISC_RESUME_PHASE_BASE + 4). Poll it via brxy (read 2 words at
// the slot base -> [resume_phase, tx_count]) to see whether TX is actually flowing after a retrain:
// a changing value = packets going out, a frozen value = TX stalled. ERISC0-only; single L1 store.
constexpr uint32_t MEM_AERISC_TX_PKT_COUNT_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 4;
inline void fabric_dbg_inc_tx_pkt_count() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(MEM_AERISC_TX_PKT_COUNT_ADDR);
    *p = *p + 1;
#endif
}

// [RX-COUNT] Free-running 32-bit count of packets received off the eth link and delivered to the local
// chip, stored in word[2] of the debug slot (MEM_AERISC_RESUME_PHASE_BASE + 8). Compare against a peer
// core's TX count (MEM_AERISC_TX_PKT_COUNT_ADDR) to detect drops: sender_tx > receiver_rx => packets
// lost on that link. Pinned to ERISC1: the receiver channel is serviced by ERISC1 only (the sender/TX
// counter runs on ERISC0), so this is a single writer on the receiver's own RISC.
constexpr uint32_t MEM_AERISC_RX_PKT_COUNT_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 8;
inline void fabric_dbg_inc_rx_pkt_count() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 1)
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RX_PKT_COUNT_ADDR);
    *p = *p + 1;
#endif
}
// Push the current TX packet count into the watcher ring buffer. Called on every context switch so the
// per-core ring buffer becomes a time series of the counter -- if the values keep changing across
// dumps, TX is advancing; if they flatline, TX has stalled. Replaces the old recovery/link-status
// marker pushes. No-op unless the watcher is enabled.
// Push BOTH the TX and RX packet counts into the watcher ring buffer on every context switch, so the
// per-core ring buffer becomes an interleaved time series of both -- letting us watch TX and RX advance
// (or freeze) live in the watcher log, without a post-mortem L1 read (which would require bringing the
// links up and can itself heal the stall we want to observe).
//
// Both are pushed by ERISC0: it owns the TX counter and can read the RX counter that ERISC1 writes
// (shared core L1), so there is a SINGLE writer to the ring buffer -> no cross-ERISC race on the ring
// pointer. Each entry self-identifies by its top nibble so TX/RX can be told apart regardless of where
// the ring wrapped: 0xA = TX, 0xB = RX; the low 28 bits hold the count. Valid while counts < 2^28
// (268M); the 100M-packet test is well within range (a 4G-packet budget would overflow the tag).
constexpr uint32_t FABRIC_DBG_RINGBUF_TX_TAG = 0xA0000000;
constexpr uint32_t FABRIC_DBG_RINGBUF_RX_TAG = 0xB0000000;
constexpr uint32_t FABRIC_DBG_RINGBUF_VALUE_MASK = 0x0FFFFFFF;
inline void fabric_dbg_ringbuf_push_txrx_counts() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    const uint32_t tx = *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_TX_PKT_COUNT_ADDR);
    const uint32_t rx = *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RX_PKT_COUNT_ADDR);
    WATCHER_RING_BUFFER_PUSH(FABRIC_DBG_RINGBUF_TX_TAG | (tx & FABRIC_DBG_RINGBUF_VALUE_MASK));
    WATCHER_RING_BUFFER_PUSH(FABRIC_DBG_RINGBUF_RX_TAG | (rx & FABRIC_DBG_RINGBUF_VALUE_MASK));
#endif
}

// Tiny local reader (direct volatile load, same technique as the PCS_STATUS read above) so we don't
// pull in the eth_txq API headers here.
inline uint32_t fabric_dbg_rd_reg(uint32_t addr) { return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(addr); }

// [PKTMODE-PROBE] FULL eth-queue config snapshot, pushed to the watcher ring buffer once per successful
// retrain (link down->up edge, from recover_eth_link_if_down) -- edge-triggered, so exactly one 9-word
// entry per retrain. Captures every config register eth_enable_packet_mode() sets at init
// (fabric_txq_setup.h) -- which the router sets ONCE and never re-arms in recovery -- so we can see
// which (if any) a retrain clears. Register addresses per fabric_txq_setup.h / tt_eth_ss_regs.h.
//
// The watcher displays the ring buffer NEWEST-FIRST, and we push codeword-first, so in the log a
// snapshot reads as [ ...word8, word7, ..., word1, CODEWORD ] (codeword last). Push order after codeword:
//   [1] TXQ_CTRL           low16 = TXQ0 (0xFFB90000), high16 = TXQ1 (0xFFB91000)  packet_resend = bit0
//   [2] RXQ_CTRL           low16 = RXQ0 (0xFFB94000), high16 = RXQ1 (0xFFB95000)  packet_mode   = bit1
//   [3] MAC_RX_ADDR_ROUTING (0xFFB98154) raw   -- type->RXQ steering (prime suspect)
//   [4] MAC_RX_ROUTING      (0xFFB98150) raw
//   [5] TXPKT_CFG_SEL_SW    low16 = TXQ0 (0xFFB90080), high16 = TXQ1 (0xFFB91080)
//   [6] TXPKT_CFG_SEL_HW    low16 = TXQ0 (0xFFB90084), high16 = TXQ1 (0xFFB91084)
//   [7] TXQ1 dest MAC HI    (0xFFB9829C) raw
//   [8] TXQ1 dest MAC LO    (0xFFB98298) raw
//   [9] ACCEPT_AHEAD        low16 = TXQ0 (0xFFB9006C), high16 = TXQ1 (0xFFB9106C)  -- set at init by
//       eth_txq_reg_write(ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD), NOT by eth_enable_packet_mode(), so the
//       recovery-path config restore does NOT re-apply it. Prime suspect: does a retrain reset it?
// Expected post-init values: [1] bit0 set; [2] bit1 set; [3] routing bits (bcast->RXQ0, mcast->RXQ1);
// [4] 0; [5] TXQ0=0, TXQ1=0x111; [6] TXQ0=0, TXQ1=1; [7]/[8] TXQ1 mcast MAC (0x0100/0x00000001);
// [9] TXQ0/TXQ1 = DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD.
// NOTE: guarded on COMPILE_FOR_AERISC only (not PHYSICAL_AERISC_ID==0), so it fires on whichever ERISC
// calls it -- the retrain-edge call is in ERISC0-only code, but the init call (txq1-active-mode setup)
// runs on the receiver ERISC (ERISC1). The eth queue CTRL/routing regs are shared per-core, so either
// ERISC reads the same values; call sites ensure a single writer at a time.
// Distinct codewords for each lifecycle point at which we snapshot the eth-queue config, so they can be
// told apart in the same log. 7 measurement points across the config's life:
//   [1] PREINIT    - before the init eth_enable_packet_mode() runs (registers at power-on/pre-config)
//   [2] INIT       - right after the init eth_enable_packet_mode() (the golden baseline)
//   [3] RUNTIME    - steady state during a test, traffic flowing, before any link drop (one-shot)
//   [4] DROP       - the context switch a link-down edge is first detected
//   [5] RETRAIN    - the link came back up (retrain succeeded), before the config-restore sequence
//   [6] STATUSCHK  - after the (2nd) update_boot_results_eth_link_status_check() in the restore block
//   [7] PKTMODE    - after eth_enable_packet_mode() re-applied the config in the restore block
// NOTE: ring buffer is 32 entries and a snapshot is now 10 words (codeword + 9 regs), so exactly 3
// snapshots fit (30 entries). Run in phases of <=3 active probes so they coexist in one dump without
// eviction (phase 1: [1]/[2]/[3]; phase 2: [4]/[5]/[6]; phase 3: [5]/[6]/[7]).
constexpr uint32_t FABRIC_DBG_PKTMODE_CODEWORD_PREINIT = 0x5E5EDA01;    // [1] before init config
constexpr uint32_t FABRIC_DBG_PKTMODE_CODEWORD_INIT = 0x5E5EDA02;       // [2] after init config (baseline)
constexpr uint32_t FABRIC_DBG_PKTMODE_CODEWORD_RETRAIN = 0x5E5EDA03;    // [5] retrain-complete edge
constexpr uint32_t FABRIC_DBG_PKTMODE_CODEWORD_RUNTIME = 0x5E5EDA04;    // [3] steady-state during test
constexpr uint32_t FABRIC_DBG_PKTMODE_CODEWORD_DROP = 0x5E5EDA05;       // [4] link-down edge detected
constexpr uint32_t FABRIC_DBG_PKTMODE_CODEWORD_STATUSCHK = 0x5E5EDA06;  // [6] after 2nd link-status check
constexpr uint32_t FABRIC_DBG_PKTMODE_CODEWORD_PKTMODE = 0x5E5EDA07;    // [7] after eth_enable_packet_mode
inline void fabric_dbg_ringbuf_push_pktmode_snapshot([[maybe_unused]] uint32_t codeword) {
#if defined(COMPILE_FOR_AERISC)
    WATCHER_RING_BUFFER_PUSH(codeword);
    // [TWO-REG MODE] snapshot reduced to TWO registers (codeword + 2 words = 3 ring entries) so all 7
    // stage probes can be active at once and coexist in the 32-entry ring (7 x 3 = 21). Keeping
    // DEST_MAC_LO [8] and ACCEPT_AHEAD [9]; re-enable the others to go back to the full 9-register snapshot.
    // WATCHER_RING_BUFFER_PUSH(
    //     (fabric_dbg_rd_reg(0xFFB90000) & 0xFFFF) | ((fabric_dbg_rd_reg(0xFFB91000) & 0xFFFF) << 16));  // [1] TXQ
    //     CTRL
    // WATCHER_RING_BUFFER_PUSH(
    //     (fabric_dbg_rd_reg(0xFFB94000) & 0xFFFF) | ((fabric_dbg_rd_reg(0xFFB95000) & 0xFFFF) << 16));  // [2] RXQ
    //     CTRL
    // WATCHER_RING_BUFFER_PUSH(fabric_dbg_rd_reg(0xFFB98154));  // [3] MAC_RX_ADDR_ROUTING
    // WATCHER_RING_BUFFER_PUSH(fabric_dbg_rd_reg(0xFFB98150));  // [4] MAC_RX_ROUTING
    // WATCHER_RING_BUFFER_PUSH(
    //     (fabric_dbg_rd_reg(0xFFB90080) & 0xFFFF) |
    //     ((fabric_dbg_rd_reg(0xFFB91080) & 0xFFFF) << 16));  // [5] TXPKT_CFG_SEL_SW
    // WATCHER_RING_BUFFER_PUSH(
    //     (fabric_dbg_rd_reg(0xFFB90084) & 0xFFFF) |
    //     ((fabric_dbg_rd_reg(0xFFB91084) & 0xFFFF) << 16));    // [6] TXPKT_CFG_SEL_HW
    // WATCHER_RING_BUFFER_PUSH(fabric_dbg_rd_reg(0xFFB9829C));  // [7] TXQ1 dest MAC HI
    WATCHER_RING_BUFFER_PUSH(fabric_dbg_rd_reg(0xFFB98298));  // [8] TXQ1 dest MAC LO
    WATCHER_RING_BUFFER_PUSH(
        (fabric_dbg_rd_reg(0xFFB9006C) & 0xFFFF) |
        ((fabric_dbg_rd_reg(0xFFB9106C) & 0xFFFF) << 16));  // [9] ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD (off 0x6C)
#endif
}

// Unconditionally stamp the resume-phase word. ERISC0-only; no-op elsewhere and on the host, so call
// sites need no guard beyond ARCH_BLACKHOLE for the macro to exist. A single direct L1 store (not a
// NOC transaction), so it does not perturb the router's dedicated-NOC state.
inline void fabric_dbg_set_resume_phase(uint32_t code) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RESUME_PHASE_BASE) = code;
#endif
}
// Advance the phase word from `from` to `to` only if it currently holds `from`. Lets the hot TX/RX
// paths mark the FIRST post-retrain send/receive exactly once (subsequent packets are a read+compare
// no-op) without a separate flag.
inline void fabric_dbg_advance_resume_phase(uint32_t from, uint32_t to) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RESUME_PHASE_BASE);
    if (*p == from) {
        *p = to;
    }
#endif
}

// This should only be run on ERISC0, and ERISC1 should not be sending/receiving traffic while this is called.
static void recover_eth_link_if_down() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    // Rising-edge state for a *completed* retrain: the link was seen DOWN, then UP again. Persists
    // across context switches (single firmware TU -> one instance per eth core) so a recovery that
    // spans several calls still fires the config restore exactly once, on the call the link comes back.
    static bool eth_link_was_down = false;

    // Read PCS link status RAW (no is_link_up() debounce). Testing showed the debounce -- which on a
    // down-read busy-waits ERISC0 ~5.4B cycles -- correlates with ~4-8 links freezing after retrain (the
    // peer times out while ERISC0 is stalled and the link never re-establishes), whereas the raw-read
    // baseline had 0 frozen. So we read PCS_STATUS directly here: value ==1 means link up.
    const auto pcs_link_up = []() {
        return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(ETH_CORE_A_ETH_CTRL_A_PCS_STATUS_REG_ADDR) == 1;
    };

    // [PROBE 3 - RUNTIME] One-shot config snapshot during steady-state traffic: once ~100k packets have
    // gone out and the link is up (i.e. a normal running test, before any injected link drop), capture the
    // live config once. Static flag so it fires exactly once per core for the whole run.
    static bool eth_runtime_snap_done = false;
    if (!eth_runtime_snap_done && pcs_link_up() &&
        *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_TX_PKT_COUNT_ADDR) > 100000u) {
        // [TXRX MODE] probe disabled.
        // fabric_dbg_ringbuf_push_pktmode_snapshot(FABRIC_DBG_PKTMODE_CODEWORD_RUNTIME);
        eth_runtime_snap_done = true;
    }

    // [#1] Always run the FW link-recovery entry point every context switch (formerly gated behind the
    // `if (true)` debug hack -- now the permanent behavior). When the link is up this is a quick no-op
    // in base FW; when it is down it drives the retrain. The entry point is optional in the FW API
    // table (base/older FW may leave it null) -- gate on a non-zero pointer, since calling through null
    // would jump to address 0 and hang the core.
    const uint32_t eth_link_recovery_ptr =
        (uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->eth_link_recovery_ptr);

    // Detect the DOWN edge exactly once, BEFORE the recovery call. The `!eth_link_was_down` gate latches
    // the down state on the first context switch that sees the link down, then short-circuits so we stop
    // re-reading while it stays down. eth_link_was_down persists until the post-recovery check below
    // confirms the link is back up. Kept BEFORE the recovery call so a blocking recovery (returns with
    // the link already up in this same call) can't hide the down state and make us miss the edge.
    if (!eth_link_was_down && !pcs_link_up()) {
        eth_link_was_down = true;
        // [PROBE 4 - DROP] Config as it stands the moment the link-down edge is first detected (before the
        // FW recovery/retrain touches anything). Edge-gated by eth_link_was_down, so one snapshot per drop.
        // [TXRX MODE] probe disabled.
        // fabric_dbg_ringbuf_push_pktmode_snapshot(FABRIC_DBG_PKTMODE_CODEWORD_DROP);
    }

    if (eth_link_recovery_ptr != 0) {
        // Resume-phase: about to enter FW recovery. If the word stays here, we wedged in the call.
        fabric_dbg_set_resume_phase(RESUME_PHASE_RETRAIN_ENTER);
        reinterpret_cast<void (*)()>(eth_link_recovery_ptr)();
        // Resume-phase: recovery returned (link retrained). The TX/RX paths advance from here.
        fabric_dbg_set_resume_phase(RESUME_PHASE_RETRAIN_DONE);
    }

    // [#2] Immediately after recovery, if the link is now UP and we had seen it DOWN, restore the
    // eth-queue config the retrain corrupted (MAC_DA_HI/LO, TXQ_CTRL incl. disable_remote_drop_
    // notification, TXPKT_CFG_SEL, RXQ_CTRL, TX->RX queue map). Doing it here -- in the SAME context
    // switch the link came back, before the main loop resumes traffic -- means no packet is sent/received
    // over the freshly-retrained link under the corrupted config (the lost-in-flight window). Edge-
    // triggered via eth_link_was_down (cleared here), so it fires exactly ONCE per retrain.
    //
    // Restore sequence: wait 1s, run the FW link-status check, run the FW link-recovery entry point a
    // SECOND time, then re-apply the eth-queue config (eth_enable_packet_mode + ACCEPT_AHEAD). This is the
    // "replicating context switch" restore, now WITH the ACCEPT_AHEAD write included.
    if (eth_link_was_down && pcs_link_up()) {
        eth_link_was_down = false;
        // [PROBE 5 - RETRAIN] Config the instant the link is back up (retrain succeeded), BEFORE the restore
        // sequence below. Diffing this against PROBE 4 (DROP) shows exactly what the retrain corrupted.
        // [TXRX MODE] probe disabled.
        // fabric_dbg_ringbuf_push_pktmode_snapshot(FABRIC_DBG_PKTMODE_CODEWORD_RETRAIN);
        // 1s settle. Also clears the update_boot_results_eth_link_status_check() 1000ms debounce so the FW
        // link-status check below actually runs. Inside the context switch (before the main loop resumes).
        eth_wait_cycles(1000 * ETH_CLOCK_CYCLE_1MS);  // 1s (ETH_CLOCK_CYCLE_1MS = 1e6 cycles = 1ms)
        // FW link-status check -- now past its 1s debounce (the wait above), so this actually invokes it.
        update_boot_results_eth_link_status_check();
        // [PROBE 6 - STATUSCHK] disabled in TXRX mode.
        // fabric_dbg_ringbuf_push_pktmode_snapshot(FABRIC_DBG_PKTMODE_CODEWORD_STATUSCHK);
        // Second link-recovery pass (same entry point as [#1]). Null-guarded identically.
        if (eth_link_recovery_ptr != 0) {
            fabric_dbg_set_resume_phase(RESUME_PHASE_RETRAIN_ENTER);
            reinterpret_cast<void (*)()>(eth_link_recovery_ptr)();
            fabric_dbg_set_resume_phase(RESUME_PHASE_RETRAIN_DONE);
        }
        // receiver_txq_id == 1 in the fabric router (see static_assert in kernel_main). Hardcoded 1
        // because receiver_txq_id is a kernel-TU constant not visible in this base header.
        eth_enable_packet_mode(1);
        // [ACCEPT_AHEAD RESTORE] eth_enable_packet_mode() does NOT touch ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD.
        // The retrain resets it from the configured depth (32) to the power-on default (8) on the
        // downed/retrained end, and nothing else in the recovery path restores it -- leaving the TXQ with a
        // 4x-shallower accept-ahead depth that throttles TX pipelining for the rest of the run (the residual
        // tail-stall / lost-in-flight source). Re-write both TXQs here, mirroring the init path in
        // initialize_state_for_txq1_active_mode(). Value hardcoded to 32 (= DEFAULT_NUM_ETH_TXQ_DATA_PACKET_
        // ACCEPT_AHEAD, a kernel-TU constant not visible in this base header -- same reason receiver_txq_id is
        // hardcoded to 1 above).
        eth_txq_reg_write(0, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, 32);
        eth_txq_reg_write(1, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, 32);
        // [PROBE 7 - PKTMODE] Config after eth_enable_packet_mode() + ACCEPT_AHEAD restore -- should now match
        // the INIT baseline (PROBE 2) on ALL registers incl. ACCEPT_AHEAD (32/32) if the restore is complete.
        // [TXRX MODE] probe disabled.
        // fabric_dbg_ringbuf_push_pktmode_snapshot(FABRIC_DBG_PKTMODE_CODEWORD_PKTMODE);
        if (was_retrained == 0) {
            was_retrained = 1;  // edge-triggered freeze/debug flag; one-shot, matches WAS_RETRAINED gate
        }
    }
#endif
}

// Essentially a copy of what the base erisc main loop does
FORCE_INLINE void aerisc_context_switch() {
#if defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    volatile boot_results_t* const boot_results = (volatile boot_results_t*)(MEM_SYSENG_BOOT_RESULTS_BASE);

    // Update heartbeat - base fw populates 0xabcdxxxx into heartbeat[0], software
    // fabric will also populate that heartbeat. To further help denote that SW has
    // taken over the core, we will populate heartbeat[1] with 0xdcbaxxxx
    volatile uint32_t heartbeat_val = (boot_results->eth_status.heartbeat[0] & 0xFFFF);
    heartbeat_val++;
    heartbeat_val &= 0xFFFF;
    boot_results->eth_status.heartbeat[1] = 0xdcba0000 | heartbeat_val;

    service_eth_msg();
    update_boot_results_eth_link_status_check();

#endif
}

#endif
