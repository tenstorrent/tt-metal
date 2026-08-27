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

// Aliases for the architecture-specific constants used by the debug probes and the post-retrain config
// restore below. Pure preprocessor, so host/device safe.
#include "dbg_reg_refs.h"

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

static __attribute__((unused)) void service_eth_msg() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    invalidate_l1_cache();
    reinterpret_cast<void (*)()>((uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->service_eth_msg_ptr))();
#endif
}

static __attribute__((unused)) void update_boot_results_eth_link_status_check() {
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

// [POST-RETRAIN HANDSHAKE] Retrain notification via a small L1 counter (production behavior -- NOT gated
// on the watcher/testing path). Stored at MEM_AERISC_RETRAIN_COUNT_BASE (see dev_mem_map.h): one per-core
// word holding the running number of spontaneous eth retrains ERISC0 has recovered from on this core.
//   - ERISC0's recovery calls fabric_inc_retrain_count() on each retrain up-edge, AFTER config restore.
//   - The fabric router (ERISC0), inside its coordinated context switch, reads the counter before and
//     after each recovery pass; if it advanced, a retrain completed, so it runs the post-retrain handshake
//     to reconfirm the link is bidirectionally alive before resuming traffic.
// An L1 word rather than a C++ global: it is real device state -- survives as production behavior, is
// host-inspectable, and the router observes it through a plain L1 read like any other shared slot.
// Single writer / single reader, both ERISC0 on this core, so there is no atomicity concern.
inline uint32_t fabric_get_retrain_count() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    return *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RETRAIN_COUNT_BASE);
#else
    return 0;
#endif
}
inline void fabric_inc_retrain_count() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RETRAIN_COUNT_BASE);
    *p = *p + 1;
#endif
}
inline void fabric_reset_retrain_count() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_RETRAIN_COUNT_BASE) = 0;
#endif
}

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
// Declared here (not with the other word[23] helpers further down) because
// fabric_dbg_inc_tx_pkt_count below resets this latch and must see the constant.
constexpr uint32_t MEM_AERISC_SYNC_MIN_FREE_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 92;  // word[23]

inline void fabric_dbg_inc_tx_pkt_count() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(MEM_AERISC_TX_PKT_COUNT_ADDR);
    *p = *p + 1;
    // A packet just went out -> open a fresh min-free window. After the last data packet this latch
    // therefore accumulates over the barrier only, where the sync packet is the sole possible doorbell.
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_SYNC_MIN_FREE_ADDR) = 0xFFFFFFFFu;
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

// [CRED-PROBE] Sender's absolute count of COMPLETION credits it has RECEIVED from the peer receiver
// (i.e. *completions_received_counter_ptr for a stalled sender channel), stored in the previously-unused
// word[3] of the debug slot (MEM_AERISC_RESUME_PHASE_BASE + 12). ERISC0 (sender) writes it; ERISC0's
// context-switch push emits it. Compare against the PEER core's RX count (MEM_AERISC_RX_PKT_COUNT_ADDR,
// which ~= completions the peer SENT, since it emits one completion per received+flushed packet, and each
// loudbox core is exactly one link): peer_RX - this_recvd_completions == completion credits lost over the
// link. A nonzero gap on a tail-stalled sender is the smoking gun for signature (a) (lost final completion).
constexpr uint32_t MEM_AERISC_CRED_RECVD_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 12;
// [#45872 QUIESCE HANDSHAKE] word[3] repurposed as the sender->router ACK. Router raises STOP (word[10]) at the
// link-down edge and spins on this until the connected payload sender sets it to 1 (after it stops sending +
// flushes its outstanding NoC writes), so all register/occupancy measurement happens on a quiescent channel.
constexpr uint32_t MEM_AERISC_HANDSHAKE_ACK_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 12;  // word[3]
// [#45872 STREAM-REG TRAJECTORY] Router-local get_ptr_val(stream 22) captured at the four regions of interest,
// all on the quiescent channel (measurement begins only after the STOP/ACK handshake):
//   R0 = raw register at the instant of down (pre-handshake; may include in-flight)  -> word[4]
//   R1 = settled register at retrain-begin (post-handshake)                          -> word[11] FS22_AT_DOWN
//   min= lowest register seen while down                                             -> word[12] FS22_MIN
//   R2 = register at retrain-finish (up-edge)                                        -> word[5]
//   R3 = how far it climbs back after retrain (max free_slots, retrain_count>0)      -> word[6]
constexpr uint32_t MEM_AERISC_REG_AT_DOWN_RAW_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 16;     // word[4]  R0
constexpr uint32_t MEM_AERISC_REG_AT_RETRAIN_END_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 20;  // word[5]  R2
constexpr uint32_t MEM_AERISC_REG_CLIMB_MAX_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 24;       // word[6]  R3
// [#45872 OCCUPANCY COMPARE] Router-side, both read locally in the SAME speedy iteration (true simultaneity, no
// NoC, no shadow staleness). After the STOP/ACK handshake the sender is frozen, so:
//   occ_true = occ_at_STOP - (local_read_counter - F0)   [counter-based: sender writes fixed, router forwards]
//   occ_reg  = num_buffers - get_ptr_val()               [what the router reads in the stream register]
// Amortized credits mean occ_true steps in batches -> trust the SETTLED end-of-test values, not mid-drain.
constexpr uint32_t MEM_AERISC_OCC_TRUE_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 28;  // word[7]   counter-based occupancy
constexpr uint32_t MEM_AERISC_OCC_REG_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 60;   // word[15]  register-based occupancy
// [#45872 READ-POINTER] Router's sender-channel-0 read pointer (local_read_counter = packets it has forwarded)
// latched at three points, to see how much it actually drains AFTER retrain -- counter-based, immune to the
// register. rp_finish - rp_begin = forwards during the retrain window; rp_settled - rp_finish = post-retrain drain.
constexpr uint32_t MEM_AERISC_RP_BEGIN_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 52;   // word[13] read ptr @ retrain-begin
constexpr uint32_t MEM_AERISC_RP_FINISH_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 56;  // word[14] read ptr @ retrain-finish
constexpr uint32_t MEM_AERISC_RP_SETTLED_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 24;  // word[6]  read ptr live (settled)

// [RECEIVER-SIDE PROBES] Full receiver-side flow-control state, so we can see exactly where completions
// live (which channel, and the local completion_counter). All written by ERISC1 (receiver) from the
// receiver step, read+pushed by ERISC0.
//   word[4] (+16) = local_receiver_completion_counters[0]  (completions SENT for sender channel 0)
//   word[5] (+20) = local_receiver_completion_counters[1]  (completions SENT for sender channel 1)
//   word[6] (+24) = receiver_channel_pointers.completion_counter.counter (local completions PROCESSED)
// (RX packets received is already tracked at MEM_AERISC_RX_PKT_COUNT_ADDR, word[2].)
constexpr uint32_t MEM_AERISC_COMPLETION_SENT_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 16;     // LRC ch0
constexpr uint32_t MEM_AERISC_COMPLETION_SENT1_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 20;    // LRC ch1
constexpr uint32_t MEM_AERISC_RECV_COMPL_COUNTER_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 24;  // local completion_counter
// [RX PIPELINE PROBE] words 8..14 of the debug slot. ALL written by ERISC1 (the receiver), so they stay
// live even when ERISC0 has wedged -- unlike the watcher ring buffer, which ERISC0 owns and stops
// pushing the moment it stops executing. Read these out of L1 with exalens while the core is hung.
//
// Purpose: test the "receiver buffer is full" hypothesis DIRECTLY rather than by elimination.
// occupancy == to_receiver_pkts_sent_id, i.e. packets the peer has announced minus packets this
// receiver has dequeued. Buffer full => occupancy == RECEIVER_NUM_BUFFERS; buffer empty => 0.
constexpr uint32_t MEM_AERISC_RX_OCCUPANCY_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 32;
constexpr uint32_t MEM_AERISC_RX_OCC_HIWATER_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 36;
constexpr uint32_t MEM_AERISC_RX_WR_SENT_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 40;
// [#45872 DOORBELL SENT-vs-RECEIVED] Reclaimed words 8-10 (RX-pipeline probe disabled above). The stream
// HW applies a space-available doorbell (reg 270) silently -- no native received-count -- so the sender
// worker bumps a SHADOW counter here (noc_semaphore_inc) alongside each real doorbell. That shadow is the
// ground-truth "received/arrived" count, immune to the stale local read and the retrain reset.
//   word[8] RECV_CUM       = cumulative doorbells that ARRIVED at this router (shadow, worker-incremented)
//   word[9] RECV_WHILE_DOWN = doorbells that arrived while the link was DOWN/retraining (ERISC-gated delta)
//   word[10] RECV_AT_DOWN  = shadow snapshot latched at the PCS down-edge (ERISC scratch for the delta)
// Compare against the sender's own cumulative SENT (worker-side, pushed to the watcher ring, tag 0x99):
//   SENT == RECV_CUM        -> every doorbell arrived (no in-transit loss)
//   SENT >  RECV_CUM        -> doorbells sent but NOT received (dropped before the router)
//   RECV_WHILE_DOWN > 0     -> doorbells DO arrive during the retrain (rules out "workers paused")
constexpr uint32_t MEM_AERISC_DBELL_RECV_CUM_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 32;         // word[8]
constexpr uint32_t MEM_AERISC_DBELL_RECV_WHILE_DOWN_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 36;  // word[9]
constexpr uint32_t MEM_AERISC_DBELL_RECV_AT_DOWN_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 40;     // word[10]
// [#45872] Does the STREAM REGISTER reflect the arriving doorbells during the down? The ERISC reads its own
// channel-0 free-slots (stream 22, get_ptr_val) and records the value at the down-edge + the MIN while down.
// Compare against RECV_WHILE_DOWN (word[9], ~30 arrivals): if FS22_AT_DOWN - FS22_MIN ~= 30, the register
// reflected the decrements; if FS22_MIN stays == FS22_AT_DOWN (no drop), the arrivals are NOT reflected in
// the register the router polls -> it can't see them -> the hang.
constexpr uint32_t MEM_AERISC_DBELL_FS22_AT_DOWN_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 44;  // word[11]
constexpr uint32_t MEM_AERISC_DBELL_FS22_MIN_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 48;      // word[12]
// [#45872 NEUTRAL READ] word[13] LINK_DOWN_FLAG: ERISC sets =1 at the PCS down-edge, =0 at the up-edge, so a
// bystander (neutral) core can gate its own measurement on the down window without reading PCS itself.
// word[14] NEUTRAL_MIN: the min free-slots a NEUTRAL third core (sync worker) observes by reading this eth
// core's stream 22 over the NoC WHILE the flag is set. Compare vs FS22_MIN (word[12], the router's LOCAL
// read): both stay 32 -> decrements not applied; NEUTRAL_MIN drops but FS22_MIN stays 32 -> local read stale.
constexpr uint32_t MEM_AERISC_DBELL_LINK_DOWN_FLAG_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 52;  // word[13]
constexpr uint32_t MEM_AERISC_DBELL_NEUTRAL_MIN_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 56;     // word[14]
constexpr uint32_t MEM_AERISC_RX_WR_FLUSH_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 44;
constexpr uint32_t MEM_AERISC_RX_ACK_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 48;
constexpr uint32_t MEM_AERISC_RX_FLUSHSTATE_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 52;
constexpr uint32_t MEM_AERISC_RX_HEARTBEAT_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 56;

// [RESTORE-WINDOW PROBE] Did the hardware transmit anything between the link coming back up and the
// eth-queue config being restored?
//
// Theory under test: after a retrain the link is live but the config is still the post-retrain (wrong)
// state -- notably MAC_DA_LO == 0, which addresses TXQ1 traffic to RXQ0 instead of RXQ1. Anything the
// hardware drains from an already-queued TXQ in that window is therefore misrouted and lost. TXQ1 is
// the credit path, TXQ0 is data and its MAC is *supposed* to be 0, which would explain why credits go
// missing across a retrain while bulk data does not.
//
// The SOFTWARE TX counter cannot see this: fabric_dbg_inc_tx_pkt_count() fires when ERISC0 issues a
// send, and ERISC0 is parked inside the coordinated context switch for the whole window. We therefore
// sample the HARDWARE frame counter, boot_results.eth_live_status.frames_txd.
//
// Address: boot_results base 0x7CC00 + eth_status(128) + serdes_results(256) + macpcs_results(128)
// = 0x7CE00, which is exactly MEM_RETRAIN_COUNT_ADDR (eth_live_status_t.retrain_count, the first
// field) -- that cross-check pins the base. frames_txd is then +32 bytes past retrain_count,
// rx_link_up, link_flap_count, link_poll_alive_count and spare[4]. Low 32 bits are enough: we only
// ever take a difference across a sub-second window.
constexpr uint32_t MEM_SYSENG_ETH_FRAMES_TXD_LO = 0x7CE20;
constexpr uint32_t MEM_AERISC_HWTX_AT_LINKUP_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 28;      // word[7]
constexpr uint32_t MEM_AERISC_HWTX_AFTER_RESTORE_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 60;  // word[15]

// [SEND-GATE PROBE] Why did / didn't the router transmit, per sender channel.
//
// The router only puts a packet on the wire when BOTH inputs to its send gate are true:
//     receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet()
//     has_unsent_packet             = get_ptr_val(free_slots_stream) != num_buffers
//     can_send                      = receiver_has_space && has_unsent_packet
// Knowing WHICH input is false separates the two candidate explanations for the wedged barrier:
//   has_unsent=0                -> nothing was handed to this channel; the packet isn't here
//   has_unsent=1, recv_space=0  -> the packet IS queued but the remote receiver shows zero free
//                                  slots, i.e. the sender is credit-starved and can never transmit
//                                  it. This is the case that ties the barrier hang to the credit
//                                  loss we measured, and it means the sync packet never reaches
//                                  the wire at all.
//
// This matters most at END OF RUN: data traffic has stopped by then, so the only thing left to send
// is the sync packet, and whatever the gate says at the hang is the reason the barrier died.
// Overwritten every sender step, so a wedged router leaves the gate state frozen at the stuck value.
//
// Layout: [31:24] 0xC0|channel   [23:16] remote receiver free slots   [15:8] local free-slot stream
//         [7:0] flags: bit0 has_unsent_packet, bit1 receiver_has_space, bit2 can_send
constexpr uint32_t MEM_AERISC_SENDER_GATE_CH0_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 64;  // word[16]
constexpr uint32_t MEM_AERISC_SENDER_GATE_CH1_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 68;  // word[17]

// [SYNC-PACKET COUNTERS] Monotonic counts of atomic-inc packets crossing the eth link.
//
// A sync packet is distinguishable from bulk traffic by its header: the global-sync patterns are built
// with ntype = NOC_UNICAST_ATOMIC_INC and size = 0, whereas this test's data pattern is
// NOC_UNICAST_WRITE with a 4KB payload. So noc_send_type == NOC_UNICAST_ATOMIC_INC (2) uniquely
// identifies a barrier sync signal here, and both the sender and receiver steps already have the
// header field to hand -- the receiver literally already decodes it into packed.noc_send_type.
//
// These are COUNTERS, not state snapshots. That distinction matters: the send-gate words above record
// the CURRENT gate state, which cannot tell "a packet was never queued" from "a packet was queued and
// successfully transmitted" -- both leave the channel empty. A monotonic count settles it.
//
// Compare across a link: sender's TX count vs the peer's RX count.
//   TX advances, peer RX advances -> packet crossed the wire; failure is further downstream
//   TX advances, peer RX flat     -> transmitted and lost in flight
//   TX flat                       -> never transmitted; the failure is upstream of the wire
constexpr uint32_t MEM_AERISC_SYNC_TX_COUNT_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 72;  // word[18]
constexpr uint32_t MEM_AERISC_SYNC_RX_COUNT_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 76;  // word[19]

// [SLOT-CONTENT PROBE] What is actually sitting in the buffer slot the router would read NEXT?
//
// Everything so far says the sync packet reaches the router's L1 -- but that rests on the NoC ack
// (noc_async_writes_flushed returning), which is indirect. This reads the BYTES instead: the router
// dereferences its own next buffer slot every sender step and records the packet header's
// payload_size/noc_send_type word, which lives at offset 40 (sizeof(NocCommandFields) == 40, and
// PackedPayloadAndSendType::load reads exactly this word: size = raw & 0xFFFF, type = (raw>>16)&0xFF).
//
// A sync packet is size=0, noc_send_type=NOC_UNICAST_ATOMIC_INC(2)  ->  word == 0x00020000
// A data packet here is size=4096(0x1000), NOC_UNICAST_WRITE(0)     ->  word == 0x00001000
//
// So at the hang:
//   word[21] == 0x00020000  -> the sync packet IS in the slot the router would read next. Independent
//                              confirmation, no NoC ack involved: delivered, and the router is simply
//                              not picking it up.
//   word[21] == 0x00001000  -> the slot still holds the last DATA packet; the sync packet is not here
//                              (wrong channel, or never landed).
// word[20] records the slot address that was read, so the value can be tied to a concrete location.
constexpr uint32_t MEM_AERISC_SLOT_ADDR_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 80;  // word[20]
constexpr uint32_t MEM_AERISC_SLOT_HDR_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 84;   // word[21]
constexpr uint32_t MEM_AERISC_SYNC_SEEN_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 88;  // word[22]

// Offset of PacketHeaderBase::payload_size_bytes within the packet header (see comment above).
constexpr uint32_t FABRIC_DBG_PKT_HDR_SIZE_TYPE_OFFSET = 40;

// [STREAM-ID PROBE] Which stream register does the ROUTER poll for "a packet is waiting"?
//
// The data path proves this register works after a retrain: 100M packets flow post-retrain, each one
// decrementing it from the worker side and being seen here. So the register is not corrupted. What
// differs for the sync path is that its connection is CLOSED and REOPENED across the retrain, and the
// worker resolves its target register at open time:
//     edm_buffer_remote_free_slots_update_addr = get_stream_reg_write_addr(sender_channel_credits_stream_id)
// If the rebuilt connection resolves a different id than the router polls, the decrement lands on a
// register nobody reads -- which fits every observation (both writes acked, localfree stuck at
// num_buffers, round 0 fine because that connection predates the teardown).
//
// Packed with the slot address since both are ERISC0 and one word is enough for each:
//   [31:16] stream id polled by the router   [15:0] low 16 bits of the next-read slot address
// The full slot address is still recoverable: the region is well under 64KB and the high bits are
// fixed, and word[21] carries the header word read from it.
// [DOORBELL CROSS-CHECK] The router's OWN view of the doorbell, recorded so it can be compared against
// an independent host-side read of the same hardware register.
//
//   word[24] = the stream id the router actually polls (measured, not assumed -- lets us verify it
//              matches the worker's target, stream 22, without relying on recollection)
//   word[25] = the raw free_slots value from get_ptr_val (NOC_STREAM_READ_REG, UNCACHED -- so unlike
//              the L1 header probes this is not subject to data-cache staleness)
//
// Pair with the host's STREAMREG dump of stream 22 AVAILABLE (0xFFB564A4):
//   both 32          -> counter genuinely never moved; worker's write acked but never applied
//   host 31, w25 32  -> router's read is stale (would be surprising: this is an uncached reg read)
//   both 31          -> doorbell DID fire; the bug is downstream of notification
constexpr uint32_t MEM_AERISC_POLLED_STREAM_ID_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 96;    // word[24]
constexpr uint32_t MEM_AERISC_POLLED_FREE_SLOTS_ADDR = MEM_AERISC_RESUME_PHASE_BASE + 100;  // word[25]

inline void fabric_dbg_set_polled_doorbell(
    [[maybe_unused]] uint32_t free_slots_stream_id, [[maybe_unused]] uint32_t free_slots) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_POLLED_STREAM_ID_ADDR) = free_slots_stream_id;
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_POLLED_FREE_SLOTS_ADDR) = free_slots;
#endif
}

inline void fabric_dbg_set_next_slot_content(
    [[maybe_unused]] uint32_t slot_addr, [[maybe_unused]] uint32_t free_slots_stream_id) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    // Store the FULL 32-bit router read address (was: polled_id<<16 | low16). We are testing whether the
    // router's read slot differs from the worker's write address (WRADDR=0x16ad0) in the HIGH bits -- the
    // low 16 match (0x6ad0) but SLOT_HDR here reads DATA while the sync is in L1 at 0x16ad0. polled_id is a
    // confirmed constant (22), so dropping it costs nothing.
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_SLOT_ADDR_ADDR) = slot_addr;
    invalidate_l1_cache();  // see note above: header is NoC-written, must fence before reading
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_SLOT_HDR_ADDR) =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr + FABRIC_DBG_PKT_HDR_SIZE_TYPE_OFFSET);
#endif
}

// [DECREMENT-LOST vs DECREMENT-RESET PROBE] The round-1 sync PAYLOAD demonstrably lands in the ERISC slot
// (L1 scan), but free_slots (stream 22) reads 32 so the router thinks the slot is empty. Two mechanisms:
//   (1) the worker's stream-22 decrement never landed, or
//   (2) it landed (32->31) then connection reinit `init_ptr_val(stream22, num_buffers)` reset it to 32.
// This tracks the MINIMUM free_slots seen while a SYNC packet occupies the slot, resetting to a sentinel
// whenever the slot is NOT a sync packet -- so between sync windows it clears, and the final value
// reflects only the current (round-1, stuck) sync window, not round 0's brief forwarded window.
//   final == 32  -> free_slots never dipped while round-1's sync sat in the slot -> decrement NEVER landed.
//   final <  32  -> free_slots was seen below num_buffers -> decrement DID land, then got reset.
// REPURPOSED (same word[23], no base move): minimum free_slots seen SINCE THE LAST TRANSMITTED PACKET.
//
// Why the previous version was useless: it gated on reading the slot header from L1 WITHOUT
// invalidate_l1_cache(), so it saw stale cache contents -- the host scan proved the sync header sat at
// slot+40 while this same read returned 0x1000. Its sentinel-reset branch then fired constantly and the
// probe was effectively blind. It also snapshotted a value that is 32 almost all the time.
//
// Why a bare min() is also useless: the doorbell is SHARED with the 100M-packet data path, so a latched
// minimum hits 31 in the first millisecond and stays there forever, on every core, regardless of sync.
//
// The fix uses ordering rather than packet type: each sender completes ALL its data packets, THEN closes
// its connection, THEN the barrier runs. So the latch is reset on every TX increment (see
// fabric_dbg_inc_tx_pkt_count) and only accumulates afterwards -- meaning it holds "min free_slots since
// the last packet actually went out". Once traffic stops, the ONLY thing that can ring the doorbell is
// the sync packet. Config-independent: no hardcoded 100M threshold, and no dependence on num_packets or
// on round-0's sync inflating TX to 100,000,001.
//
// Reads no L1, so it is immune to the data-cache staleness that broke the header-based probes.
//
// Interpretation at the hang, paired with syncTX (word[18]):
//   syncTX==1, min==32 -> doorbell NEVER rang after the last packet -> the write never took effect
//   syncTX==1, min< 32 -> doorbell DID ring and the router saw it, yet never sent -> fault is downstream
//   syncTX==2, min==32 -> core sent its round-1 sync fine (healthy control, e.g. 4/8) -- the latch was
//                         reset by that very transmission, so 32 here means success, not failure.
inline void fabric_dbg_track_min_free_since_tx([[maybe_unused]] uint32_t free_slots) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(MEM_AERISC_SYNC_MIN_FREE_ADDR);
    if (free_slots < *p) {
        *p = free_slots;
    }
#endif
}

// [SELF-LOOPBACK PROBE -- issue #45872] word[23], repurposed from the min-free latch above (whose
// conclusion has since been superseded).
//
// Established so far: the worker's REMOTE NOC read of stream 22 index 297 returns 31 while the router's
// LOCAL read of that same address returns 32, at the same instant, on the same live core, 19/20
// connections, 3/3 runs. One indirection remains in that comparison: the "router's value" is read out of
// this debug slot rather than executed on the ERISC at an instant of our choosing.
//
// This probe removes the second endpoint entirely. The ERISC reads its OWN register back through the
// NOC and latches it here, in the same loop iteration in which it did the local read that lands in
// word[25]. Two access paths, one core, one iteration, no worker involved:
//
//   word[23] loopback != word[25] local  -> the two paths genuinely disagree on one register
//   word[23] loopback == word[25] local  -> both local paths agree; the divergence is specific to
//                                           access arriving from another core, which is a different bug
//
// Layout: [31:20] sample counter (proves the probe fired and kept firing), [16:0] the value read back.
inline void fabric_dbg_latch_loopback([[maybe_unused]] uint32_t loopback_val, [[maybe_unused]] uint32_t sample_idx) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_SYNC_MIN_FREE_ADDR) =
        ((sample_idx & 0xFFF) << 20) | (loopback_val & 0x1FFFF);
#endif
}

// Raw PCS link-up read (value == 1 means up), for gating the self-loopback probe. The loopback's
// noc_async_read_barrier hangs the sender step while a link is DOWN (recovering NOC), which prevents the
// router from ever reaching the context switch that runs eth recovery -- so only fire the probe when the
// link is up (the end-of-run barrier hang is post-retrain, link up).
inline bool fabric_dbg_link_is_up() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    return *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(ETH_CORE_A_ETH_CTRL_A_PCS_STATUS_REG_ADDR) == 1;
#else
    return true;
#endif
}

// Reset the min-free latch: a packet just went out, so start a fresh window.
inline void fabric_dbg_reset_min_free_latch() {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    *reinterpret_cast<volatile uint32_t*>(MEM_AERISC_SYNC_MIN_FREE_ADDR) = 0xFFFFFFFFu;
#endif
}

inline void fabric_dbg_set_sender_gate(
    [[maybe_unused]] uint32_t channel_index,
    [[maybe_unused]] uint32_t recv_free_slots,
    [[maybe_unused]] uint32_t local_free_slots,
    [[maybe_unused]] bool has_unsent,
    [[maybe_unused]] bool recv_has_space,
    [[maybe_unused]] bool can_send) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    // Only channels 0/1 have a slot; anything else is dropped rather than scribbling past the region.
    if (channel_index > 1) {
        return;
    }
    const uint32_t addr = (channel_index == 0) ? MEM_AERISC_SENDER_GATE_CH0_ADDR : MEM_AERISC_SENDER_GATE_CH1_ADDR;
    const uint32_t packed = ((0xC0u | (channel_index & 0xF)) << 24) | ((recv_free_slots & 0xFF) << 16) |
                            ((local_free_slots & 0xFF) << 8) | (has_unsent ? 0x1u : 0u) | (recv_has_space ? 0x2u : 0u) |
                            (can_send ? 0x4u : 0u);
    *reinterpret_cast<volatile uint32_t*>(addr) = packed;
#endif
}

// [CONN-LIFECYCLE PROBE #45872] Repurpose word[17] (SENDER_GATE_CH1, always 0 in single-link runs and
// ERISC0-owned) to pack channel-0 (stream 22) connect/disconnect edge counts: high 16 = connects, low 16 =
// disconnects. Stamped from check_worker_connections on the connect/teardown edges. Verifies clean
// disconnect-before-reconnect: connects - disconnects should be 0 (idle) or 1 (a worker currently
// connected). If it ever exceeds 1, a fresh connect happened without the prior worker's teardown = dirty
// handoff. If at the barrier hang connects == disconnects (delta 0), the sync's connect never registered.
constexpr uint32_t MEM_AERISC_CH0_CONNLIFE_ADDR = MEM_AERISC_SENDER_GATE_CH1_ADDR;  // word[17], repurposed

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
constexpr uint32_t FABRIC_DBG_RINGBUF_CRED_TAG = 0xC0000000;    // sender's received-completion count
constexpr uint32_t FABRIC_DBG_RINGBUF_CSENT_TAG = 0xD0000000;   // receiver's sent-completion count, chan 0 (LRC0)
constexpr uint32_t FABRIC_DBG_RINGBUF_CSENT1_TAG = 0xE0000000;  // receiver's sent-completion count, chan 1 (LRC1)
constexpr uint32_t FABRIC_DBG_RINGBUF_RXCC_TAG = 0xF0000000;    // receiver's local completion_counter
constexpr uint32_t FABRIC_DBG_RINGBUF_VALUE_MASK = 0x0FFFFFFF;

// [#45872 DRAIN TIME-SERIES] Time-resolved trace of the post-retrain drain, pushed from the speedy sender
// step (channel 0) once the sender has quiesced. The existing 0xE4 probe only fires inside if(can_send), so
// it stops the instant forwarding stops and cannot show the settle; this one samples every loop iteration
// and pushes only when something MOVES, which is what keeps a 32-entry ring viable across a whole drain.
//
// One entry per change, packed into a single word:
//   [31:28] tag   0x9 = drain sample, 0x8 = retrain_count-increment marker
//   [27:22] seq   free-running 0..63, so a wrap past the 32-entry ring is visible instead of silent
//   [21:16] reg   stream-22 free slots as the router reads them (0..32, saturating at 63)
//   [15:10] occ   counter-derived occupancy, occ_at_stop - forwards_since (0..32, saturating at 63)
//   [ 9: 0] dt    wall-clock delta since the previous entry, in 64-cycle units (64ns), saturating at 1023
// The eth wall clock is 1 GHz (ETH_CLOCK_CYCLE_1MS = 1e6 cycles/ms), so dt spans ~65us before saturating.
constexpr uint32_t FABRIC_DBG_DRAIN_TS_TAG = 0x9u;   // drain sample
constexpr uint32_t FABRIC_DBG_DRAIN_RC_TAG = 0x8u;   // retrain_count incremented here
constexpr uint32_t FABRIC_DBG_DRAIN_DT_SHIFT = 6u;   // wall-clock cycles per dt unit (2^6 = 64)
constexpr uint32_t FABRIC_DBG_DRAIN_FIELD_MAX = 63u;  // 6-bit saturation for reg/occ/seq
constexpr uint32_t FABRIC_DBG_DRAIN_DT_MAX = 1023u;   // 10-bit saturation for dt
// Number of periodic samples emitted after the values stop changing, so a settled trace is distinguishable
// from one that merely stopped recording. Spaced by FABRIC_DBG_DRAIN_SETTLE_CYCLES.
constexpr uint32_t FABRIC_DBG_DRAIN_SETTLE_SAMPLES = 4u;
constexpr uint32_t FABRIC_DBG_DRAIN_SETTLE_CYCLES = ETH_CLOCK_CYCLE_1MS;  // 1ms between settle samples

inline uint32_t fabric_dbg_drain_pack(uint32_t tag, uint32_t seq, uint32_t reg, uint32_t occ, uint32_t dt_units) {
    const uint32_t s = (seq & FABRIC_DBG_DRAIN_FIELD_MAX);
    const uint32_t r = (reg > FABRIC_DBG_DRAIN_FIELD_MAX) ? FABRIC_DBG_DRAIN_FIELD_MAX : reg;
    const uint32_t o = (occ > FABRIC_DBG_DRAIN_FIELD_MAX) ? FABRIC_DBG_DRAIN_FIELD_MAX : occ;
    const uint32_t d = (dt_units > FABRIC_DBG_DRAIN_DT_MAX) ? FABRIC_DBG_DRAIN_DT_MAX : dt_units;
    return (tag << 28) | (s << 22) | (r << 16) | (o << 10) | d;
}

// [#45872 DRAIN SELF-CHECK] No dedicated counter word is needed, and there isn't a free one: word[16] looks
// spare but fabric_dbg_set_sender_gate() rewrites it every iteration, and growing the slot would move the
// base out from under a dozen hardcoded 0x6F1F8 / 0x6F220 sites in the test kernel.
//
// Instead the trace is self-checking by construction. The enclosing ACK-gated block writes word[6] (read
// pointer), word[7] and word[15] on EVERY speedy iteration, so a populated word[6] proves the block ran; and
// the first drain sample always pushes, because ts_last_reg starts at a value free_slots cannot hold. So:
//   word[6] live + ring entries present -> trace is working
//   word[6] live + ring EMPTY           -> the push executed and the ring lost it
//   word[6] dead                        -> the block never ran (ACK never set)
// That is the same three-way split the 0xE4 probe currently leaves ambiguous, at no L1 cost.

// MUST stay noinline. The speedy sender step is inlined wholesale into the router main loop, and ERISC0's
// stack is a fixed 2048B set up by base firmware -- bh_hal.cpp caps kernels at -Werror=stack-usage=1912.
// Inlining this sampler's locals and their live ranges into that already-enormous frame took it to 3680B
// and failed the build; given its own frame it costs the caller only the argument marshalling. The clock is
// read as the low 32 bits only (wraps every ~4.3s at 1GHz) so no 64-bit arithmetic reaches the hot loop --
// unsigned subtraction still yields the correct delta across a wrap, and deltas here are sub-millisecond.
__attribute__((noinline)) inline void fabric_dbg_drain_sample(
    [[maybe_unused]] uint32_t free_slots, [[maybe_unused]] uint32_t occ, [[maybe_unused]] uint32_t rc) {
#if defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)
    static uint32_t ts_seq = 0;
    static uint32_t ts_last_reg = 0xFFFFFFFFu;  // impossible free-slot value -> the first sample always pushes
    static uint32_t ts_last_occ = 0xFFFFFFFFu;
    static uint32_t ts_last_t = 0;
    static uint32_t ts_rc_at_arm = 0xFFFFFFFFu;
    static bool ts_rc_marked = false;
    static bool ts_started = false;
    static uint32_t ts_settle_left = FABRIC_DBG_DRAIN_SETTLE_SAMPLES;

    if (ts_rc_at_arm == 0xFFFFFFFFu) {
        ts_rc_at_arm = rc;  // arm point: first call after the sender ACKed the quiesce handshake
    }
    const uint32_t now = eth_risc_reg_read(ETH_RISC_WALL_CLOCK_0);
    const bool moved = (free_slots != ts_last_reg) || (occ != ts_last_occ);
    const bool rc_edge = !ts_rc_marked && (rc != ts_rc_at_arm);
    // Once nothing moves, emit a bounded number of spaced samples so a settled trace is distinguishable
    // from one that merely stopped recording.
    const bool settle_due = !moved && ts_started && (ts_settle_left != 0u) &&
                            ((now - ts_last_t) >= FABRIC_DBG_DRAIN_SETTLE_CYCLES);
    if (!(moved || rc_edge || settle_due)) {
        return;
    }
    const uint32_t dt_units = ts_started ? ((now - ts_last_t) >> FABRIC_DBG_DRAIN_DT_SHIFT) : 0u;
    WATCHER_RING_BUFFER_PUSH(fabric_dbg_drain_pack(
        rc_edge ? FABRIC_DBG_DRAIN_RC_TAG : FABRIC_DBG_DRAIN_TS_TAG, ts_seq, free_slots, occ, dt_units));
    ts_seq++;
    ts_last_t = now;
    ts_started = true;
    if (rc_edge) {
        ts_rc_marked = true;
    }
    if (moved) {
        ts_last_reg = free_slots;
        ts_last_occ = occ;
        ts_settle_left = FABRIC_DBG_DRAIN_SETTLE_SAMPLES;  // re-arm the tail after any motion
    } else if (settle_due) {
        ts_settle_left--;
    }
#endif
}

// [CREDIT-STALL DUMP MODE] Alternative to fabric_dbg_ringbuf_push_txrx_counts: instead of a per-context-
// switch TX/RX/CRED time series (which floods the 32-entry ring), this keeps the ring QUIET and only emits
// a one-shot dump when a core has STOPPED TRANSMITTING (TX frozen) for ~5 minutes. This fires for ANY core
// whose TX freezes that long -- whether it froze because it STALLED (TX < budget) or COMPLETED (TX at
// budget). Done cores dumping is intentional: for a one-sided stall (stalled sender + done peer) we need the
// DONE peer's CSENT (completions it sent to the stalled end) to compute the loss. The dump is the flow-
// control completion credits; word order in the ring (newest-first) reads [CSENT, CRED, TX, CODE]. Pair the
// two endpoints of a link via peers_bh.json and compare: receiver CSENT - peer sender CRED == credits lost.
// A TX=budget dump = a completed core (reference); a TX<budget dump = the stalled core. ERISC0-only (owns TX
// + does the dump); reads the CSENT slot ERISC1 writes (shared core L1).
constexpr uint32_t FABRIC_DBG_CREDIT_STALL_CODE = 0x5E5ECD00;  // "credit dump" marker
// Stall timeout at the 1 GHz eth wall clock (ETH_CLOCK_CYCLE_1MS = 1e6 cycles/ms). Lowered 5min->2min:
// tail-stalls freeze LATE (~99.99M, ~250s into the run), so a 5-min timeout pushed their dump to ~9 min --
// right at the capture-window edge. 2 min reliably catches late-freezing tail-stalls. 64-bit: fits.
constexpr uint64_t FABRIC_DBG_CREDIT_STALL_CYCLES = (uint64_t)2 * 60 * 1000 * ETH_CLOCK_CYCLE_1MS;

// [HANDSHAKE DEBUG] Watcher ring-buffer markers for the eth handshake, split by ROLE so a wedged core's
// ring tells you directly whether it was the sender-side (master) or receiver-side (subordinate) end --
// no dump_peers pairing needed. ENTER pushed before the spin, DONE right after it returns; ENTER with no
// matching DONE == wedged (far end never answered). Codeword layout 0x5E5EAA[role][kind] (AA == the
// handshake MAGIC_HANDSHAKE_VALUE, so they stand out from resume-phase 0x5E5E00xx / pktmode 0x5E5EDAxx):
//   role nibble: 1 = SENDER (master), 2 = RECEIVER (subordinate)   -> grep "5e5eaa1" vs "5e5eaa2"
//   kind nibble: 1 = post-retrain ENTER, 2 = post-retrain DONE, 3 = init ENTER, 4 = init DONE
// Pushed on ERISC0 (the only ERISC that runs either handshake); role is the compile-time is_handshake_sender.
constexpr uint32_t FABRIC_DBG_HANDSHAKE_SENDER_ENTER = 0x5E5EAA11;       // post-retrain, sender-side spin starting
constexpr uint32_t FABRIC_DBG_HANDSHAKE_SENDER_DONE = 0x5E5EAA12;        // post-retrain, sender-side returned
constexpr uint32_t FABRIC_DBG_HANDSHAKE_RECV_ENTER = 0x5E5EAA21;         // post-retrain, receiver-side spin starting
constexpr uint32_t FABRIC_DBG_HANDSHAKE_RECV_DONE = 0x5E5EAA22;          // post-retrain, receiver-side returned
constexpr uint32_t FABRIC_DBG_HANDSHAKE_INIT_SENDER_ENTER = 0x5E5EAA13;  // init (boot), sender-side spin starting
constexpr uint32_t FABRIC_DBG_HANDSHAKE_INIT_SENDER_DONE = 0x5E5EAA14;   // init (boot), sender-side returned
constexpr uint32_t FABRIC_DBG_HANDSHAKE_INIT_RECV_ENTER = 0x5E5EAA23;    // init (boot), receiver-side spin starting
constexpr uint32_t FABRIC_DBG_HANDSHAKE_INIT_RECV_DONE = 0x5E5EAA24;     // init (boot), receiver-side returned

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
//   [1] TXQ_CTRL            low16 = TXQ0, high16 = TXQ1   (packet_resend_mode_active = bit0)
//   [2] RXQ_CTRL            low16 = RXQ0, high16 = RXQ1   (packet_mode = bit1)
//   [3] MAC_RX_ADDR_ROUTING raw   -- type->RXQ steering (prime suspect)
//   [4] MAC_RX_ROUTING      raw
//   [5] TXPKT_CFG_SEL_SW    low16 = TXQ0, high16 = TXQ1
//   [6] TXPKT_CFG_SEL_HW    low16 = TXQ0, high16 = TXQ1
//   [7] TXQ1 MAC_DA_HI      raw
//   [8] TXQ1 MAC_DA_LO      raw
//   [9] low16 = DBG_REG_D, high16 = DBG_REG_H  -- set at init by
//       eth_txq_reg_write(ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD), NOT by eth_enable_packet_mode(), so the
//       recovery-path config restore does NOT re-apply it. Prime suspect: does a retrain reset it?
// Expected post-init values: [1] bit0 set; [2] bit1 set; [3] routing bits (bcast->RXQ0, mcast->RXQ1);
// [4] 0; [5] TXQ0=0, TXQ1=0x111; [6] TXQ0=0, TXQ1=1; [7]/[8] TXQ1 mcast MAC (MCAST_MAC_ADDR halves);
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
    }

    if (eth_link_recovery_ptr != 0) {
        reinterpret_cast<void (*)()>(eth_link_recovery_ptr)();
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
        // 1s settle. Also clears the update_boot_results_eth_link_status_check() 1000ms debounce so the FW
        // link-status check below actually runs. Inside the context switch (before the main loop resumes).
        eth_wait_cycles(1000 * ETH_CLOCK_CYCLE_1MS);  // 1s (ETH_CLOCK_CYCLE_1MS = 1e6 cycles = 1ms)
        // FW link-status check -- now past its 1s debounce (the wait above), so this actually invokes it.
        update_boot_results_eth_link_status_check();
        // Second link-recovery pass (same entry point as [#1]). Null-guarded identically.
        if (eth_link_recovery_ptr != 0) {
            reinterpret_cast<void (*)()>(eth_link_recovery_ptr)();
        }
        // receiver_txq_id == 1 in the fabric router (see static_assert in kernel_main). Hardcoded 1
        // because receiver_txq_id is a kernel-TU constant not visible in this base header.
        eth_enable_packet_mode(1);
        // [ACCEPT_AHEAD RESTORE] eth_enable_packet_mode() does NOT touch ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD.
        // The retrain resets it from the configured depth (32) to DBG_VAL_G on the downed/retrained end, and
        // nothing else in the recovery path restores it -- leaving the TXQ with a shallower accept-ahead
        // depth that throttles TX pipelining for the rest of the run (the residual tail-stall /
        // lost-in-flight source). Re-write both TXQs here, mirroring the init path in
        // initialize_state_for_txq1_active_mode(). Value hardcoded to 32 (= DEFAULT_NUM_ETH_TXQ_DATA_PACKET_
        // ACCEPT_AHEAD, a kernel-TU constant not visible in this base header -- same reason receiver_txq_id
        // is hardcoded to 1 above).
        eth_txq_reg_write(0, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, 32);
        eth_txq_reg_write(1, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, 32);
        // [POST-RETRAIN HANDSHAKE] Config is now restored; bump the L1 retrain counter. The router's
        // coordinated context switch brackets this recovery pass with a before/after read of the counter,
        // sees it advance, and runs the post-retrain handshake before resuming traffic.
        fabric_inc_retrain_count();
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
