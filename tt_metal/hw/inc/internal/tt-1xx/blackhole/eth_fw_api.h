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
// For WATCHER_RING_BUFFER_PUSH(), used by recover_eth_link_if_down(). The macro is a no-op unless the
// watcher is enabled (TT_METAL_WATCHER), so this costs nothing in production. Guarded to device builds
// because host translation units (e.g. the HAL) also include this header and lack the ring-buffer
// include path / build flags.
#include "api/debug/ring_buffer.h"
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

// Marker pushed to the watcher ring buffer when ERISC0 detects its link is down and invokes FW
// recovery. The 0xD0... prefix reads as "link DOWN" in the watcher dump (cf. the existing
// MEM_SYSENG_ETH_MSG_DONE 0xD0E5 convention). One entry is pushed per recovery invocation, so a
// persistently-down link shows repeated codes. The ring buffer is per-eth-core, so the watcher log
// already identifies which core observed the down event.
constexpr uint32_t ETH_LINK_DOWN_RING_BUF_CODE = 0xD09D0000;

// Marker pushed once when ERISC0 observes its link transition back UP after having been down (i.e. FW
// recovery succeeded). 0x600D reads as "GOOD" in the watcher dump, pairing with the 0xD09D "DOWN"
// marker above, so a down->recovered cycle reads as ... D09D D09D ... 600D in the per-core ring buffer.
constexpr uint32_t ETH_LINK_UP_RING_BUF_CODE = 0x600D0000;

// Pushed immediately before ERISC0 actually invokes the FW link-recovery entry point (i.e. the pointer
// was non-null). 0xCA11ED reads as "CALLED" in the watcher dump, confirming the recovery function was
// reached. Seeing this interleaved with 0xD09D means we are calling recovery on every down poll.
constexpr uint32_t ETH_LINK_RECOVERY_CALLED_RING_BUF_CODE = 0xCA11ED00;

// Pushed when the FW link-recovery pointer is null (recovery unsupported on this FW), so the call is
// skipped. 0xDEAD reads as "dead/absent pointer" in the watcher dump. If this shows up instead of
// 0xCA11ED, the FW API table never populated eth_link_recovery_ptr.
constexpr uint32_t ETH_LINK_RECOVERY_UNAVAIL_RING_BUF_CODE = 0xDEAD0000;

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
    // Per-core down/up edge state (single firmware TU -> one instance per eth core). Constant-
    // initialized to false, so no static-init guard variable is emitted.
    static bool eth_link_was_down = false;
    // if (!is_link_up()) {
    if (true) {
        // volatile eth_status_t* eth_status = (volatile eth_status_t*)(MEM_SYSENG_ETH_STATUS);
        // invalidate_l1_cache();  // train_status is FW-written; read a fresh value, not a stale cache line
        // if (eth_status->train_status == link_train_status_e::LINK_TRAIN_REQUESTED_DOWN) {  //
        // LINK_TRAIN_REQUESTED_DOWN
        //     // Convert an administrative down into a recoverable training-failure state so the FW recovery
        //     // path below isn't precluded from running.
        //     eth_status->train_status =
        //         link_train_status_e::LINK_TRAIN_TIMEOUT_MANUAL_EQ;  // LINK_TRAIN_TIMEOUT_MANUAL_EQ
        //     eth_status->port_status = port_status_e::PORT_UP;
        // }
        // Record the down-link/recovery event for the watcher (no-op unless the watcher is enabled).
        WATCHER_RING_BUFFER_PUSH(ETH_LINK_DOWN_RING_BUF_CODE);
        eth_link_was_down = true;
        // The link-recovery entry point is optional in the FW API table: base/older FW may leave it
        // null. Gate on a non-zero pointer -- calling through a null here would jump to address 0 and
        // hang the core. When unsupported we still record the down edge above but take no action.
        const uint32_t eth_link_recovery_ptr =
            (uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->eth_link_recovery_ptr);
        if (eth_link_recovery_ptr != 0) {
            // Resume-phase: about to enter FW recovery. If the word stays here, we wedged in the call.
            fabric_dbg_set_resume_phase(RESUME_PHASE_RETRAIN_ENTER);
            reinterpret_cast<void (*)()>(eth_link_recovery_ptr)();
            WATCHER_RING_BUFFER_PUSH(ETH_LINK_RECOVERY_CALLED_RING_BUF_CODE);
            // Resume-phase: recovery returned (link retrained). The TX/RX paths advance from here.
            fabric_dbg_set_resume_phase(RESUME_PHASE_RETRAIN_DONE);
        } else {
            // FW does not provide a recovery entry point; record that we skipped the call.
            WATCHER_RING_BUFFER_PUSH(ETH_LINK_RECOVERY_UNAVAIL_RING_BUF_CODE);
        }
    } else if (eth_link_was_down) {
        // Link came back up after a down/recovery sequence -- record the recovery edge once.
        WATCHER_RING_BUFFER_PUSH(ETH_LINK_UP_RING_BUF_CODE);
        eth_link_was_down = false;
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
