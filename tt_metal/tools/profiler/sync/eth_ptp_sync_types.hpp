// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Layout of what the PTP sync kernels leave in eth L1, shared with the host reader.

#pragma once

#include <cstdint>

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kPtpMagic = 0x50545053;  // 'PTPS'

enum PtpFlags : uint32_t {
    PTP_FLAG_MAC_FIFO_ALL_PACKETS = 1u << 0,  // TX_MAC_CFG.tx_ts_fifo_enb instead of relying on TS_CMD
    PTP_FLAG_TH_POP_AFTER = 1u << 1,          // pop the RX timestamp FIFO before reading the head, not after
    PTP_FLAG_NO_TIMER_START = 1u << 2,        // leave the PTP timer as found
    PTP_FLAG_RAW_DUMP = 1u << 3,              // record every FIFO word popped in rounds 0 and 1
    PTP_FLAG_QUIET_LINK = 1u << 4,            // hold the TX queues' idle sequence-number updates off during the run
    PTP_FLAG_SET_OVERRIDE = 1u << 5,          // also set FD.OVERRIDE_DECISION = 2 (the original, unnecessary, enable)
    PTP_FLAG_TXQ2 = 1u << 6,                  // send the sync frames on TX queue 2 (fabric uses queue 0)
    PTP_FLAG_TCAM_LABEL = 1u << 7,            // own header row + TCAM rule: only sync frames get RX stamps, labeled
    PTP_FLAG_LAZY_POLL = 1u << 8,             // poll for frames only every ~5 us, as a core busy with other work would
};

struct PtpSample {
    uint32_t mac_tx_lo, mac_tx_hi;    // MAC egress stamp of the packet this side sent this round
    uint32_t mac_tag_lo, mac_tag_hi;  // tag the MAC echoed with it (the queue's RX_TIMESTAMP value)
    uint32_t th_rx_lo, th_rx_hi;      // MAC ingress stamp of the packet this side received this round
    uint32_t th_label;                // RX FIFO label word as read (bit 31 = valid)
    uint32_t ptp_a_lo, ptp_a_hi;      // PTP timer adjacent to this side's first wall-clock stamp (t0 / t1)
    uint32_t ptp_b_lo, ptp_b_hi;      // PTP timer adjacent to its second event (t2 / echo send)
    uint32_t diag;                    // [7:0] RX FIFO entries drained after arrival, [15:8] MAC FIFO entries drained,
                                      // [16] MAC stamp found with matching tag, [17] RX stamp found
};

struct PtpResult {
    uint32_t magic;
    uint32_t status;  // 0 running, 1 done
    uint32_t n_samples;
    uint32_t timer_ctrl_before, pti_stat_before, tx_mac_cfg_before, rate_sel, th_status_before;
    uint32_t pti_acked;
    uint32_t no_match_prev, override_prev;
    uint32_t cfr_start_lo, cfr_start_hi, wall_start_lo, wall_start_hi, ptp_start_lo, ptp_start_hi;
    uint32_t cfr_end_lo, cfr_end_hi, wall_end_lo, wall_end_hi, ptp_end_lo, ptp_end_hi;
    uint32_t pad[9];
    // Followed by n_wanted PtpSample entries.
};
static_assert(sizeof(PtpResult) == 128, "host reads a fixed 128-byte header");

}  // namespace tt::tt_metal::eth_ptp
