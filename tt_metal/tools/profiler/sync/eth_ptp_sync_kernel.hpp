// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The wall-clock sync round trip (Mo's sync protocol) with the Blackhole 1588 hardware sampled alongside:
// every packet is sent with TS_CMD = two-step, so the MAC pushes {tag, egress time} into its FIFO, and every
// received frame's ingress time lands in the RX classifier's timestamp FIFO. Per round each side records the
// hardware egress/ingress stamps of its own packet and the peer's, and a PTP-timer read adjacent to each of its
// software wall-clock stamps, so the software path (command issue -> MAC, MAC -> software observation) is
// measured in nanoseconds on the same chip with no cross-chip mapping involved.

#pragma once

#include <cstdint>
#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "tools/profiler/sync/eth_wallclock_sync_types.hpp"
#include "tools/profiler/sync/eth_ptp_link.hpp"
#include "tools/profiler/sync/eth_ptp_sync_types.hpp"

namespace tt::tt_metal::eth_ptp {

namespace detail {

using raw::rd;
using raw::wr;
constexpr uint32_t kWallClockL = kWallClockLo;
constexpr uint32_t kWallClockH = kWallClockHi;
inline __attribute__((always_inline)) void read_wall_clock(uint32_t& hi, uint32_t& lo) {
    lo = *reinterpret_cast<volatile uint32_t*>(kWallClockL);  // latches H
    hi = *reinterpret_cast<volatile uint32_t*>(kWallClockH);
}
inline __attribute__((always_inline)) uint64_t now64() {
    uint32_t hi, lo;
    detail::read_wall_clock(hi, lo);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

inline volatile eth_sync::EthSyncResult* result_at(uint32_t addr) {
    return reinterpret_cast<volatile eth_sync::EthSyncResult*>(addr);
}
inline eth_sync::EthSyncSample* samples_at(uint32_t addr) {
    return reinterpret_cast<eth_sync::EthSyncSample*>(addr + sizeof(eth_sync::EthSyncResult));
}
inline void publish(uint32_t addr, uint32_t status, uint32_t n_done) {
    volatile eth_sync::EthSyncResult* r = result_at(addr);
    r->n_samples = n_done;
    r->status = status;  // written last: the host reads status, then samples
}

inline volatile PtpResult* presult(uint32_t addr) { return reinterpret_cast<volatile PtpResult*>(addr); }
inline PtpSample* psamples(uint32_t addr) { return reinterpret_cast<PtpSample*>(addr + sizeof(PtpResult)); }

inline void split(uint64_t v, volatile uint32_t& lo, volatile uint32_t& hi) {
    lo = static_cast<uint32_t>(v);
    hi = static_cast<uint32_t>(v >> 32);
}
inline void split(uint64_t v, uint32_t& lo, uint32_t& hi) {
    lo = static_cast<uint32_t>(v);
    hi = static_cast<uint32_t>(v >> 32);
}

// Raw FIFO dumps for the first rounds: MAC entries as 4 words, RX entries as 3 words, so the word order of
// both FIFOs is settled by data rather than by the register description.
constexpr uint32_t kRawMacOff = 0x20000, kRawRxOff = 0x21000, kRawMax = 48;  // past 2048 samples
inline volatile uint32_t* raw_words(uint32_t ptp_addr, uint32_t off) {
    return reinterpret_cast<volatile uint32_t*>(ptp_addr + off);
}
inline uint32_t g_raw_mac = 0, g_raw_rx = 0;

inline void lazy_gap(uint32_t flags) {
    if (flags & PTP_FLAG_LAZY_POLL) {
        const uint64_t until = now64() + 6750;  // ~5 us at 1.35 GHz
        while (now64() < until) {
        }
    }
}

inline uint32_t sync_txq(uint32_t flags) { return (flags & (PTP_FLAG_TXQ2 | PTP_FLAG_TCAM_LABEL)) ? 2u : 0u; }

inline bool txq_idle_bounded(uint32_t q, uint64_t deadline) {
    while (internal_::eth_txq_is_busy(q)) {
        if (detail::now64() >= deadline) {
            return false;
        }
    }
    return true;
}
inline uint32_t frame_words(uint32_t flags) { return 1u + ((flags >> PTP_FLAG_FRAME_WORDS_SHIFT) & 7u); }
inline void send_sync_frame(uint32_t q, uint32_t channel_addr, uint32_t words = 1) {
    internal_::eth_send_packet(q, channel_addr >> 4, channel_addr >> 4, words);
}

// PTP_FLAG_COUNTER_TRACE: what the queue's counters, its status word and the MAC FIFO do around one frame. Each
// iteration reads the FIFO flag, PKT_START_CNT, PKT_END_CNT and STATUS in that order; the first iteration and wall
// cycle at which each changed are kept, so the host can order the events to within one iteration (~60 cycles).
struct CounterTrace {
    uint32_t start0, end0, word0;
    uint32_t it_start = 0, it_fifo = 0, it_end = 0;
    uint32_t t_start = 0, t_fifo = 0, t_end = 0;
    uint32_t status_or = 0;
    uint32_t iters = 0;
};
inline void trace_begin(CounterTrace& t, uint32_t q) {
    t.start0 = rd(txq_reg(q, ETH_TXQ_PKT_START_CNT));
    t.end0 = rd(txq_reg(q, ETH_TXQ_PKT_END_CNT));
    t.word0 = rd(txq_reg(q, ETH_TXQ_WORD_CNT));
}
// Polls until the frame has started, its stamp is in the FIFO and it has ended, or 4096 iterations.
inline void trace_follow(CounterTrace& t, uint32_t q, uint32_t t_cmd) {
    for (t.iters = 1; t.iters <= 4096; t.iters++) {
        if (t.it_fifo == 0 && raw::mac_tx_fifo_not_empty()) {
            t.it_fifo = t.iters;
            t.t_fifo = rd(kWallClockL) - t_cmd;
        }
        if (t.it_start == 0 && rd(txq_reg(q, ETH_TXQ_PKT_START_CNT)) != t.start0) {
            t.it_start = t.iters;
            t.t_start = rd(kWallClockL) - t_cmd;
        }
        if (t.it_end == 0 && rd(txq_reg(q, ETH_TXQ_PKT_END_CNT)) != t.end0) {
            t.it_end = t.iters;
            t.t_end = rd(kWallClockL) - t_cmd;
        }
        t.status_or |= rd(txq_reg(q, ETH_TXQ_STATUS));
        if (t.it_fifo != 0 && t.it_start != 0 && t.it_end != 0) {
            break;
        }
    }
}

// The handshake rides erisc_info->channels[0].bytes_sent, the eth firmware's own channel state, and every
// wait calls run_routing(), the firmware's link service; without it the packets never move.
inline bool eth_wait_for_bytes_bounded(uint32_t num_bytes, uint64_t deadline) {
    while (erisc_info->channels[0].bytes_sent != num_bytes) {
        invalidate_l1_cache();
        run_routing();
        if (detail::now64() >= deadline) {
            return false;
        }
    }
    return true;
}
inline bool eth_wait_for_receiver_done_bounded(uint64_t deadline) {
    const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&erisc_info->channels[0].bytes_sent));
    internal_::eth_send_packet(0, addr >> 4, addr >> 4, 1);
    while (erisc_info->channels[0].bytes_sent != 0) {
        invalidate_l1_cache();
        run_routing();
        if (detail::now64() >= deadline) {
            return false;
        }
    }
    return true;
}
inline bool handshake_bounded(uint32_t handshake_addr, bool is_sender, uint64_t deadline) {
    if (is_sender) {
        if (!txq_idle_bounded(0, deadline)) {
            return false;
        }
        eth_send_bytes(handshake_addr, handshake_addr, 16);
        return eth_wait_for_receiver_done_bounded(deadline);
    }
    if (!eth_wait_for_bytes_bounded(16, deadline)) {
        return false;
    }
    eth_receiver_channel_done(0);
    return true;
}
inline uint32_t g_seq_prev[kNumTxq] = {0, 0, 0};
inline uint32_t g_want_label =
    0xFFFFFFFFu;  // TH FIFO label word: [4:0] flow label, [5] always set; all-ones = accept any
inline raw::TxHeaderPrev g_tx_hdr_prev{};

inline bool label_ok(uint32_t lb) {
    return (lb & kRxThLabelValid) && (g_want_label == 0xFFFFFFFFu || (lb & 0x1Fu) == g_want_label);
}

// Drains the MAC FIFO. The entry whose tag word (in either half) equals want_lo is kept; bit 16 of the
// returned diag says one was found. Words are stored as read: w0/w1 into tag lo/hi, w2/w3 into ts lo/hi.
inline uint32_t drain_mac_fifo(uint32_t want_lo, uint32_t want_hi, uint32_t* w, uint32_t ptp_addr, bool dump) {
    uint32_t n = 0;
    bool found = false;
    while (n < 255) {
        const uint32_t w0 = rd(kMacTsFifo0);
        if (w0 == 0xFFFFFFFFu) {
            break;
        }
        const uint32_t w1 = rd(kMacTsFifo1), w2 = rd(kMacTsFifo2), w3 = rd(kMacTsFifo3);
        n++;
        if (dump && g_raw_mac < kRawMax) {
            volatile uint32_t* d = raw_words(ptp_addr, kRawMacOff) + 4 * g_raw_mac++;
            d[0] = w0;
            d[1] = w1;
            d[2] = w2;
            d[3] = w3;
        }
        const bool match = (w0 == want_lo && w1 == want_hi) || (w1 == want_lo && w0 == want_hi);
        if (!found) {
            w[0] = w0;
            w[1] = w1;
            w[2] = w2;
            w[3] = w3;
            found = found || match;
        }
    }
    return (n & 0xFF) << 8 | (found ? 1u << 16 : 0u);
}

// Pops one RX FIFO entry into lo/hi/label; false when empty. The pop bit is driven as a 1-then-0 strobe and
// the status word is sampled before, between and after, so the dump shows what actually advanced the FIFO.
inline bool rx_pop_raw(bool pop_after, uint32_t& lo, uint32_t& hi, uint32_t& label, uint32_t* st) {
    st[0] = rd(kRxThStatus);
    if (st[0] & kRxThStatusEmpty) {
        return false;
    }
    if (pop_after) {
        wr(kRxThStatus, kRxThStatusPop);
        wr(kRxThStatus, 0);
    }
    lo = rd(kRxThTsLow);
    hi = rd(kRxThTsHigh);
    label = rd(kRxThTsLabel);
    if (!pop_after) {
        wr(kRxThStatus, kRxThStatusPop);
        st[1] = rd(kRxThStatus);
        wr(kRxThStatus, 0);
    } else {
        st[1] = rd(kRxThStatus);
    }
    st[2] = rd(kRxThStatus);
    return true;
}

// Drains the RX FIFO. Keeps the entry with the largest stamp not after obs_ptp (the frame the software just
// observed), falling back to the newest. Bit 17 of the returned diag says one was found.
inline uint32_t drain_rx_fifo(
    bool pop_after,
    uint64_t obs_ptp,
    uint64_t& rx_ts,
    uint32_t& label,
    uint32_t ptp_addr,
    bool dump,
    bool have_prev = false,
    uint64_t prev_ts = 0,
    uint32_t prev_label = 0) {
    uint32_t n = 0;
    bool found = have_prev, found_below = have_prev && prev_ts <= obs_ptp;
    uint64_t best_below = found_below ? prev_ts : 0, newest = have_prev ? prev_ts : 0;
    uint32_t best_label = prev_label, newest_label = prev_label;
    uint32_t lo = 0, hi = 0, lb = 0, st[3] = {0, 0, 0};
    while (n < 64 && rx_pop_raw(pop_after, lo, hi, lb, st)) {
        n++;
        if (dump && g_raw_rx < kRawMax) {
            volatile uint32_t* d = raw_words(ptp_addr, kRawRxOff) + 6 * g_raw_rx++;
            d[0] = st[0];
            d[1] = lo;
            d[2] = hi;
            d[3] = lb;
            d[4] = st[1];
            d[5] = st[2];
        }
        const uint64_t ts = (static_cast<uint64_t>(hi) << 32) | lo;
        if (!label_ok(lb)) {
            continue;
        }
        if (!found || ts > newest) {
            newest = ts;
            newest_label = lb;
        }
        if (ts <= obs_ptp && (!found_below || ts > best_below)) {
            best_below = ts;
            best_label = lb;
            found_below = true;
        }
        found = true;
    }
    if (found) {
        rx_ts = found_below ? best_below : newest;
        label = found_below ? best_label : newest_label;
    }
    return (n & 0xFF) | (found ? 1u << 17 : 0u) | (found_below ? 1u << 18 : 0u);
}

// Bounded wait for the RX FIFO to hold something: the RX queue's L1 write is cut-through and can be observed
// before the classifier has pushed the frame's stamp.
inline void rx_wait_nonempty(uint32_t spins) {
    for (uint32_t i = 0; i < spins && (rd(kRxThStatus) & kRxThStatusEmpty); i++) {
    }
}

// Pops everything currently in the RX FIFO, remembering the newest valid entry. Called while waiting for a
// frame: the frame's stamp is pushed at start-of-frame, before the RX queue's L1 write is visible, so the
// stamp we want may already have been popped here by the time the wait loop sees the frame.
inline void rx_discard_all(uint64_t& newest, uint32_t& newest_label, bool& have) {
    uint32_t lo, hi, lb, st[3];
    uint32_t n = 0;
    while (n < 64 && rx_pop_raw(false, lo, hi, lb, st)) {
        n++;
        if (label_ok(lb)) {
            newest = (static_cast<uint64_t>(hi) << 32) | lo;
            newest_label = lb;
            have = true;
        }
    }
}
inline void rx_discard_all() {
    uint64_t ts = 0;
    uint32_t lb = 0;
    bool have = false;
    rx_discard_all(ts, lb, have);
}

// PTP_FLAG_KEEPALIVE_TRACE: when, relative to its hand-off, a keepalive samples the stamp request. Each sample arms
// the request at whatever phase of the keepalive cycle the loop is at, waits for the next hand-off (PKT_START_CNT)
// and looks for its FIFO entry: the arming lead below which no entry appears is the sampling point.
// PTP_FLAG_FRAME_ARM_SWEEP does the same for the kernel's own frames, arming a swept delay after the command.
// Per sample: mac_tag_lo = cycles from the arming to the hand-off's count (frame sweep: the swept delay),
// mac_tag_hi = 1 if stamped, mac_tx_lo = cycles from the count to the entry, mac_tx_hi = cycles from the command to
// the count (frame sweep), th_rx_lo = entries drained.
inline void arm_trace_sample(uint32_t q, uint32_t i, PtpSample& out, int32_t frame_delay, uint32_t channel_addr) {
    uint32_t junk[4];
    drain_mac_fifo(0xFFFFFFFFu, 0xFFFFFFFFu, junk, 0, false);
    const uint32_t s0 = rd(txq_reg(q, ETH_TXQ_PKT_START_CNT));
    uint32_t t_cmd = 0;
    if (frame_delay >= 0) {
        t_cmd = rd(kWallClockL);
        send_sync_frame(q, channel_addr, 1);
        while (rd(kWallClockL) - t_cmd < static_cast<uint32_t>(frame_delay)) {
        }
    }
    const uint32_t t_arm = rd(kWallClockL);
    raw::txq_request_two_step(q, 0x5000'0000'0000'0000ull | i);
    uint32_t t_start = t_arm;
    for (uint32_t spin = 0; spin < 100000; spin++) {
        if (rd(txq_reg(q, ETH_TXQ_PKT_START_CNT)) != s0) {
            t_start = rd(kWallClockL);
            break;
        }
    }
    uint32_t t_fifo = 0;
    bool stamped = false;
    for (uint32_t spin = 0; spin < 64; spin++) {
        if (raw::mac_tx_fifo_not_empty()) {
            t_fifo = rd(kWallClockL);
            stamped = true;
            break;
        }
    }
    raw::txq_clear_timestamp_cmd(q);
    uint32_t w[4] = {0, 0, 0, 0};
    const uint32_t diag = drain_mac_fifo(i, 0x50000000u, w, 0, false);
    out.mac_tag_lo = frame_delay >= 0 ? static_cast<uint32_t>(frame_delay) : t_start - t_arm;
    out.mac_tag_hi = stamped && (diag & (1u << 16)) ? 1u : 0u;
    out.mac_tx_lo = stamped ? t_fifo - t_start : 0u;
    out.mac_tx_hi = frame_delay >= 0 ? t_start - t_cmd : 0u;
    out.th_rx_lo = (diag >> 8) & 0xFF;
    out.th_rx_hi = 0;
    out.th_label = 0;
    out.ptp_a_lo = out.ptp_a_hi = out.ptp_b_lo = out.ptp_b_hi = 0;
    out.diag = diag;
}

// PTP_FLAG_BLEED_TEST: whether a frame of another queue leaving the MAC right behind (even samples) or right ahead
// (odd) of an armed queue-2 frame is stamped under queue 2's tag. Per sample: mac_tag_lo = stamps under the tag,
// mac_tag_hi = entries popped, mac_tx_lo/hi = second stamp minus the first (ns, two words), th_rx_lo = queue-0
// hand-offs counted, th_rx_hi = queue-2 hand-offs counted, th_label = cycles from the first command to the second.
inline void bleed_sample(uint32_t q, uint32_t i, PtpSample& out, uint32_t channel_addr, uint32_t scratch_addr) {
    uint32_t junk[4];
    drain_mac_fifo(0xFFFFFFFFu, 0xFFFFFFFFu, junk, 0, false);
    const uint32_t s0_q0 = rd(txq_reg(0, ETH_TXQ_PKT_START_CNT));
    const uint32_t s0_q2 = rd(txq_reg(q, ETH_TXQ_PKT_START_CNT));
    while (internal_::eth_txq_is_busy(0) || internal_::eth_txq_is_busy(q)) {
    }
    raw::txq_request_two_step(q, 0x5000'0000'0000'0000ull | i);
    const uint32_t t0 = rd(kWallClockL);
    if ((i & 1) == 0) {
        internal_::eth_send_packet<false>(q, channel_addr >> 4, channel_addr >> 4, 1);
        internal_::eth_send_packet<false>(0, scratch_addr >> 4, scratch_addr >> 4, 1);
    } else {
        internal_::eth_send_packet<false>(0, scratch_addr >> 4, scratch_addr >> 4, 1);
        internal_::eth_send_packet<false>(q, channel_addr >> 4, channel_addr >> 4, 1);
    }
    const uint32_t t1 = rd(kWallClockL);
    for (uint32_t spin = 0; spin < 512; spin++) {
        rd(kWallClockL);
    }
    raw::txq_clear_timestamp_cmd(q);
    uint64_t ts[4] = {0, 0, 0, 0};
    uint32_t tagged = 0, popped = 0;
    raw::MacTxStamp e;
    while (raw::mac_tx_fifo_pop(e)) {
        if (static_cast<uint32_t>(e.tag) == i && popped < 4) {
            ts[tagged] = e.tx_ts;
            tagged++;
        }
        popped++;
    }
    out.mac_tag_lo = tagged;
    out.mac_tag_hi = popped;
    split(tagged >= 2 ? ts[1] - ts[0] : 0, out.mac_tx_lo, out.mac_tx_hi);
    out.th_rx_lo = rd(txq_reg(0, ETH_TXQ_PKT_START_CNT)) - s0_q0;
    out.th_rx_hi = rd(txq_reg(q, ETH_TXQ_PKT_START_CNT)) - s0_q2;
    out.th_label = t1 - t0;
    out.ptp_a_lo = out.ptp_a_hi = out.ptp_b_lo = out.ptp_b_hi = 0;
    out.diag = 0;
}

inline void snapshot_start(volatile PtpResult* r, uint32_t flags) {
    r->magic = kPtpMagic;
    r->status = 0;
    r->n_samples = 0;
    r->timer_ctrl_before = rd(kPtpTimerCtrl);
    r->pti_stat_before = rd(kPtpPtiStat);
    r->tx_mac_cfg_before = rd(kMacTxCfg);
    r->rate_sel = rd(0xFFBA2004);
    r->th_status_before = rd(kRxThStatus);
    r->pad[0] = rd(kTxqRegsBase + 0x00);  // TXQ0 CTRL
    r->pad[1] = rd(kTxqRegsBase + 0x48);  // TXQ0 REMOTE_SEQ_TIMEOUT
    r->pad[2] = rd(kTxqRegsBase + kTxqLocalSeqUpdateTimeoutOff);
    r->pad[3] = rd(kTxqRegsBase + 0x1000);
    r->pad[4] = rd(kTxqRegsBase + 0x1000 + kTxqLocalSeqUpdateTimeoutOff);
    r->pad[5] = rd(kTxqRegsBase + 0x2000);
    r->pad[6] = rd(kTxqRegsBase + 0x2000 + kTxqLocalSeqUpdateTimeoutOff);
    r->pti_acked = (flags & PTP_FLAG_NO_TIMER_START)
                       ? 2u
                       : (ptp_timer_start(kPtiRefclk, kTimerLeadTicks, kTimerAckSpins) ? 1u : 0u);
    raw::RxThPrev prev{rd(kRxFlNoMatchActions), rd(kRxFdOverrideDecision)};
    if (flags & PTP_FLAG_TCAM_LABEL) {
        wr(kRxThStatus, kRxThStatusFlush);
        wr(kRxFlNoMatchActions, prev.no_match_actions & ~kRxFlKeepTimestamp);
        raw::rx_stamp_rule_install(kLinkTcamRow, kLinkLabel);
        g_tx_hdr_prev = raw::tx_header_row_install(sync_txq(flags), kLinkHeaderRow, kStampFrameDa);
        g_want_label = kLinkLabel;
    } else {
        prev = raw::rx_timestamps_enable_all((flags & PTP_FLAG_SET_OVERRIDE) != 0);
        g_want_label = 0xFFFFFFFFu;
    }
    r->no_match_prev = prev.no_match_actions;
    r->override_prev = prev.override_decision;
    r->pad[7] = g_tx_hdr_prev.sel_sw;
    if (flags & PTP_FLAG_MAC_FIFO_ALL_PACKETS) {
        wr(kMacTxCfg, rd(kMacTxCfg) | kMacTxCfgTsFifoEnb);
    }
    if (flags & PTP_FLAG_QUIET_LINK) {
        raw::txq_keepalives_off(g_seq_prev);
    }
    uint32_t junk[4];
    drain_mac_fifo(0xFFFFFFFFu, 0xFFFFFFFFu, junk, 0, false);
    rx_discard_all();
    uint32_t hi, lo;
    split(read_cfr(), r->cfr_start_lo, r->cfr_start_hi);
    detail::read_wall_clock(hi, lo);
    r->wall_start_lo = lo;
    r->wall_start_hi = hi;
    split(read_ptp64ns(), r->ptp_start_lo, r->ptp_start_hi);
}

inline void snapshot_end(volatile PtpResult* r, uint32_t flags, uint32_t n_done) {
    uint32_t hi, lo;
    split(read_cfr(), r->cfr_end_lo, r->cfr_end_hi);
    detail::read_wall_clock(hi, lo);
    r->wall_end_lo = lo;
    r->wall_end_hi = hi;
    split(read_ptp64ns(), r->ptp_end_lo, r->ptp_end_hi);
    raw::txq_clear_timestamp_cmd(sync_txq(flags));
    if (flags & PTP_FLAG_MAC_FIFO_ALL_PACKETS) {
        wr(kMacTxCfg, rd(kMacTxCfg) & ~kMacTxCfgTsFifoEnb);
    }
    if (flags & PTP_FLAG_TCAM_LABEL) {
        raw::tx_header_row_restore(sync_txq(flags), kLinkHeaderRow, g_tx_hdr_prev);
        raw::rx_stamp_rule_remove(kLinkTcamRow);
    }
    raw::rx_timestamps_restore(raw::RxThPrev{r->no_match_prev, r->override_prev});
    if (flags & PTP_FLAG_QUIET_LINK) {
        raw::txq_keepalives_restore(g_seq_prev);
    }
    r->n_samples = n_done;
    r->status = 1;
}

}  // namespace detail

// Same protocol and sample layout as Mo's wallclock sync kernels, plus the hardware stamps.
inline bool ptp_sync_sender(
    uint32_t result_addr,
    uint32_t ptp_addr,
    uint32_t channel_addr,
    uint32_t handshake_addr,
    uint32_t n_samples,
    uint64_t timeout_cycles,
    uint32_t gap_cycles,
    uint32_t flags) {
    using namespace eth_sync;
    volatile EthSyncResult* res = detail::result_at(result_addr);
    EthSyncSample* samples = detail::samples_at(result_addr);
    volatile PtpResult* pres = detail::presult(ptp_addr);
    PtpSample* ps = detail::psamples(ptp_addr);
    res->magic = kEthSyncMagic;
    res->n_wanted = n_samples;
    res->n_samples = 0;
    res->status = ETH_SYNC_RUNNING;
    detail::snapshot_start(pres, flags);

    const uint64_t deadline = detail::now64() + timeout_cycles;
    volatile eth_channel_sync_t* sync = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);
    const bool pop_after = (flags & PTP_FLAG_TH_POP_AFTER) != 0;
    const uint32_t q = detail::sync_txq(flags);

    if (!detail::handshake_bounded(handshake_addr, /*is_sender=*/true, deadline)) {
        detail::publish(result_addr, ETH_SYNC_TIMEOUT_HANDSHAKE, 0);
        detail::snapshot_end(pres, flags, 0);
        return false;
    }
    uint32_t junk[4];
    detail::drain_mac_fifo(0xFFFFFFFFu, 0xFFFFFFFFu, junk, 0, false);
    detail::rx_discard_all();
    const bool trace = (flags & PTP_FLAG_COUNTER_TRACE) != 0;
    const uint32_t words = detail::frame_words(flags);
    uint32_t done = 0;
    if (flags & (PTP_FLAG_KEEPALIVE_TRACE | PTP_FLAG_FRAME_ARM_SWEEP | PTP_FLAG_BLEED_TEST)) {
        for (uint32_t i = 0; i < n_samples; i++) {
            detail::txq_idle_bounded(q, deadline);
            const int32_t delay = (flags & PTP_FLAG_FRAME_ARM_SWEEP) ? static_cast<int32_t>((i % 24) * 8) : -1;
            if (flags & PTP_FLAG_BLEED_TEST) {
                detail::bleed_sample(q, i, ps[i], channel_addr, handshake_addr);
            } else {
                detail::arm_trace_sample(q, i, ps[i], delay, channel_addr);
            }
            done = i + 1;
            pres->n_samples = done;
            // The gap grows 64 cycles a sample, so the arming walks the 8002-cycle keepalive period rather than
            // sitting at one phase of it.
            const uint64_t until = detail::now64() + gap_cycles + 64u * i;
            while (detail::now64() < until) {
            }
        }
        detail::publish(result_addr, ETH_SYNC_DONE, done);
        detail::snapshot_end(pres, flags, done);
        return true;
    }
    for (uint32_t i = 0; i < n_samples; i++) {
        if (!detail::txq_idle_bounded(q, deadline)) {
            detail::publish(result_addr, ETH_SYNC_TIMEOUT_TXQ, done);
            detail::snapshot_end(pres, flags, done);
            return false;
        }
        const bool dump = (flags & PTP_FLAG_RAW_DUMP) && i < 2;
        detail::rx_discard_all();
        raw::txq_request_two_step(q, 0x5000'0000'0000'0000ull | i);
        sync->bytes_sent = 1;
        sync->receiver_ack = 0;

        uint32_t t0_hi, t0_lo;
        detail::read_wall_clock(t0_hi, t0_lo);
        const uint64_t p0 = read_ptp64ns();
        detail::CounterTrace tr{};
        if (trace) {
            detail::trace_begin(tr, q);
        }
        const uint32_t t_cmd = rd(detail::kWallClockL);
        detail::send_sync_frame(q, channel_addr, words);
        if (trace) {
            detail::trace_follow(tr, q, t_cmd);
        }
        // Pop the egress stamp as soon as the frame has left and disarm: TS_CMD is sticky, and a queue that stays
        // armed past its idle timeout stamps its own keepalive under our tag.
        detail::txq_idle_bounded(q, deadline);
        for (uint32_t spin = 0; spin < 4096 && !raw::mac_tx_fifo_not_empty(); spin++) {
        }
        raw::txq_clear_timestamp_cmd(q);
        uint32_t w[4] = {0, 0, 0, 0};
        uint32_t diag = detail::drain_mac_fifo(i, 0x50000000u, w, ptp_addr, dump);

        bool ok = true;
        uint64_t prev_ts = 0;
        uint32_t prev_lb = 0;
        bool have_prev = false;
        while (sync->bytes_sent != 0) {
            invalidate_l1_cache();
            if (sync->bytes_sent != 0) {
                detail::rx_discard_all(prev_ts, prev_lb, have_prev);
                detail::lazy_gap(flags);
            }
            if (detail::now64() >= deadline) {
                ok = false;
                break;
            }
        }
        uint32_t t2_hi, t2_lo;
        detail::read_wall_clock(t2_hi, t2_lo);
        const uint64_t p2 = read_ptp64ns();
        if (!ok) {
            detail::publish(result_addr, ETH_SYNC_TIMEOUT_WAIT, done);
            detail::snapshot_end(pres, flags, done);
            return false;
        }

        samples[i].t0_hi = t0_hi;
        samples[i].t0_lo = t0_lo;
        samples[i].t2_hi = t2_hi;
        samples[i].t2_lo = t2_lo;
        samples[i].t1_hi = 0;
        samples[i].t1_lo = 0;

        uint64_t rx_ts = 0;
        uint32_t label = 0;
        diag |= detail::drain_rx_fifo(pop_after, p2, rx_ts, label, ptp_addr, dump, have_prev, prev_ts, prev_lb);
        ps[i].mac_tag_lo = w[0];
        ps[i].mac_tag_hi = w[1];
        ps[i].mac_tx_lo = w[2];
        ps[i].mac_tx_hi = w[3];
        detail::split(rx_ts, ps[i].th_rx_lo, ps[i].th_rx_hi);
        ps[i].th_label = label;
        detail::split(p0, ps[i].ptp_a_lo, ps[i].ptp_a_hi);
        detail::split(p2, ps[i].ptp_b_lo, ps[i].ptp_b_hi);
        ps[i].diag = diag;
        if (trace) {
            // Counter deltas are taken here, after the echo, so the frame has certainly ended.
            ps[i].mac_tag_lo = rd(txq_reg(q, ETH_TXQ_PKT_START_CNT)) - tr.start0;
            ps[i].mac_tag_hi = rd(txq_reg(q, ETH_TXQ_PKT_END_CNT)) - tr.end0;
            ps[i].mac_tx_lo = rd(txq_reg(q, ETH_TXQ_WORD_CNT)) - tr.word0;
            ps[i].mac_tx_hi = tr.status_or;
            ps[i].th_rx_lo = tr.t_start;
            ps[i].th_rx_hi = tr.t_fifo;
            ps[i].th_label = tr.t_end;
            ps[i].ptp_a_lo = tr.it_start;
            ps[i].ptp_a_hi = tr.it_fifo;
            ps[i].ptp_b_lo = tr.it_end;
            ps[i].ptp_b_hi = tr.iters;
        }
        done = i + 1;
        pres->n_samples = done;

        if (gap_cycles != 0 && i + 1 < n_samples) {
            const uint64_t next = ((static_cast<uint64_t>(t0_hi) << 32) | t0_lo) + gap_cycles;
            while (detail::now64() < next) {
                if (detail::now64() >= deadline) {
                    detail::publish(result_addr, ETH_SYNC_DONE, done);
                    detail::snapshot_end(pres, flags, done);
                    return true;
                }
            }
        }
    }
    detail::publish(result_addr, ETH_SYNC_DONE, done);
    detail::snapshot_end(pres, flags, done);
    return true;
}

inline bool ptp_sync_receiver(
    uint32_t result_addr,
    uint32_t ptp_addr,
    uint32_t channel_addr,
    uint32_t handshake_addr,
    uint32_t n_samples,
    uint64_t timeout_cycles,
    uint32_t flags) {
    using namespace eth_sync;
    volatile EthSyncResult* res = detail::result_at(result_addr);
    EthSyncSample* samples = detail::samples_at(result_addr);
    volatile PtpResult* pres = detail::presult(ptp_addr);
    PtpSample* ps = detail::psamples(ptp_addr);
    res->magic = kEthSyncMagic;
    res->n_wanted = n_samples;
    res->n_samples = 0;
    res->status = ETH_SYNC_RUNNING;
    detail::snapshot_start(pres, flags);

    const uint64_t deadline = detail::now64() + timeout_cycles;
    volatile eth_channel_sync_t* sync = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);
    const bool pop_after = (flags & PTP_FLAG_TH_POP_AFTER) != 0;
    const uint32_t q = detail::sync_txq(flags);
    sync->bytes_sent = 0;
    sync->receiver_ack = 0;

    if (!detail::handshake_bounded(handshake_addr, /*is_sender=*/false, deadline)) {
        detail::publish(result_addr, ETH_SYNC_TIMEOUT_HANDSHAKE, 0);
        detail::snapshot_end(pres, flags, 0);
        return false;
    }
    uint32_t junk[4];
    detail::rx_discard_all();
    detail::drain_mac_fifo(0xFFFFFFFFu, 0xFFFFFFFFu, junk, 0, false);

    uint32_t done = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        const bool dump = (flags & PTP_FLAG_RAW_DUMP) && i < 2;
        bool ok = true;
        uint64_t prev_ts = 0;
        uint32_t prev_lb = 0;
        bool have_prev = false;
        while (sync->bytes_sent == 0) {
            invalidate_l1_cache();
            if (sync->bytes_sent == 0) {
                detail::rx_discard_all(prev_ts, prev_lb, have_prev);
                detail::lazy_gap(flags);
            }
            if (detail::now64() >= deadline) {
                ok = false;
                break;
            }
        }
        uint32_t t1_hi, t1_lo;
        detail::read_wall_clock(t1_hi, t1_lo);
        const uint64_t p1 = read_ptp64ns();
        if (!ok) {
            detail::publish(result_addr, ETH_SYNC_TIMEOUT_WAIT, done);
            detail::snapshot_end(pres, flags, done);
            return false;
        }
        uint64_t rx_ts = 0;
        uint32_t label = 0;
        uint32_t diag = detail::drain_rx_fifo(pop_after, p1, rx_ts, label, ptp_addr, dump, have_prev, prev_ts, prev_lb);

        samples[i].t1_hi = t1_hi;
        samples[i].t1_lo = t1_lo;
        samples[i].t0_hi = 0;
        samples[i].t0_lo = 0;
        samples[i].t2_hi = 0;
        samples[i].t2_lo = 0;
        done = i + 1;

        sync->bytes_sent = 0;
        sync->receiver_ack = 0;
        if (!detail::txq_idle_bounded(q, deadline)) {
            detail::publish(result_addr, ETH_SYNC_TIMEOUT_TXQ, done);
            detail::snapshot_end(pres, flags, done);
            return false;
        }
        raw::txq_request_two_step(q, 0x5200'0000'0000'0000ull | i);
        const uint64_t p1b = read_ptp64ns();
        detail::send_sync_frame(q, channel_addr);
        // The egress stamp lands once the frame has left the MAC; the next round's message is at least a
        // link round trip away, so waiting for the queue and then draining costs nothing on the critical path.
        detail::txq_idle_bounded(q, deadline);
        for (uint32_t spin = 0; spin < 4096 && !raw::mac_tx_fifo_not_empty(); spin++) {
        }
        raw::txq_clear_timestamp_cmd(q);
        uint32_t w[4] = {0, 0, 0, 0};
        diag |= detail::drain_mac_fifo(i, 0x52000000u, w, ptp_addr, dump);

        ps[i].mac_tag_lo = w[0];
        ps[i].mac_tag_hi = w[1];
        ps[i].mac_tx_lo = w[2];
        ps[i].mac_tx_hi = w[3];
        detail::split(rx_ts, ps[i].th_rx_lo, ps[i].th_rx_hi);
        ps[i].th_label = label;
        detail::split(p1, ps[i].ptp_a_lo, ps[i].ptp_a_hi);
        detail::split(p1b, ps[i].ptp_b_lo, ps[i].ptp_b_hi);
        ps[i].diag = diag;
        pres->n_samples = done;
    }
    detail::publish(result_addr, ETH_SYNC_DONE, done);
    detail::snapshot_end(pres, flags, done);
    return true;
}

}  // namespace tt::tt_metal::eth_ptp
