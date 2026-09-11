// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hardware-stamped link half for the streaming profiler's d2d sync kernels. The exchange frames go out on their
// own TX queue and header row so the MAC stamps their egress and the RX classifier stamps their ingress under a
// label (eth_ptp_sync.hpp); stamps are reported in this chip's refclk tick count, the domain the host already fits.

#pragma once

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"
#include "tools/profiler/sync/eth_ptp_sync.hpp"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kLinkTxq = 2;  // fabric routers send on queue 0

struct LinkHwState {
    int64_t ns_minus_20cfr = 0;  // PTP64NS - 20 * CFR: the same 50 MHz refclk, offset by when the timer was started
    TxSyncHeaderPrev hdr_prev{};
    uint32_t no_match_prev = 0;
    uint32_t round = 0;
};

// PTP64NS - 20 * CFR. The two counters advance on the same 50 MHz edge, so the difference is one constant, a
// multiple of 20 ns, for as long as the timer runs; but a single pair of reads puts the register latency between
// them (a tick or more) into it, and a stall between the two reads puts in several -- measured once at kernel start,
// that sat in every stamp of one side for the whole run as an 80 ns link bias that came and went between launches.
// Each pair is read in both orders so the skew cancels in the sum, the median over pairs discards a stalled one, and
// the result is rounded to the tick.
inline int64_t measure_ns_minus_20cfr() {
    constexpr int kPairs = 16;
    int64_t sum2[kPairs];
    for (int i = 0; i < kPairs; i++) {
        const uint64_t c1 = read_cfr();
        const uint64_t n1 = read_ptp64ns();
        const uint64_t n2 = read_ptp64ns();
        const uint64_t c2 = read_cfr();
        const int64_t k1 = static_cast<int64_t>(n1) - static_cast<int64_t>(c1) * 20;
        const int64_t k2 = static_cast<int64_t>(n2) - static_cast<int64_t>(c2) * 20;
        int64_t v = k1 + k2;
        int j = i;
        for (; j > 0 && sum2[j - 1] > v; j--) {
            sum2[j] = sum2[j - 1];
        }
        sum2[j] = v;
    }
    const int64_t k = (sum2[kPairs / 2 - 1] + sum2[kPairs / 2]) / 4;
    return ((k + (k >= 0 ? 10 : -10)) / 20) * 20;
}

inline void link_hw_begin(LinkHwState& st) {
    ptp_timer_start(kPtiRefclk50MHz, 5000, 200000);
    st.ns_minus_20cfr = measure_ns_minus_20cfr();
    st.no_match_prev = rd(kRxFlNoMatchActions);
    wr(kRxThStatus, kRxThStatusFlush);
    wr(kRxFlNoMatchActions, st.no_match_prev & ~kRxFlKeepTimestamp);
    rx_sync_stamp_rule_install(kSyncTcamRow, kSyncLabel);
    st.hdr_prev = tx_sync_header_install(kLinkTxq, kSyncHeaderRow, kSyncFrameDa);
    txq_clear_timestamp_cmd(kLinkTxq);
    MacTxStamp junk;
    while (mac_tx_fifo_pop(junk)) {
    }
    st.round = 0;
}

inline void link_hw_end(const LinkHwState& st) {
    txq_clear_timestamp_cmd(kLinkTxq);
    tx_sync_header_restore(kLinkTxq, kSyncHeaderRow, st.hdr_prev);
    rx_sync_stamp_rule_remove(kSyncTcamRow);
    wr(kRxFlNoMatchActions, st.no_match_prev);
    wr(kRxThStatus, kRxThStatusFlush);
}

// A round is kTripsPerRound back-to-back exchanges whose stamps are averaged on each side: every stamp is quantised
// to the timer's 20 ns tick, the trips sit at different phases of it, so the mean's rounding noise falls by the
// square root of the count (8.8 -> ~0.4 ns per round at 256), and the averages are reported in quarter-ns units
// so that gain reaches the host whole. A round of 256 trips takes ~320 us of the link's 1 ms cadence.
constexpr uint32_t kTripsPerRound = 256;
constexpr uint32_t kHwUnitsPerNs = 4;
inline uint64_t link_hw_q(const LinkHwState& st, int64_t ns_sum, uint32_t count) {
    const int64_t elapsed_sum = ns_sum - st.ns_minus_20cfr * static_cast<int64_t>(count);
    const int64_t c = static_cast<int64_t>(count);
    return static_cast<uint64_t>((elapsed_sum * kHwUnitsPerNs + c / 2) / c);
}

// Sends one frame of `words` 16-byte words on the sync queue with a two-step stamp request and returns its egress
// stamp, 0 if the MAC never produced one. TS_CMD is sticky and is cleared as soon as the frame has left: a queue
// still armed at its idle timeout would stamp its own keepalive under the same tag. `before_issue` runs with the
// queue armed, right before the frame is issued: where a software stamp of the same instant belongs.
template <typename BeforeIssue>
inline uint64_t link_hw_send(uint64_t tag, uint32_t src_addr, uint32_t dst_addr, uint32_t words, BeforeIssue&& before_issue) {
    while (internal_::eth_txq_is_busy(kLinkTxq)) {
    }
    txq_request_two_step(kLinkTxq, tag);
    before_issue();
    internal_::eth_send_packet(kLinkTxq, src_addr >> 4, dst_addr >> 4, words);
    while (internal_::eth_txq_is_busy(kLinkTxq)) {
    }
    for (uint32_t spin = 0; spin < 4096 && !mac_tx_fifo_not_empty(); spin++) {
    }
    txq_clear_timestamp_cmd(kLinkTxq);
    MacTxStamp s;
    uint64_t ts = 0;
    bool found = false;
    while (mac_tx_fifo_pop(s)) {
        if (!found && s.tag == tag) {
            ts = s.tx_ts;
            found = true;
        }
    }
    return ts;
}
inline uint64_t link_hw_send(uint64_t tag, uint32_t src_addr, uint32_t dst_addr, uint32_t words) {
    return link_hw_send(tag, src_addr, dst_addr, words, [] {});
}

// Ingress stamps land in the RX FIFO at start-of-frame, before the frame's bytes are visible in L1, so the wait
// loop pops them as they come and take() hands over the newest once the frame has been seen.
struct LinkHwRx {
    uint64_t newest = 0;
    bool have = false;
};
inline void link_hw_rx_poll(LinkHwRx& t) {
    RxStamp s;
    while (rx_th_pop(s)) {
        if ((s.label & kRxThLabelValid) && (s.label & 0x1Fu) == kSyncLabel) {
            t.newest = s.rx_ts;
            t.have = true;
        }
    }
}
inline uint64_t link_hw_rx_take(LinkHwRx& t) {
    link_hw_rx_poll(t);
    const uint64_t ts = t.have ? t.newest : 0;
    t.have = false;
    return ts;
}

}  // namespace tt::tt_metal::eth_ptp
