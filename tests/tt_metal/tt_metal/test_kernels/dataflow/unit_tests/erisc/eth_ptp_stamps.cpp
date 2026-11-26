// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The kernel for ActiveEthPtpStamps. It runs the stamped frame exchange on one port of a link, as either the
// transmitter or the receiver. At the end, the transmitter also sends a run of two-step stamped frames.

#include <atomic>
#include <cstdint>
#include <optional>

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "eth_ptp_stamps.hpp"

using namespace eth_ptp_stamps;

static_assert(kStampTickNs == eth_ptp::kNsPerRefclkTick);

constexpr bool kTransmitter = get_compile_time_arg_val(0) != 0;
// The Ethernet firmware sends only on TX queue 0 and uses only TX header rows 0 to 2, so this queue and header row are
// free. The TCAM row and label can be anything.
constexpr uint32_t kTxq = 2, kHeaderRow = 3, kTcamRow = 63, kLabel = 0x15;
// Each port of the test's link reads the arrival times of the stamped frames it receives from the RX timestamp FIFO,
// which records only the frames that a TCAM rule matches. Both ports send their stamped frames to this destination
// address and match it with that rule, so the FIFO holds only those frames' arrival times. No other frame is sent here,
// because the Ethernet firmware only sends to ff:ff:ff:ff:ff:ff, 01:00:00:00:00:00 and 02:00:00:00:00:00.
constexpr uint64_t kStampFrameDestination = 0x0211'2233'4455ull;
constexpr eth_ptp::RxTcamNonIpMatch kStampMatch = eth_ptp::rx_tcam_match_destination(kStampFrameDestination);
struct Frame {
    eth_ptp::FrameStampSlot stamp;
    uint32_t key;
    uint32_t echo_key;
};
static_assert(sizeof(Frame) <= kFrameBytes);
constexpr uint32_t kSpins = 1u << 20;

template <typename Pred>
bool wait_for(Pred&& pred) {
    for (uint32_t spin = 0; spin < kSpins; spin++) {
        invalidate_l1_cache();
        if (pred()) {
            return true;
        }
    }
    return false;
}

bool send(volatile tt_l1_ptr Frame* frame) {
    eth_ptp::clear_frame_stamp(frame->stamp);
    const uint32_t addr = reinterpret_cast<uint32_t>(frame);
    internal_::eth_send_packet<false>(kTxq, addr >> 4, addr >> 4, kFrameBytes >> 4);
    return wait_for([] { return !internal_::eth_txq_is_busy(kTxq); });
}

std::optional<uint64_t> take_ingress() {
    if (eth_ptp::rx_stamp_fifo::holds_exactly<1>()) {
        const eth_ptp::RxStampLabel label = eth_ptp::kRxStampLabel.read();
        if (label.valid && label.label == kLabel) {
            return eth_ptp::rx_stamp_fifo::pop();
        }
    }
    eth_ptp::rx_stamp_fifo::flush();
    return std::nullopt;
}

int32_t restart_error_ns() {
    const eth_ptp::ClocksLo update = eth_ptp::await_refclk_update();
    const uint64_t ptp_ns = eth_ptp::read_ptp_ns();
    return static_cast<int32_t>(static_cast<uint32_t>(ptp_ns) - update.refclk * eth_ptp::kNsPerRefclkTick);
}

void handshake(uint32_t base) {
    if constexpr (kTransmitter) {
        eth_send_bytes(base, base, kHandshakeBytes);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(kHandshakeBytes);
        eth_receiver_channel_done(0);
    }
}

// Sends up to kTwoStepFrames frames, each armed for a two-step stamp with its own tag. Returns how many frames got a TX
// stamp with the right tag and a time between the PTP times read before and after the send. Each frame's stamp is taken
// from the FIFO before the next frame is armed, so no frame can pick up another's tag.
uint32_t send_two_step(volatile tt_l1_ptr Frame* frame) {
    eth_ptp::tx_stamp_fifo::clear();
    uint32_t matched = 0;
    for (uint32_t tag = 0; tag < kTwoStepFrames; tag++) {
        const uint64_t before_ns = eth_ptp::read_ptp_ns();
        eth_ptp::txq_arm_two_step(kTxq, tag);
        if (!send(frame) || !wait_for([] { return !eth_ptp::tx_stamp_fifo::empty(); })) {
            break;
        }
        const std::optional<eth_ptp::tx_stamp_fifo::Entry> entry = eth_ptp::tx_stamp_fifo::pop();
        const uint64_t after_ns = eth_ptp::read_ptp_ns();
        matched += entry && entry->tag == tag && entry->time_ns >= before_ns && entry->time_ns <= after_ns;
    }
    eth_ptp::txq_disarm(kTxq);
    return matched;
}

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr Result* result = reinterpret_cast<volatile tt_l1_ptr Result*>(base + kResultOffset);
    volatile tt_l1_ptr Frame* frame = reinterpret_cast<volatile tt_l1_ptr Frame*>(base + kFrameOffset);
    frame->key = 0;
    frame->echo_key = 0;
    result->header_select_before = eth_ptp::word_of(eth_ptp::txq_header_select(kTxq).read());
    result->no_match_before = eth_ptp::word_of(eth_ptp::kRxNoMatchActions.read());
    eth_ptp::restart_ptp_timer();
    result->restart_error_ns = restart_error_ns();
    eth_ptp::RxStampRule<kTcamRow, kLabel> rule;
    rule.install(kStampMatch);
    eth_ptp::TxHeaderRow<kTxq, kHeaderRow> header;
    header.install(kStampFrameDestination);
    eth_ptp::txq_arm_in_frame(kTxq);

    handshake(base);

    uint32_t unstamped = 0, ingress_mismatched = 0, round = 0;
    for (; round < kRounds; round++) {
        const uint32_t key = round + 1;
        if constexpr (kTransmitter) {
            frame->key = key;
            frame->echo_key = 0;
            if (!send(frame) || !wait_for([&] { return frame->echo_key == key; })) {
                break;
            }
        } else {
            if (!wait_for([&] { return frame->key == key; })) {
                break;
            }
        }
        const uint64_t peer_egress = eth_ptp::frame_stamp_ns(frame->stamp);
        const std::optional<uint64_t> ingress = take_ingress();
        ingress_mismatched += !ingress;
        unstamped += peer_egress == 0;
        result->stamps[round].peer_egress = peer_egress;
        result->stamps[round].ingress = ingress.value_or(0);
        if constexpr (!kTransmitter) {
            frame->echo_key = key;
            if (!send(frame)) {
                break;
            }
        }
    }

    // The MAC stamps a frame's egress time only while the queue is armed, and the receiver's last echo can still be
    // queued when its loop ends, so both ports wait for each other before disarming.
    handshake(base);
    eth_ptp::txq_disarm(kTxq);
    result->two_step_matched = kTransmitter && round == kRounds ? send_two_step(frame) : 0;
    header.uninstall();
    rule.uninstall();
    result->header_select_after = eth_ptp::word_of(eth_ptp::txq_header_select(kTxq).read());
    result->no_match_after = eth_ptp::word_of(eth_ptp::kRxNoMatchActions.read());
    result->unstamped = unstamped;
    result->ingress_mismatched = ingress_mismatched;
    result->rounds = round;
    std::atomic_thread_fence(std::memory_order_release);
    result->done = kDone;
}
