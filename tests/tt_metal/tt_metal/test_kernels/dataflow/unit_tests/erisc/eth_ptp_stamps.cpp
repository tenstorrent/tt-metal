// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One end of the 1588 stamping test, stamping the way the profiler's link sync does: starts the PTP timer, installs
// the stamp rule and header row and arms its queue to write each frame's send time into the frame, handshakes with the
// peer, then each round the initiator sends a frame and the peer echoes it, each end keeping the egress stamp the
// other's frame carried and that frame's ingress stamp; the rule and row are removed and the registers they borrowed
// are read back. Compile args: initiator (1) or echo (0).
// Runtime arg: the unreserved L1 base (eth_ptp_stamps.hpp's layout).

#include <cstdint>

#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "eth_ptp_stamps.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;
using namespace eth_ptp_stamps;

constexpr bool kInitiator = get_compile_time_arg_val(0) != 0;
constexpr uint32_t kTxq = 2, kLabel = 0x15;
constexpr uint32_t kSyncWord = 4;
static_assert(eth_ptp::kFrameStampField + 10 <= 4 * kSyncWord);
static_assert(4 * kSyncWord + sizeof(eth_channel_sync_t) <= kFrameBytes);
constexpr eth_ptp::TxQueue<kTxq> g_txq{};
static eth_ptp::PtpTimer g_timer;
static eth_ptp::TxHeaderRow<kTxq, 3> g_header;
static eth_ptp::RxStampRule<63, kLabel> g_rule;
static constexpr uint32_t kHandshake = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
static constexpr uint32_t kSpins = 1u << 20;

// The frame's slot still holds the stamp of the last frame that came in through it, so it is cleared first and a
// frame the MAC did not stamp arrives as zero.
inline void send(volatile tt_l1_ptr uint32_t* w) {
    w[eth_ptp::kFrameStampHiWord] = 0;
    w[eth_ptp::kFrameStampHiWord + 1] = 0;
    const uint32_t addr = reinterpret_cast<uint32_t>(w);
    internal_::eth_send_packet<false>(kTxq, addr >> 4, addr >> 4, kFrameBytes >> 4);
    while (internal_::eth_txq_is_busy(kTxq)) {
    }
}

bool take_ingress(uint64_t& ts, uint32_t& extra) {
    uint32_t n = 0;
    eth_ptp::RxStampFifo{}.drain<kLabel>([&](uint64_t t) {
        if (n++ == 0) {
            ts = t;
        }
    });
    extra += n > 1 ? n - 1 : 0;
    return n != 0;
}

template <typename Pred>
bool wait_for(Pred&& pred) {
    for (uint32_t s = 0; s < kSpins; s++) {
        invalidate_l1_cache();
        if (pred()) {
            return true;
        }
    }
    return false;
}

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr Result* res = reinterpret_cast<volatile tt_l1_ptr Result*>(base + kResultOffset);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + kFrameOffset);
    volatile tt_l1_ptr eth_channel_sync_t* frame =
        reinterpret_cast<volatile tt_l1_ptr eth_channel_sync_t*>(w + kSyncWord);
    frame->bytes_sent = 0;
    frame->receiver_ack = 0;
    res->done = 0;
    res->sel_before = eth_ptp::bits(eth_ptp::txq_pkt_cfg_sel_sw(kTxq).read());
    res->no_match_before = eth_ptp::bits(eth_ptp::kRxNoMatchActions.read());
    res->timer_ok = g_timer.start();
    g_rule.install();
    g_header.install();
    g_txq.arm_in_frame();

    if constexpr (kInitiator) {
        eth_send_bytes(kHandshake, kHandshake, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }

    uint32_t unstamped = 0, rx_missing = 0, rx_extra = 0, i = 0;
    for (; i < kRounds; i++) {
        const uint32_t key = 0x5A000000u | (i + 1);
        uint64_t ingress = 0;
        if constexpr (kInitiator) {
            frame->bytes_sent = key;
            frame->receiver_ack = 0;
            send(w);
            if (!wait_for([&] { return frame->receiver_ack == key; })) {
                break;
            }
        } else {
            if (!wait_for([&] { return frame->bytes_sent == key; })) {
                break;
            }
        }
        const uint64_t egress = eth_ptp::frame_stamp(w);
        rx_missing += !take_ingress(ingress, rx_extra);
        unstamped += egress == 0;
        res->stamps[i][0] = egress;
        res->stamps[i][1] = ingress;
        if constexpr (!kInitiator) {
            frame->receiver_ack = key;
            send(w);
        }
    }

    g_txq.disarm();
    g_header.restore();
    g_rule.remove();
    res->sel_after = eth_ptp::bits(eth_ptp::txq_pkt_cfg_sel_sw(kTxq).read());
    res->no_match_after = eth_ptp::bits(eth_ptp::kRxNoMatchActions.read());
    res->unstamped = unstamped;
    res->rx_missing = rx_missing;
    res->rx_extra = rx_extra;
    res->rounds = i;
    res->done = kDone;
}
