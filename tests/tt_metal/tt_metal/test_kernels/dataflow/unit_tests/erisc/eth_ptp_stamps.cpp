// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One end of the 1588 stamping test: opens a StampSession, handshakes with the peer, then each round the initiator
// sends a frame and the peer echoes it, each end keeping its frame's egress stamp and the other's frame's ingress
// stamp; the session is closed and the registers it borrows are read back. Compile args: initiator (1) or echo (0).
// Runtime arg: the unreserved L1 base (eth_ptp_stamps.hpp's layout).

#include <cstdint>

#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "eth_ptp_stamps.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;
namespace raw = eth_ptp::raw;
using namespace eth_ptp_stamps;

constexpr bool kInitiator = get_compile_time_arg_val(0) != 0;
using Session = eth_ptp::StampSession<2, 3, 63, 0x15>;
static Session g_sess;
static constexpr uint32_t kHandshake = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
static constexpr uint32_t kSpins = 1u << 20;

inline void send(uint32_t addr) {
    internal_::eth_send_packet<false>(Session::kTxq, addr >> 4, addr >> 4, kFrameBytes >> 4);
    while (internal_::eth_txq_is_busy(Session::kTxq)) {
    }
}

// The frame at `frame` sent under `tag`; its egress stamp, false when the MAC gave none. A pilot goes first under the
// queue's boot header row and unarmed, so a keepalive the queue had waiting leaves before the request is armed and
// cannot take the frame's tag. The request is disarmed as soon as the frame's stamp is in: armed ~6 us past it, the
// queue's idle keepalive is stamped under the tag too.
bool send_stamped(uint32_t pilot, uint32_t frame, uint64_t tag, uint64_t& ts, uint32_t& extra) {
    const uint32_t units0 = raw::txq_word_cnt(Session::kTxq);
    eth_ptp::tx_header_row_select(g_sess, true);
    send(pilot);
    eth_ptp::tx_header_row_select(g_sess, false);
    for (uint32_t s = 0; raw::txq_word_cnt(Session::kTxq) - units0 < 2 && s < kSpins; s++) {
    }
    eth_ptp::stamps_arm(g_sess, tag);
    send(frame);
    uint32_t n = 0;
    const auto first = [&](uint64_t t) {
        if (n++ == 0) {
            ts = t;
        }
    };
    for (uint32_t s = 0; n == 0 && s < kSpins; s++) {
        eth_ptp::tx_stamps_drain(static_cast<uint32_t>(tag), first);
    }
    eth_ptp::stamps_disarm(g_sess);
    extra += n > 1 ? n - 1 : 0;
    return n != 0;
}

bool take_ingress(uint64_t& ts, uint32_t& extra) {
    uint32_t n = 0;
    eth_ptp::rx_stamps_drain(g_sess, [&](uint64_t t) {
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
    volatile tt_l1_ptr eth_channel_sync_t* frame =
        reinterpret_cast<volatile tt_l1_ptr eth_channel_sync_t*>(base + kFrameOffset);
    frame->bytes_sent = 0;
    frame->receiver_ack = 0;
    res->done = 0;
    res->sel_before = raw::rd(eth_ptp::txq_reg(Session::kTxq, eth_ptp::kTxqPktCfgSelSwOff));
    res->no_match_before = raw::rd(eth_ptp::kRxFlNoMatchActions);
    res->timer_ok = g_sess.begin();
    eth_ptp::raw::mac_tx_fifo_drain();
    res->ptp_offset_lo = static_cast<uint32_t>(g_sess.ptp_offset_64);
    res->ptp_offset_hi = static_cast<uint32_t>(static_cast<uint64_t>(g_sess.ptp_offset_64) >> 32);

    if constexpr (kInitiator) {
        eth_send_bytes(kHandshake, kHandshake, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }

    uint32_t tx_missing = 0, tx_extra = 0, rx_missing = 0, rx_extra = 0, i = 0;
    for (; i < kRounds; i++) {
        const uint32_t key = 0x5A000000u | (i + 1);
        uint64_t egress = 0, ingress = 0;
        if constexpr (kInitiator) {
            frame->bytes_sent = key;
            frame->receiver_ack = 0;
            tx_missing += !send_stamped(base + kPilotOffset, base + kFrameOffset, 0xA000'0000ull | i, egress, tx_extra);
            if (!wait_for([&] { return frame->receiver_ack == key; })) {
                break;
            }
            rx_missing += !take_ingress(ingress, rx_extra);
            res->stamps[i][0] = egress;
            res->stamps[i][1] = ingress;
        } else {
            if (!wait_for([&] { return frame->bytes_sent == key; })) {
                break;
            }
            rx_missing += !take_ingress(ingress, rx_extra);
            frame->receiver_ack = key;
            tx_missing += !send_stamped(base + kPilotOffset, base + kFrameOffset, 0xB000'0000ull | i, egress, tx_extra);
            res->stamps[i][0] = ingress;
            res->stamps[i][1] = egress;
        }
    }

    g_sess.end();
    res->sel_after = raw::rd(eth_ptp::txq_reg(Session::kTxq, eth_ptp::kTxqPktCfgSelSwOff));
    res->no_match_after = raw::rd(eth_ptp::kRxFlNoMatchActions);
    res->tx_missing = tx_missing;
    res->tx_extra = tx_extra;
    res->rx_missing = rx_missing;
    res->rx_extra = rx_extra;
    res->rounds = i;
    res->done = kDone;
}
