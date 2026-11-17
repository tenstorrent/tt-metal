// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ethernet bandwidth sender, on CCE0 of one end of a Keraunos Ethernet loopback (IDLE_ETH, DM0). Streams N sends of
// SIZE bytes to the link peer (eth_bw_receiver.cpp), spread round-robin over Q TX queues, each from the data region to
// the same offset on the peer (eth_cce_send.h: an iDMA copy on command buffer 1 per send, as the fabric router's
// eth_send_packet does).
//
// Sequence:
//   1. W warm-up sends, then drain (every send acked on the send buffer's transaction id).
//   2. Handshake: flag = 1 to the receiver (inline write, TX queue 0); wait for its ack = 1. Measures an idle round
//      trip (cycles kHandshakeRtt) and proves the receiver is running.
//   3. Timed: t0; N sends; t_inj; drain; t_acked; flag = 2; wait for ack = 2; t_delivered.
//        INJECTION = t_inj - t0      issue rate, limited by command-buffer programming and TX queue space
//        ACKED     = t_acked - t0    every send acked by this core's NIU
//        DELIVERED = t_delivered - t0  the receiver has seen the closing flag, which left after every send was acked,
//                    and its ack (a CPU store back across tiles) arrived: includes one return path, so it is a bound,
//                    not a one-way time. The host reports kHandshakeRtt next to it.
//
// Runtime args: 0 base, 1 size, 2 sends N, 3 warm-up W, 4 TX queues Q (1..4), 5 polls per wait.
// The Ethernet-link slot must hold the peer's HSIO tile and the valid word before launch.
//
// Result (hart 0 slot): cycles[kInjection/kAcked/kDelivered/kHandshakeRtt], aux[kTxqBusySpins].

#include <cstdint>

#include "internal/ethernet/tunneling.h"  // eth_cce_send.h, which needs tunneling.h's postcode pointers
#include "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/keraunos_ubench.h"

using namespace keraunos_ubench;
namespace eb = keraunos_ubench::eth_bw;
namespace cce_eth = internal_::cce_eth;

namespace {

constexpr uint32_t kHart = 0;

bool drain_sends(uint32_t polls) {
    for (uint32_t p = 0; p < polls; p++) {
        if (__builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(cce_eth::kCmdBuf, cce_eth::kTrid) == 0) {
            return true;
        }
    }
    return __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(cce_eth::kCmdBuf, cce_eth::kTrid) == 0;
}

void send_flag(uint32_t value) {
    while (cce_eth::txq_busy(0)) {
    }
    cce_eth::send_dword(0, kEthFlagOffset, value);
}

}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t size = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const uint32_t warmup = get_arg_val<uint32_t>(3);
    const uint32_t txqs = get_arg_val<uint32_t>(4);
    const uint32_t polls = get_arg_val<uint32_t>(5);

    volatile Result* r = result_block(base, kHart);
    r->kernel = static_cast<uint32_t>(KernelId::EthBwSender);
    r->hart = kHart;
    r->status = kStatusRunning;

    uint32_t bad = 0;
    if (size == 0 || size % 16 != 0 || size > kMaxTransferBytes) {
        bad = 1;
    } else if (n == 0) {
        bad = 2;
    } else if (txqs == 0 || txqs > cce_eth::kNumTxqs) {
        bad = 3;
    }
    if (bad != 0) {
        r->detail = bad;
        finish(r, kStatusBadArgs);
        return;
    }
    cce_eth::check_on_cce0();

    const uint32_t data = base + src_offset(kHart);
    const volatile uint32_t* ack = uncached(base + kEthAckOffset);
    uint64_t busy_spins = 0;

    auto send = [&](uint32_t i) {
        const uint32_t q = i % txqs;
        while (cce_eth::txq_busy(q)) {
            busy_spins++;
        }
        cce_eth::send_bytes(q, data, data, size);
    };

    // 1. Warm-up.
    for (uint32_t i = 0; i < warmup; i++) {
        send(i);
    }
    if (!drain_sends(polls)) {
        r->detail = 1;  // warm-up drain
        finish(r, kStatusTimeout);
        return;
    }

    // 2. Handshake. The receiver can start long after this core, so its first wait gets the full poll bound.
    const uint64_t h0 = now();
    send_flag(1);
    if (!wait_eq<uint32_t>(ack, 1, polls)) {
        r->detail = 2;  // handshake ack
        finish(r, kStatusTimeout);
        return;
    }
    r->cycles[eb::kHandshakeRtt] = now() - h0;

    // 3. Timed.
    busy_spins = 0;
    const uint64_t t0 = now();
    for (uint32_t i = 0; i < n; i++) {
        send(i);
    }
    const uint64_t t_inj = now();
    if (!drain_sends(polls)) {
        r->detail = 3;  // timed drain
        finish(r, kStatusTimeout);
        return;
    }
    const uint64_t t_acked = now();
    send_flag(2);
    if (!wait_eq<uint32_t>(ack, 2, polls)) {
        r->detail = 4;  // closing ack
        finish(r, kStatusTimeout);
        return;
    }
    const uint64_t t_delivered = now();

    r->cycles[eb::kInjection] = t_inj - t0;
    r->cycles[eb::kAcked] = t_acked - t0;
    r->cycles[eb::kDelivered] = t_delivered - t0;
    r->aux[eb::kTxqBusySpins] = busy_spins;
    r->iterations_done = n;
    finish(r, kStatusDone);
}
