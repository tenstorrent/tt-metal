// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ordering sender, on a Keraunos CCE (IDLE_ETH, DM0). For iterations 1..N: wait for the receiver's ack of the previous
// iteration, write iteration i's pattern into the payload buffer, send it with eth_send_packet, then send credit = i
// with an inline remote update, both on one TX queue (CM-087). The receiver checks the payload when it sees the
// credit.
//
// Compile-time define (the host passes it):
//   KERAUNOS_CREDIT_SEND_DWORD  1: the credit is cce_eth::send_dword() (compat's inline update) called directly;
//                               unset: internal_::eth_write_remote_reg(), which wraps the same 8-byte write.
//
// Runtime args: 0 base, 1 iterations, 2 payload bytes (multiple of 16), 3 TX queue, 4 poll bound per wait. The
// Ethernet-link slot must hold the peer's HSIO tile and the valid word before launch.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/tunneling.h"
#include "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_eth_ordering.h"

using namespace keraunos_eth_ordering;
namespace cce_eth = internal_::cce_eth;

namespace {

// Polls the ack word until it reads @p want, at most @p polls times. False on a timeout.
bool wait_ack(const volatile uint32_t* ack, uint32_t want, uint32_t polls) {
    for (uint32_t p = 0; *ack != want && p < polls; p++) {
    }
    return *ack == want;
}

void send_credit(uint32_t q, uint32_t credit_addr, uint32_t value) {
#if defined(KERAUNOS_CREDIT_SEND_DWORD)
    while (internal_::eth_txq_is_busy(q)) {
    }
    cce_eth::send_dword(q, credit_addr, value);
#else
    internal_::eth_write_remote_reg(q, credit_addr, value);
#endif
}

}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t iterations = get_arg_val<uint32_t>(1);
    const uint32_t payload_bytes = get_arg_val<uint32_t>(2);
    const uint32_t q = get_arg_val<uint32_t>(3);
    const uint32_t polls = get_arg_val<uint32_t>(4);

    auto* result = reinterpret_cast<volatile SenderResult*>(uncached(base + kResultOffset));
    result->status = kStatusRunning;
    if (payload_bytes == 0 || payload_bytes % 16 != 0 || payload_bytes > kMaxPayloadBytes ||
        q >= cce_eth::kNumTxqs || iterations == 0) {
        result->status = kStatusBadArgs;
        return;
    }

    volatile uint32_t* ack = uncached(base + kAckOffset);
    volatile uint32_t* payload = uncached(base + kPayloadOffset);
    const uint32_t payload_words = payload_bytes / sizeof(uint32_t);

    for (uint32_t i = 1; i <= iterations; i++) {
        if (!wait_ack(ack, i - 1, polls)) {
            result->ack_timeout_iteration = i;
            result->status = kStatusTimeout;
            return;
        }

        for (uint32_t j = 0; j < payload_words; j++) {
            payload[j] = payload_word(i, j);
        }
        internal_::eth_send_packet(
            q,
            (base + kPayloadOffset) >> BYTES_TO_ETH_WORD_SHIFT,
            (base + kPayloadOffset) >> BYTES_TO_ETH_WORD_SHIFT,
            payload_bytes >> BYTES_TO_ETH_WORD_SHIFT);
        send_credit(q, base + kCreditOffset, i);
        result->iterations_sent = i;
    }

    if (!wait_ack(ack, iterations, polls)) {
        result->ack_timeout_iteration = iterations + 1;
        result->status = kStatusTimeout;
        return;
    }
    result->idma_outstanding_at_end = drain_sends(polls);
    asm volatile("fence" ::: "memory");
    result->status = kStatusDone;
}
