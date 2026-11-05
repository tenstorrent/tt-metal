// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SRAM layout and result blocks shared by the Keraunos Ethernet ordering kernels and their host test
// (test_keraunos_eth_send.cpp). Every offset is relative to a base the host passes as runtime arg 0: a 64-byte-aligned
// flat offset in CCE SRAM at or above MEM_IERISC_ROUTER_UNRESERVED_BASE, so the Ethernet-link slot below it stays
// intact. Sender and receiver use the same base, because a send lands at the same offset in the peer's SRAM.

#pragma once

#include <cstdint>

#if defined(KERNEL_BUILD)
#include "dev_mem_map.h"
#include "internal/tt-2xx/quasar/eth_cce_send.h"
#endif

namespace keraunos_eth_ordering {

// The block a kernel writes its outcome into (sender's own result on the sender, receiver's on the receiver).
constexpr uint32_t kResultOffset = 0x000;
constexpr uint32_t kResultSize = 0x100;
// Receiver -> sender: the last iteration the receiver checked. A CPU store into the sender's SRAM.
constexpr uint32_t kAckOffset = 0x100;
// Sender -> receiver: the credit, one inline remote update per iteration. Its own 16-byte slot so an 8-byte inline
// write cannot touch a neighbour.
constexpr uint32_t kCreditOffset = 0x110;
// Payload: the sender's source buffer and the receiver's destination, at the same offset.
constexpr uint32_t kPayloadOffset = 0x1000;
constexpr uint32_t kMaxPayloadBytes = 0x10000;

constexpr uint32_t kCanary = 0xCAFEF00Du;

// Status word values.
constexpr uint32_t kStatusRunning = 0x5254'0001u;
constexpr uint32_t kStatusDone = 0x444F'4E45u;
constexpr uint32_t kStatusTimeout = 0x544D'4F55u;
constexpr uint32_t kStatusBadArgs = 0xBAD0'A465u;

// Word j of iteration i's payload. Distinct per iteration and per word.
constexpr uint32_t payload_word(uint32_t iteration, uint32_t j) { return (iteration * 0x9E3779B1u) ^ (j * 0x01000193u); }

struct SenderResult {
    uint32_t status;
    uint32_t iterations_sent;
    // Iteration whose ack wait timed out, 0 if none.
    uint32_t ack_timeout_iteration;
    // Transfers still unacked on the send buffer's transaction id when the kernel returned.
    uint32_t idma_outstanding_at_end;
};
static_assert(sizeof(SenderResult) <= kResultSize);

struct ReceiverResult {
    uint32_t status;
    // Credits observed and checked.
    uint32_t iterations_seen;
    uint32_t last_credit;
    // Iterations whose payload did not match when its credit was seen.
    uint32_t mismatch_count;
    uint32_t first_mismatch_iteration;
    uint32_t first_mismatch_word;
    uint32_t first_mismatch_got;
    uint32_t first_mismatch_want;
    // Mismatched iterations whose payload matched on a later poll: the credit overtook the payload.
    uint32_t mismatch_resolved_count;
    // Credits that were not last + 1 (a skipped or out-of-order update).
    uint32_t credit_skips;
    uint32_t first_skip_from;
    uint32_t first_skip_to;
    // Iteration the receiver was waiting for when it timed out, 0 if none.
    uint32_t timeout_waiting_for;
};
static_assert(sizeof(ReceiverResult) <= kResultSize);

#if defined(KERNEL_BUILD)
// This CCE's SRAM at flat offset @p flat, through the uncached alias.
inline volatile uint32_t* uncached(uint32_t flat) {
    return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(flat + MEM_IERISC_UNCACHED_BASE));
}

// Polls the Ethernet send buffer's transaction id until every send is acked, at most @p polls times; returns the
// count still outstanding.
inline uint32_t drain_sends(uint32_t polls) {
    uint64_t outstanding = 0;
    for (uint32_t p = 0; p < polls; p++) {
        outstanding = __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(
            internal_::cce_eth::kCmdBuf, internal_::cce_eth::kTrid);
        if (outstanding == 0) {
            break;
        }
    }
    return static_cast<uint32_t>(outstanding);
}
#endif

}  // namespace keraunos_eth_ordering
