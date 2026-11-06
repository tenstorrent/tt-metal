// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ordering receiver, on the Keraunos CCE at the far end of the link (IDLE_ETH, DM0). Polls the credit word the sender
// updates over Ethernet; for each new value v, checks the payload is iteration v's pattern before acking v with a
// CPU store into the sender's SRAM. A mismatch means the credit was visible before the payload: the kernel records
// it, then polls the payload again to tell a late payload from a wrong one, and acks either way.
//
// Runtime args: 0 base, 1 iterations, 2 payload bytes, 3 poll bound per wait, 4-5 low/high word of the global HSIO
// SRAM address of the sender's ack word (0x1206800000 + hsio * tile_stride + base + kAckOffset; remap entries 6-9
// deliver it, as in eth_write_tile_sram.cpp).
//
// Ethernet RX writes are not snooped (no cmd_snoop on the send), so every read of them goes through the uncached
// alias (KERAUNOS-HW-FINDINGS Q5).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_eth_ordering.h"

using namespace keraunos_eth_ordering;

namespace {

// Index of the first word of @p payload that is not iteration @p iteration's pattern, or @p words if all match.
uint32_t first_bad_word(const volatile uint32_t* payload, uint32_t words, uint32_t iteration) {
    for (uint32_t j = 0; j < words; j++) {
        if (payload[j] != payload_word(iteration, j)) {
            return j;
        }
    }
    return words;
}

}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t iterations = get_arg_val<uint32_t>(1);
    const uint32_t payload_bytes = get_arg_val<uint32_t>(2);
    const uint32_t polls = get_arg_val<uint32_t>(3);
    const uint64_t ack_global = (static_cast<uint64_t>(get_arg_val<uint32_t>(5)) << 32) | get_arg_val<uint32_t>(4);

    auto* result = reinterpret_cast<volatile ReceiverResult*>(uncached(base + kResultOffset));
    result->status = kStatusRunning;
    if (payload_bytes == 0 || payload_bytes % 16 != 0 || payload_bytes > kMaxPayloadBytes || iterations == 0 ||
        static_cast<uint64_t>(static_cast<uintptr_t>(ack_global)) != ack_global) {
        result->status = kStatusBadArgs;
        return;
    }

    volatile uint32_t* credit = uncached(base + kCreditOffset);
    const volatile uint32_t* payload = uncached(base + kPayloadOffset);
    volatile uint32_t* ack = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(ack_global));
    const uint32_t words = payload_bytes / sizeof(uint32_t);

    uint32_t last = 0;
    while (last < iterations) {
        uint32_t v = *credit;
        for (uint32_t p = 0; v == last && p < polls; p++) {
            v = *credit;
        }
        if (v == last) {
            result->timeout_waiting_for = last + 1;
            result->status = kStatusTimeout;
            return;
        }
        if (v != last + 1) {
            if (result->credit_skips == 0) {
                result->first_skip_from = last;
                result->first_skip_to = v;
            }
            result->credit_skips = result->credit_skips + 1;
        }

        const uint32_t bad = first_bad_word(payload, words, v);
        if (bad != words) {
            if (result->mismatch_count == 0) {
                result->first_mismatch_iteration = v;
                result->first_mismatch_word = bad;
                result->first_mismatch_got = payload[bad];
                result->first_mismatch_want = payload_word(v, bad);
            }
            result->mismatch_count = result->mismatch_count + 1;
            for (uint32_t p = 0; p < polls; p++) {
                if (first_bad_word(payload, words, v) == words) {
                    result->mismatch_resolved_count = result->mismatch_resolved_count + 1;
                    break;
                }
            }
        }

        result->iterations_seen = result->iterations_seen + 1;
        result->last_credit = v;
        *ack = v;
        // The ack window is outside this core's cacheable SRAM; the fence orders the store before the next poll.
        asm volatile("fence" ::: "memory");
        last = v;
    }
    asm volatile("fence" ::: "memory");
    result->status = kStatusDone;
}
