// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The receiver credit transport for a bridged link. Every other transport ends with a wire push;
// a bridged upstream has no cable, so the totals stop in L1 and the host relays them.
#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"

// Included from fabric_router_flow_control.hpp, which the kernel pulls in well before any other
// bridge header. Nothing heavier than the CT args may be reached from here.

// Laid out as two uint32 arrays, completions then acks, so the host reads one contiguous block.
// MAX_NUM_SENDER_CHANNELS, not NUM: src_id arrives on the wire and is indexed unchecked.
constexpr uint32_t kE2hRecvCreditWords = MAX_NUM_SENDER_CHANNELS;
constexpr uint32_t kE2hRecvCreditBytes = 2 * kE2hRecvCreditWords * sizeof(uint32_t);

struct ReceiverChannelHostBridgeCreditSender {
    ReceiverChannelHostBridgeCreditSender() = default;

    // Takes the channel index for the same reason the counter form does -- construct_credit_sender_in_place
    // selects on it -- though the block address is a CT arg and does not vary per channel.
    explicit ReceiverChannelHostBridgeCreditSender(size_t /*receiver_channel_index*/) :
        completion_counters_base_ptr(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e2h_bridge_recv_credits_addr)),
        ack_counters_base_ptr(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e2h_bridge_recv_credits_addr) + kE2hRecvCreditWords),
        completion_counters({}),
        ack_counters({}) {
        // Zeroed here rather than by the host: L1 persists across processes, so a stale total from
        // a previous run would read as credit the far sender never earned.
        for (size_t i = 0; i < kE2hRecvCreditWords; i++) {
            completion_counters[i] = 0;
            ack_counters[i] = 0;
            completion_counters_base_ptr[i] = 0;
            ack_counters_base_ptr[i] = 0;
        }
    }

    // RUNNING TOTALS, never deltas: the host samples whenever it polls, so a delta it happened to
    // miss would be lost. The reader takes an unsigned difference against what it last relayed.
    FORCE_INLINE void send_completion_credit(uint8_t src_id, uint32_t num_completions) {
        completion_counters[src_id] += num_completions;
        completion_counters_base_ptr[src_id] = completion_counters[src_id];
    }

    FORCE_INLINE void send_ack_credit(uint8_t src_id) {
        ack_counters[src_id]++;
        ack_counters_base_ptr[src_id] = ack_counters[src_id];
    }

    volatile tt_l1_ptr uint32_t* completion_counters_base_ptr;
    volatile tt_l1_ptr uint32_t* ack_counters_base_ptr;
    // Local copy so the increment costs no L1 load, as the counter-based sender does.
    std::array<uint32_t, kE2hRecvCreditWords> completion_counters;
    std::array<uint32_t, kE2hRecvCreditWords> ack_counters;
};
