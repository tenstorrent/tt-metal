// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// fabric_counter_based_credits.hpp
// =============================================================================
//
// Outlined L1-counter-based credit widgets, extracted from
// `fabric_router_flow_control.hpp` so callers (the CRAQ-Fabric generated
// kernel in particular) can reuse the credit-region transport WITHOUT
// transitively pulling in `fabric_erisc_router_ct_args.hpp`.
//
// All credit-region L1 addresses are now CONSTRUCTOR ARGUMENTS instead of
// being read from upstream's CT-arg-derived constants. Each caller wires
// these from its own CT-arg layer.
//
// `fabric_router_flow_control.hpp` is now a thin compatibility shim that
// includes this header AND `fabric_erisc_router_ct_args.hpp`, then
// re-exposes the legacy widget names bound to the upstream CT-arg constants.
// =============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

#include "internal/ethernet/tt_eth_api.h"
#include "internal/ethernet/tunneling.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"

// -----------------------------------------------------------------------------
// ReceiverChannelCounterBasedResponseCreditSenderOutlined<NumSenderChannels>
// -----------------------------------------------------------------------------
//
// L1-counter-based credit sender (receiver-side). Bumps per-source-channel
// counters in a local L1 region, then bulk-sends the whole region to the peer
// via `eth_send_packet_bytes_unsafe` over a caller-specified TXQ. The peer
// reads its mirror image to learn how many ack/completion credits to apply.
//
// Differences vs upstream `ReceiverChannelCounterBasedResponseCreditSender`:
//   - All four addresses + the bulk-send byte count + the TXQ id come from
//     CONSTRUCTOR ARGUMENTS, not from `fabric_erisc_router_ct_args.hpp`.
//   - `NumSenderChannels` is a TEMPLATE parameter so callers don't depend on
//     upstream's `NUM_SENDER_CHANNELS` constexpr.
template <std::size_t NumSenderChannels>
struct ReceiverChannelCounterBasedResponseCreditSenderOutlined {
    ReceiverChannelCounterBasedResponseCreditSenderOutlined() = default;

    ReceiverChannelCounterBasedResponseCreditSenderOutlined(
        std::size_t /*receiver_channel_index*/,
        std::size_t local_receiver_completion_counters_base_address,
        std::size_t local_receiver_ack_counters_base_address,
        std::size_t local_receiver_credits_base_address_in,
        std::size_t to_senders_credits_base_address_in,
        std::size_t total_number_of_receiver_to_sender_credit_num_bytes_in,
        uint8_t receiver_txq_id_in) :
        completion_counters_base_ptr(
            reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counters_base_address)),
        ack_counters_base_ptr(
            reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counters_base_address)),
        local_receiver_credits_base_address(local_receiver_credits_base_address_in),
        to_senders_credits_base_address(to_senders_credits_base_address_in),
        total_number_of_receiver_to_sender_credit_num_bytes(
            total_number_of_receiver_to_sender_credit_num_bytes_in),
        receiver_txq_id(receiver_txq_id_in),
        completion_counters({}),
        ack_counters({}) {
        for (std::size_t i = 0; i < NumSenderChannels; i++) {
            completion_counters[i] = 0;
            ack_counters[i] = 0;
        }
    }

    FORCE_INLINE void send_completion_credit(uint8_t src_id, uint32_t num_completions) {
        completion_counters[src_id] += num_completions;
        completion_counters_base_ptr[src_id] = completion_counters[src_id];
        update_sender_side_credits();
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id) {
        ack_counters[src_id]++;
        ack_counters_base_ptr[src_id] = ack_counters[src_id];
        update_sender_side_credits();
    }

    // Default member initializers ensure the `= default` constructor produces a
    // safely-zeroed object. Prior to this, `-Werror=uninitialized` flagged the
    // upstream wrapper's default-constructed instances under LTO.
    volatile tt_l1_ptr uint32_t* completion_counters_base_ptr{nullptr};
    volatile tt_l1_ptr uint32_t* ack_counters_base_ptr{nullptr};
    std::size_t local_receiver_credits_base_address{0};
    std::size_t to_senders_credits_base_address{0};
    std::size_t total_number_of_receiver_to_sender_credit_num_bytes{0};
    uint8_t receiver_txq_id{0};
    // Local memory copy to save an L1 load
    std::array<uint32_t, NumSenderChannels> completion_counters{};
    std::array<uint32_t, NumSenderChannels> ack_counters{};

private:
    FORCE_INLINE void update_sender_side_credits() const {
        internal_::eth_send_packet_bytes_unsafe(
            receiver_txq_id,
            local_receiver_credits_base_address,
            to_senders_credits_base_address,
            total_number_of_receiver_to_sender_credit_num_bytes);
    }
};

// -----------------------------------------------------------------------------
// SenderChannelFromReceiverCounterBasedCreditsReceiverOutlined
// -----------------------------------------------------------------------------
//
// L1-counter-based credit receiver (sender-side). Reads per-channel cumulative
// counter values from L1 mirror addresses written by the peer receiver via
// ETH bulk-send.
//
// Differences vs upstream `SenderChannelFromReceiverCounterBasedCreditsReceiver`:
//   - Counter region base addresses come from CONSTRUCTOR ARGUMENTS, not from
//     `fabric_erisc_router_ct_args.hpp`.
struct SenderChannelFromReceiverCounterBasedCreditsReceiverOutlined {
    SenderChannelFromReceiverCounterBasedCreditsReceiverOutlined() = default;

    SenderChannelFromReceiverCounterBasedCreditsReceiverOutlined(
        std::size_t sender_channel_index,
        std::size_t to_sender_remote_ack_counters_base_address,
        std::size_t to_sender_remote_completion_counters_base_address) :
        acks_received_counter_ptr(
            reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address) +
            sender_channel_index),
        completions_received_counter_ptr(
            reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counters_base_address) +
            sender_channel_index),
        acks_received_and_processed(0),
        completions_received_and_processed(0) {}

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        return *acks_received_counter_ptr - acks_received_and_processed;
    }

    FORCE_INLINE void increment_num_processed_acks(std::size_t num_acks) {
        acks_received_and_processed += num_acks;
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        return *completions_received_counter_ptr - completions_received_and_processed;
    }

    FORCE_INLINE void increment_num_processed_completions(std::size_t num_completions) {
        completions_received_and_processed += num_completions;
    }

    volatile uint32_t* acks_received_counter_ptr;
    volatile uint32_t* completions_received_counter_ptr;
    uint32_t acks_received_and_processed = 0;
    uint32_t completions_received_and_processed = 0;
};
