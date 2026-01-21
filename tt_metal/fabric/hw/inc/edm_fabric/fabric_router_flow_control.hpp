// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"

#include "internal/ethernet/tt_eth_api.h"
#include "internal/ethernet/tunneling.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/flow-control/credits.hpp"

struct ReceiverChannelCounterBasedResponseCreditSender {
    ReceiverChannelCounterBasedResponseCreditSender() = default;
    ReceiverChannelCounterBasedResponseCreditSender(size_t receiver_channel_index) :
        completion_counters_base_ptr(
            reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counters_base_address)),
        ack_counters_base_ptr(reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counters_base_address)),
        completion_counters({}),
        ack_counters({}) {
        for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
            completion_counters[i] = 0;
            ack_counters[i] = 0;
        }
    }

    FORCE_INLINE void send_completion_credit(uint8_t src_id) {
        completion_counters[src_id]++;
        completion_counters_base_ptr[src_id] = completion_counters[src_id];
        update_sender_side_credits();
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id, int count = 1) {
        ack_counters[src_id] += count;
        ack_counters_base_ptr[src_id] = ack_counters[src_id];
        update_sender_side_credits();
    }

    volatile tt_l1_ptr uint32_t* completion_counters_base_ptr;
    volatile tt_l1_ptr uint32_t* ack_counters_base_ptr;
    // Local memory copy to save an L1 load
    std::array<uint32_t, NUM_SENDER_CHANNELS> completion_counters;
    std::array<uint32_t, NUM_SENDER_CHANNELS> ack_counters;

private:
    FORCE_INLINE void update_sender_side_credits() const {
        internal_::eth_send_packet_bytes_unsafe(
            receiver_txq_id,
            local_receiver_credits_base_address,
            to_senders_credits_base_address,
            total_number_of_receiver_to_sender_credit_num_bytes);
    }
};

struct ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender {
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender() {
        for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
            if constexpr (ENABLE_FIRST_LEVEL_ACK) {
                sender_channel_packets_completed_stream_ids[i] = to_receiver_packets_sent_streams[0];
                // All sender channels pack into the first ack stream register (register 0)
                sender_channel_packets_ack_stream_ids[i] = to_sender_packets_acked_streams[0];
            } else {
                sender_channel_packets_completed_stream_ids[i] = to_sender_packets_completed_streams[i];
                sender_channel_packets_ack_stream_ids[i] = to_sender_packets_acked_streams[i];
            }
        }
    }

    FORCE_INLINE void send_completion_credit(uint8_t src_id) {
        WATCHER_RING_BUFFER_PUSH(0xFCC00000 | (src_id << 16) | sender_channel_packets_completed_stream_ids[src_id]);
        remote_update_ptr_val<receiver_txq_id>(sender_channel_packets_completed_stream_ids[src_id], 1);
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id, int count = 1) {
        WATCHER_RING_BUFFER_PUSH(0xFAA00000 | (src_id << 16) | sender_channel_packets_ack_stream_ids[src_id]);
        remote_update_ptr_val<receiver_txq_id>(sender_channel_packets_ack_stream_ids[src_id], count);
    }

    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_packets_completed_stream_ids;
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_packets_ack_stream_ids;
};

using ReceiverChannelResponseCreditSender = typename std::conditional_t<
    multi_txq_enabled,
    ReceiverChannelCounterBasedResponseCreditSender,
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender>;

// Packed credit flags: determine whether credits are packed into shared registers
// or sent individually per channel.
// - WH (!multi_txq_enabled): Uses packed stream registers when ENABLE_FIRST_LEVEL_ACK is true
// - BH (multi_txq_enabled): Uses counter-based mechanism (via function calls) regardless of ENABLE_FIRST_LEVEL_ACK
constexpr bool USE_PACKED_PACKET_SENT_CREDITS = ENABLE_FIRST_LEVEL_ACK && !multi_txq_enabled;
constexpr bool USE_PACKED_FIRST_LEVEL_ACK_CREDITS = ENABLE_FIRST_LEVEL_ACK && !multi_txq_enabled;
constexpr bool USE_PACKED_COMPLETION_ACK_CREDITS = ENABLE_FIRST_LEVEL_ACK && !multi_txq_enabled;

template <typename T = void>
struct init_receiver_channel_response_credit_senders_impl;

// Implementation for ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender
template <>
struct init_receiver_channel_response_credit_senders_impl<ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender> {
    template <uint8_t NUM_RECEIVER_CHANNELS>
    static constexpr auto init()
        -> std::array<ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender, NUM_RECEIVER_CHANNELS> {
        std::array<ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender, NUM_RECEIVER_CHANNELS> credit_senders;
        for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
            credit_senders[i] = ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender();
        }
        return credit_senders;
    }
};

// Implementation for ReceiverChannelCounterBasedResponseCreditSender
template <>
struct init_receiver_channel_response_credit_senders_impl<ReceiverChannelCounterBasedResponseCreditSender> {
    template <uint8_t NUM_RECEIVER_CHANNELS>
    static constexpr auto init() -> std::array<ReceiverChannelCounterBasedResponseCreditSender, NUM_RECEIVER_CHANNELS> {
        std::array<ReceiverChannelCounterBasedResponseCreditSender, NUM_RECEIVER_CHANNELS> credit_senders;
        for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
            credit_senders[i] = ReceiverChannelCounterBasedResponseCreditSender(i);
        }
        return credit_senders;
    }
};

template <uint8_t NUM_RECEIVER_CHANNELS>
constexpr FORCE_INLINE auto init_receiver_channel_response_credit_senders()
    -> std::array<ReceiverChannelResponseCreditSender, NUM_RECEIVER_CHANNELS> {
    return init_receiver_channel_response_credit_senders_impl<ReceiverChannelResponseCreditSender>::template init<
        NUM_RECEIVER_CHANNELS>();
}
struct SenderChannelFromReceiverCounterBasedCreditsReceiver {
    SenderChannelFromReceiverCounterBasedCreditsReceiver() = default;
    SenderChannelFromReceiverCounterBasedCreditsReceiver(size_t sender_channel_index) :
        acks_received_counter_ptr(
            reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address) + sender_channel_index),
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

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) { acks_received_and_processed += num_acks; }

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        return *completions_received_counter_ptr - completions_received_and_processed;
    }

    FORCE_INLINE void increment_num_processed_completions(size_t num_completions) {
        completions_received_and_processed += num_completions;
    }

    volatile uint32_t* acks_received_counter_ptr;
    volatile uint32_t* completions_received_counter_ptr;
    uint32_t acks_received_and_processed = 0;
    uint32_t completions_received_and_processed = 0;
};

struct SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver {
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver() = default;
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver(size_t sender_channel_index) :
        to_sender_packets_acked_stream(
            ENABLE_FIRST_LEVEL_ACK
                ? to_sender_packets_acked_streams[0]  // All channels use register 0 for packed credits
                : to_sender_packets_acked_streams[sender_channel_index]),
        to_sender_packets_completed_stream(
            ENABLE_FIRST_LEVEL_ACK ? to_receiver_packets_sent_streams[0]
                                   : to_sender_packets_completed_streams[sender_channel_index]) {}

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        return get_ptr_val(to_sender_packets_acked_stream);
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) {
        // WATCHER_RING_BUFFER_PUSH(0xAAA00000 | to_sender_packets_acked_stream);
        increment_local_update_ptr_val(to_sender_packets_acked_stream, -num_acks);
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        return get_ptr_val(to_sender_packets_completed_stream);
    }

    FORCE_INLINE void increment_num_processed_completions(size_t num_completions) {
        // WATCHER_RING_BUFFER_PUSH(0xBBB00000 | to_sender_packets_completed_stream);
        increment_local_update_ptr_val(to_sender_packets_completed_stream, -num_completions);
    }

    uint32_t to_sender_packets_acked_stream;
    uint32_t to_sender_packets_completed_stream;
};

template <typename T = void>
struct init_sender_channel_from_receiver_credits_flow_controllers_impl;

// Implementation for SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver

template <>
struct init_sender_channel_from_receiver_credits_flow_controllers_impl<
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver> {
    template <uint8_t NUM_SENDER_CHANNELS>
    static constexpr auto init()
        -> std::array<SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver, NUM_SENDER_CHANNELS> {
        std::array<SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver, NUM_SENDER_CHANNELS>
            flow_controllers;
        for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
            new (&flow_controllers[i]) SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver(i);
        }
        return flow_controllers;
    }
};

// Implementation for SenderChannelFromReceiverCounterBasedCreditsReceiver
template <>
struct init_sender_channel_from_receiver_credits_flow_controllers_impl<
    SenderChannelFromReceiverCounterBasedCreditsReceiver> {
    template <uint8_t NUM_SENDER_CHANNELS>
    static constexpr auto init()
        -> std::array<SenderChannelFromReceiverCounterBasedCreditsReceiver, NUM_SENDER_CHANNELS> {
        std::array<SenderChannelFromReceiverCounterBasedCreditsReceiver, NUM_SENDER_CHANNELS> flow_controllers;
        for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
            new (&flow_controllers[i]) SenderChannelFromReceiverCounterBasedCreditsReceiver(i);
        }
        return flow_controllers;
    }
};

using SenderChannelFromReceiverCredits = typename std::conditional_t<
    multi_txq_enabled,
    SenderChannelFromReceiverCounterBasedCreditsReceiver,
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver>;

// SFINAE-based overload for multi_txq_enabled case
template <uint8_t NUM_SENDER_CHANNELS>
constexpr FORCE_INLINE auto init_sender_channel_from_receiver_credits_flow_controllers()
    -> std::enable_if_t<!multi_txq_enabled, std::array<SenderChannelFromReceiverCredits, NUM_SENDER_CHANNELS>> {
    return init_sender_channel_from_receiver_credits_flow_controllers_impl<
        SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver>::template init<NUM_SENDER_CHANNELS>();
}

// SFINAE-based overload for !multi_txq_enabled case
template <uint8_t NUM_SENDER_CHANNELS>
constexpr FORCE_INLINE auto init_sender_channel_from_receiver_credits_flow_controllers()
    -> std::enable_if_t<multi_txq_enabled, std::array<SenderChannelFromReceiverCredits, NUM_SENDER_CHANNELS>> {
    return init_sender_channel_from_receiver_credits_flow_controllers_impl<
        SenderChannelFromReceiverCounterBasedCreditsReceiver>::template init<NUM_SENDER_CHANNELS>();
}

// MUST CHECK !is_eth_txq_busy() before calling
template <bool CHECK_BUSY>
FORCE_INLINE void receiver_send_completion_ack(
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender, uint8_t src_id) {
    if constexpr (CHECK_BUSY) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        };
    }
    receiver_channel_response_credit_sender.send_completion_credit(src_id);
}

template <bool CHECK_BUSY>
FORCE_INLINE void receiver_send_received_ack(
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender, uint8_t src_id, int count = 1) {
    if constexpr (CHECK_BUSY) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        };
    }
    receiver_channel_response_credit_sender.send_ack_credit(src_id, count);
}

/**
 * Adapter functions to abstract packing/unpacking logic for different architectures.
 * - WH with ENABLE_FIRST_LEVEL_ACK: Uses packed credits (multiple channels in one register)
 * - BH with multi_txq: Uses counter-based (one counter per channel, no packing)
 * - Non-ENABLE_FIRST_LEVEL_ACK: Uses individual stream registers (no packing)
 */

/**
 * Extract individual sender channel ACKs from packed value.
 * WH: Unpacks from shared register. BH: Pass-through (already unpacked).
 */
template <uint8_t sender_channel_index>
FORCE_INLINE uint32_t extract_sender_channel_acks(uint32_t packed_acks) {
    if constexpr (USE_PACKED_FIRST_LEVEL_ACK_CREDITS) {
        // WH: Stream register with packing - need to extract this channel's credits
        auto packed_acks_named =
            tt::tt_fabric::PackedCreditValue<NUM_SENDER_CHANNELS, tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS>{packed_acks};
        return tt::tt_fabric::PackedCredits<
            NUM_SENDER_CHANNELS,
            tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS,
            to_sender_packets_acked_streams[0]>::template get_value<sender_channel_index>(packed_acks_named);
    } else {
        // BH or !ENABLE_FIRST_LEVEL_ACK: Counter-based, already unpacked (one counter per channel)
        return packed_acks;
    }
}

/**
 * Build ACK decrement value for sender channel.
 * WH: Pack into register position. BH: Direct value (no packing).
 */
template <uint8_t sender_channel_index>
FORCE_INLINE uint32_t build_ack_decrement_value(uint32_t acks_count) {
    if constexpr (USE_PACKED_FIRST_LEVEL_ACK_CREDITS) {
        // WH: Pack into register position for this channel
        return tt::tt_fabric::PackedCredits<
                   NUM_SENDER_CHANNELS,
                   tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS,
                   to_sender_packets_acked_streams[0]>::template pack_value<sender_channel_index>(acks_count)
            .get();
    } else {
        // BH or !ENABLE_FIRST_LEVEL_ACK: Direct value, no packing
        return acks_count;
    }
}

/**
 * Build packet forward increment value for sender channel.
 * WH: Pack into register position. BH: Always 1 (no packing).
 */
template <uint8_t sender_channel_index, uint32_t to_receiver_pkts_sent_id>
FORCE_INLINE constexpr uint32_t build_packet_forward_value() {
    if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
        // WH: Pack 1 credit into this channel's position
        return tt::tt_fabric::PackedCredits<
                   NUM_SENDER_CHANNELS,
                   tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS,
                   to_receiver_pkts_sent_id>::template pack_value<sender_channel_index>(1)
            .get();
    } else {
        // BH or !ENABLE_FIRST_LEVEL_ACK: Always 1 (counter-based, no packing needed)
        return 1;
    }
}

/**
 * Accumulate receiver channel credits from packed value.
 * WH: Unpack and sum all channels. BH: Direct value (no packing).
 */
FORCE_INLINE uint32_t accumulate_receiver_channel_credits(uint32_t packed_value) {
    if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
        // WH: Unpack and sum across all sender channels
        using PC = tt::tt_fabric::PackedCredits<
            NUM_SENDER_CHANNELS,
            tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS,
            to_sender_packets_completed_streams[0]>;
        return PC::get_sum(typename PC::PackedCreditValueType(packed_value));
    } else {
        // BH or !ENABLE_FIRST_LEVEL_ACK: Already unpacked (direct count)
        return packed_value;
    }
}
