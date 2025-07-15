// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_constants.hpp"
#include "tt_metal/hw/inc/ethernet/tt_eth_api.h"
#include "tt_metal/hw/inc/ethernet/tunneling.h"

struct ReceiverChannelCounterBasedResponseCreditSender {
    ReceiverChannelCounterBasedResponseCreditSender() = default;
    ReceiverChannelCounterBasedResponseCreditSender(size_t receiver_channel_index) :
        completion_counter_ptr(
            reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counter_ptrs[receiver_channel_index])),
        ack_counter_ptr(reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counter_ptrs[receiver_channel_index])),
        completion_counter(0),
        ack_counter(0) {}

    FORCE_INLINE void send_completion_credit(uint8_t src_id) {
        completion_counter++;
        *completion_counter_ptr = completion_counter;
        internal_::eth_send_packet_bytes_unsafe(
            receiver_txq_id,
            reinterpret_cast<uint32_t>(this->completion_counter_ptr),
            to_sender_remote_completion_counter_addrs[src_id],
            ETH_WORD_SIZE_BYTES);
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id) {
        ack_counter++;
        *ack_counter_ptr = ack_counter;
        internal_::eth_send_packet_bytes_unsafe(
            receiver_txq_id,
            reinterpret_cast<uint32_t>(this->ack_counter_ptr),
            to_sender_remote_ack_counter_addrs[src_id],
            ETH_WORD_SIZE_BYTES);
    }

    volatile tt_l1_ptr uint32_t* completion_counter_ptr;
    volatile tt_l1_ptr uint32_t* ack_counter_ptr;
    // Local memory copy to save an L1 load
    uint32_t completion_counter;
    uint32_t ack_counter;
};

struct ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender {
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender() {}

    FORCE_INLINE void send_completion_credit(uint8_t src_id) {
        remote_update_ptr_val<receiver_txq_id>(to_sender_packets_completed_streams[src_id], 1);
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id) {
        remote_update_ptr_val<receiver_txq_id>(to_sender_packets_acked_streams[src_id], 1);
    }
};

using ReceiverChannelResponseCreditSender = typename std::conditional_t<
    multi_txq_enabled,
    ReceiverChannelCounterBasedResponseCreditSender,
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender>;

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
            reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counter_addrs[sender_channel_index])),
        completions_received_counter_ptr(
            reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counter_addrs[sender_channel_index])),
        acks_received_and_processed(0),
        completions_received_and_processed(0) {}

    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        invalidate_l1_cache();
        return *acks_received_counter_ptr - acks_received_and_processed;
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) { acks_received_and_processed += num_acks; }

    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        invalidate_l1_cache();
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
        to_sender_packets_acked_stream(to_sender_packets_acked_streams[sender_channel_index]),
        to_sender_packets_completed_stream(to_sender_packets_completed_streams[sender_channel_index]) {}

    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        return get_ptr_val(to_sender_packets_acked_stream);
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) {
        increment_local_update_ptr_val(to_sender_packets_acked_stream, -num_acks);
    }

    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        return get_ptr_val(to_sender_packets_completed_stream);
    }

    FORCE_INLINE void increment_num_processed_completions(size_t num_completions) {
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
