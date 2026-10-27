// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"

#include "internal/ethernet/tt_eth_api.h"
#include "internal/ethernet/tunneling.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_credit_transport.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

// Packets-sent credit, sender -> receiver over the link, keyed by the receiver channel's stream id in
// both modes. With stream registers the sender adds 1 to the receiver's register and the receiver
// subtracts each packet it consumes. Without, the sender keeps its free-running total in
// packets_sent_totals and sends it into the receiver's slot after each packet; the receiver counts
// what it consumed in packets_sent_processed and never writes the slot it polls.
template <uint32_t PKTS_SENT_ID>
constexpr size_t receiver_channel_of_packets_sent_stream() {
    size_t channel = 0;
    while (channel < MAX_NUM_RECEIVER_CHANNELS && to_receiver_packets_sent_streams[channel] != PKTS_SENT_ID) {
        channel++;
    }
    return channel;
}

inline std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> packets_sent_totals{};
inline std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> packets_sent_processed{};

FORCE_INLINE volatile uint32_t* packets_sent_slot(size_t base_address, size_t channel) {
    return reinterpret_cast<volatile uint32_t*>(base_address + channel * software_counter_slot_bytes);
}

// Before the Ethernet handshake, so the peer cannot write the receiver's slot yet.
template <uint32_t PKTS_SENT_ID>
FORCE_INLINE void init_packets_sent_credit() {
    if constexpr (stream_registers_available) {
        init_ptr_val<PKTS_SENT_ID>(0);
    } else {
        constexpr size_t channel = receiver_channel_of_packets_sent_stream<PKTS_SENT_ID>();
        static_assert(channel < MAX_NUM_RECEIVER_CHANNELS, "unknown stream id");
        packets_sent_totals[channel] = 0;
        packets_sent_processed[channel] = 0;
        *packets_sent_slot(to_receiver_packets_sent_counters_base_address, channel) = 0;
        *packets_sent_slot(local_sender_packets_sent_counters_base_address, channel) = 0;
    }
}

// Packets the sender has sent on this receiver channel that the receiver has not yet consumed.
template <uint32_t PKTS_SENT_ID>
FORCE_INLINE int32_t get_packets_sent_unprocessed() {
    if constexpr (stream_registers_available) {
        return get_ptr_val<PKTS_SENT_ID>();
    } else {
        constexpr size_t channel = receiver_channel_of_packets_sent_stream<PKTS_SENT_ID>();
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        return static_cast<int32_t>(
            *packets_sent_slot(to_receiver_packets_sent_counters_base_address, channel) -
            packets_sent_processed[channel]);
    }
}

// The receiver consumed one packet.
template <uint32_t PKTS_SENT_ID>
FORCE_INLINE void consume_packets_sent_credit() {
    if constexpr (stream_registers_available) {
        increment_local_update_ptr_val<PKTS_SENT_ID>(-1);
    } else {
        packets_sent_processed[receiver_channel_of_packets_sent_stream<PKTS_SENT_ID>()]++;
    }
}

// The sender sent one packet. Assumes !eth_txq_is_busy(TXQ_ID), and must follow the payload on the
// same queue.
template <uint32_t PKTS_SENT_ID, uint32_t TXQ_ID>
FORCE_INLINE void send_packets_sent_credit() {
    if constexpr (stream_registers_available) {
        remote_update_ptr_val<PKTS_SENT_ID, TXQ_ID>(1U);
    } else {
        constexpr size_t channel = receiver_channel_of_packets_sent_stream<PKTS_SENT_ID>();
        uint32_t total = ++packets_sent_totals[channel];
        *packets_sent_slot(local_sender_packets_sent_counters_base_address, channel) = total;
        fabric_send_credit_word(
            TXQ_ID,
            local_sender_packets_sent_counters_base_address + channel * software_counter_slot_bytes,
            to_receiver_packets_sent_counters_base_address + channel * software_counter_slot_bytes,
            total);
    }
}

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

    volatile tt_l1_ptr uint32_t* completion_counters_base_ptr;
    volatile tt_l1_ptr uint32_t* ack_counters_base_ptr;
    // Local memory copy to save an L1 load
    std::array<uint32_t, NUM_SENDER_CHANNELS> completion_counters;
    std::array<uint32_t, NUM_SENDER_CHANNELS> ack_counters;

private:
    FORCE_INLINE void update_sender_side_credits() const {
        fabric_send_credit_block(
            receiver_txq_id,
            local_receiver_credits_base_address,
            to_senders_credits_base_address,
            total_number_of_receiver_to_sender_credit_num_bytes);
    }
};

struct ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender {
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender() {
        for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
            sender_channel_packets_completed_stream_ids[i] = to_sender_packets_completed_streams[i];
            sender_channel_packets_ack_stream_ids[i] = to_sender_packets_acked_streams[i];
        }
    }

    FORCE_INLINE void send_completion_credit(uint8_t src_id, uint32_t num_completions) {
        remote_update_ptr_val<receiver_txq_id>(sender_channel_packets_completed_stream_ids[src_id], num_completions);
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id) {
        remote_update_ptr_val<receiver_txq_id>(sender_channel_packets_ack_stream_ids[src_id], 1);
    }

    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_packets_completed_stream_ids;
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_packets_ack_stream_ids;
};

// Credit transport is chosen per VC, so the type depends on the channel rather than being one alias for
// the whole router. One receiver channel serves each VC and credits only flow back within a VC, so the
// receiver channel index selects directly.
template <size_t RECEIVER_CHANNEL>
using ReceiverChannelResponseCreditSenderFor = std::conditional_t<
    receiver_channel_uses_counter_credits(RECEIVER_CHANNEL),
    ReceiverChannelCounterBasedResponseCreditSender,
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender>;

// The two implementations do not agree on how they are built: the counter form needs its channel
// index, the stream-register form reads the stream id table in its default constructor.
template <typename CreditSender>
FORCE_INLINE void construct_credit_sender_in_place(CreditSender& slot, size_t channel_index) {
    if constexpr (std::is_constructible_v<CreditSender, size_t>) {
        new (&slot) CreditSender(channel_index);
    } else {
        new (&slot) CreditSender();
    }
}

template <typename Sequence>
struct ReceiverChannelResponseCreditSendersImpl;

template <size_t... Is>
struct ReceiverChannelResponseCreditSendersImpl<std::index_sequence<Is...>> {
    FORCE_INLINE ReceiverChannelResponseCreditSendersImpl() {
        (construct_credit_sender_in_place(std::get<Is>(credit_senders), Is), ...);
    }

    std::tuple<ReceiverChannelResponseCreditSenderFor<Is>...> credit_senders;

    // Every access site has the channel as a compile-time value, so no runtime indexing is needed.
    template <size_t RECEIVER_CHANNEL>
    FORCE_INLINE auto& get() {
        return std::get<RECEIVER_CHANNEL>(credit_senders);
    }
};

template <uint8_t NUM_RECEIVER_CHANNELS>
using ReceiverChannelResponseCreditSenders =
    ReceiverChannelResponseCreditSendersImpl<std::make_index_sequence<NUM_RECEIVER_CHANNELS>>;

template <uint8_t NUM_RECEIVER_CHANNELS>
constexpr FORCE_INLINE auto init_receiver_channel_response_credit_senders()
    -> ReceiverChannelResponseCreditSenders<NUM_RECEIVER_CHANNELS> {
    return ReceiverChannelResponseCreditSenders<NUM_RECEIVER_CHANNELS>{};
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
        to_sender_packets_acked_stream(to_sender_packets_acked_streams[sender_channel_index]),
        to_sender_packets_completed_stream(to_sender_packets_completed_streams[sender_channel_index]) {}

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        return get_ptr_val(to_sender_packets_acked_stream);
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) {
        increment_local_update_ptr_val(to_sender_packets_acked_stream, -num_acks);
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        return get_ptr_val(to_sender_packets_completed_stream);
    }

    FORCE_INLINE void increment_num_processed_completions(size_t num_completions) {
        increment_local_update_ptr_val(to_sender_packets_completed_stream, -num_completions);
    }

    uint32_t to_sender_packets_acked_stream;
    uint32_t to_sender_packets_completed_stream;
};

// Same per-VC selection on the sender side. Flat sender channels are laid out VC0 then VC1 then VC2,
// so the channel index resolves to a VC without any extra state.
template <size_t SENDER_CHANNEL>
using SenderChannelFromReceiverCreditsFor = std::conditional_t<
    sender_channel_uses_counter_credits(SENDER_CHANNEL),
    SenderChannelFromReceiverCounterBasedCreditsReceiver,
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver>;

template <typename Sequence>
struct SenderChannelFromReceiverCreditsImpl;

template <size_t... Is>
struct SenderChannelFromReceiverCreditsImpl<std::index_sequence<Is...>> {
    FORCE_INLINE SenderChannelFromReceiverCreditsImpl() {
        (construct_credit_sender_in_place(std::get<Is>(credits), Is), ...);
    }

    std::tuple<SenderChannelFromReceiverCreditsFor<Is>...> credits;

    template <size_t SENDER_CHANNEL>
    FORCE_INLINE auto& get() {
        return std::get<SENDER_CHANNEL>(credits);
    }
};

template <uint8_t NUM_SENDER_CHANNELS>
using SenderChannelFromReceiverCredits =
    SenderChannelFromReceiverCreditsImpl<std::make_index_sequence<NUM_SENDER_CHANNELS>>;

template <uint8_t NUM_SENDER_CHANNELS>
constexpr FORCE_INLINE auto init_sender_channel_from_receiver_credits_flow_controllers()
    -> SenderChannelFromReceiverCredits<NUM_SENDER_CHANNELS> {
    return SenderChannelFromReceiverCredits<NUM_SENDER_CHANNELS>{};
}

// MUST CHECK !is_eth_txq_busy() before calling
// The sender type is a parameter because each receiver channel picks its own credit transport
// (ReceiverChannelResponseCreditSenderFor), so callers hand in whichever one their channel holds.
template <bool CHECK_BUSY, typename ReceiverChannelResponseCreditSenderT>
FORCE_INLINE void receiver_send_completion_ack(
    ReceiverChannelResponseCreditSenderT& receiver_channel_response_credit_sender,
    uint8_t src_id,
    uint32_t num_completions = 1) {
    if constexpr (CHECK_BUSY) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        };
    }
    receiver_channel_response_credit_sender.send_completion_credit(src_id, num_completions);
}

template <bool CHECK_BUSY, typename ReceiverChannelResponseCreditSenderT>
FORCE_INLINE void receiver_send_received_ack(
    ReceiverChannelResponseCreditSenderT& receiver_channel_response_credit_sender, uint8_t src_id) {
    if constexpr (CHECK_BUSY) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        };
    }
    receiver_channel_response_credit_sender.send_ack_credit(src_id);
}
