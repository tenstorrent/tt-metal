// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"

#include "internal/ethernet/tt_eth_api.h"
#include "internal/ethernet/tunneling.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/flow-control/credits.hpp"

// Packed credit flags: determine whether credits are packed into shared registers
// or sent individually per channel.
//
// NOTE: Packing behavior is determined by VC0's enable_first_level_ack setting and
// applies globally to both VCs. This is acceptable because VC1 always has
// enable_first_level_ack=false (VC1 doesn't use bubble flow control).
//
// Architecture-specific behavior:
// - WH (!multi_txq_enabled): Uses packed stream registers when ENABLE_FIRST_LEVEL_ACK_VC0 is true
// - BH (multi_txq_enabled): Uses counter-based mechanism regardless of ENABLE_FIRST_LEVEL_ACK_VC0
//
constexpr bool USE_PACKED_PACKET_SENT_CREDITS = true;//ENABLE_FIRST_LEVEL_ACK_VC0;      // && !multi_txq_enabled;
constexpr bool USE_PACKED_FIRST_LEVEL_ACK_CREDITS = true;  // && !multi_txq_enabled;
constexpr bool USE_PACKED_COMPLETION_ACK_CREDITS = true;//ENABLE_FIRST_LEVEL_ACK_VC0;   // && !multi_txq_enabled;

// Validation: If VC1 enables first-level ack, VC0 must also have it enabled
// because the packing policy (USE_PACKED_*) is derived from VC0's setting.
static_assert(
    !ENABLE_FIRST_LEVEL_ACK_VC1 || ENABLE_FIRST_LEVEL_ACK_VC0,
    "If VC1 has first-level ack enabled, VC0 must also have it enabled "
    "(global packing policy is governed by VC0's setting)");

struct ReceiverChannelCounterBasedResponseCreditSender {
    ReceiverChannelCounterBasedResponseCreditSender() = default;
    ReceiverChannelCounterBasedResponseCreditSender(size_t receiver_channel_index) :
        completion_counter_l1_ptr(
            reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counters_base_address) + receiver_channel_index),
        ack_counters_base_l1_ptr(reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counters_base_address)),
        ack_counters({}),
        completion_counter(0) {

        // DPRINT << "RECV_INIT: rx_ch=" << (uint32_t)receiver_channel_index
        //        << " local_compl_base=" << HEX() << local_receiver_completion_counters_base_address
        //        << " local_ack_base=" << HEX() << local_receiver_ack_counters_base_address
        //        << " remote_dest=" << HEX() << to_senders_credits_base_address
        //        << " tx_bytes=" << total_number_of_receiver_to_sender_credit_num_bytes << ENDL();

        // Initialize unpacked completion counters (local and L1)
        // With first-level acks, completions are per receiver channel, not per sender channel
        *completion_counter_l1_ptr = 0;  // Initialize L1 memory!
        // Initialize packed ack counters (local and L1)
        for (size_t i = 0; i < ACK_PACKED_WORDS; i++) {
            ack_counters[i] = 0;
            ack_counters_base_l1_ptr[i] = 0;  // Initialize L1 memory!
        }
    }
    
        FORCE_INLINE void send_completion_credit() {
            // Increment completion counter for this receiver channel (shared by all senders in this VC)
            // This represents free buffer space on the receiver side
            uint32_t old_val = completion_counter;
            completion_counter++;
            // DPRINT << "RECV_COMPL: rx_ch=" << (uint32_t)receiver_channel_index
            //        << " completion_counter=" << old_val << "->" << completion_counter
            //        << " L1_ptr=" << HEX() << (uint32_t)completion_counter_l1_ptr
            //        << " remote_addr=" << HEX() << to_senders_credits_base_address << ENDL();
            *completion_counter_l1_ptr = completion_counter;
            update_sender_side_credits();
        }
    
        // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
        FORCE_INLINE void send_ack_credit(uint8_t src_id, int packed_count = 1) {
            if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
                static_assert(NUM_SENDER_CHANNELS <= 8, "NUM_SENDER_CHANNELS must be less than or equal to 8");
    
                // Ack counters use 8-bit credits (4 per word), always byte-aligned
                // Increment the specific channel using byte accessors
                constexpr size_t CREDITS_PER_WORD = 4;
    
                if constexpr (NUM_SENDER_CHANNELS <= CREDITS_PER_WORD) {
                    // ≤4 channels: single word, direct byte access
                    uint8_t* bytes = reinterpret_cast<uint8_t*>(&ack_counters[0]);
                    uint8_t old_val = bytes[src_id];
                    bytes[src_id] += static_cast<uint8_t>(packed_count);
                    uint8_t new_val = bytes[src_id];
                    uint32_t ack_base = (uint32_t)ack_counters_base_l1_ptr;
                    uint32_t ack_write = (uint32_t)&ack_counters_base_l1_ptr[0];
                    // DPRINT << "RECV_ACK: ch=" << (uint32_t)src_id << " +=" << packed_count
                    //        << " byte[" << (uint32_t)src_id << "]=" << (uint32_t)old_val << "->" << (uint32_t)new_val
                    //        << " packed_word0=" << HEX() << ack_counters[0]
                    //        << " L1_base=" << HEX() << ack_base
                    //        << " write_addr=" << HEX() << ack_write
                    //        << " remote_addr=" << HEX() << to_senders_credits_base_address << ENDL();
                    ack_counters_base_l1_ptr[0] = ack_counters[0];
                } else {
                    // 5-8 channels: span 2 words, use branch to select word
                    if (src_id < CREDITS_PER_WORD) {
                        // First word (channels 0-3)
                        uint8_t* bytes = reinterpret_cast<uint8_t*>(&ack_counters[0]);
                        uint8_t old_val = bytes[src_id];
                        bytes[src_id] += static_cast<uint8_t>(packed_count);
                        // DPRINT << "RECV_ACK: ch=" << (uint32_t)src_id << " +=" << packed_count
                        //        << " byte[" << (uint32_t)src_id << "]=" << (uint32_t)old_val << "->" << (uint32_t)bytes[src_id]
                        //        << " packed_word0=" << HEX() << ack_counters[0]
                        //        << " L1_addr=" << HEX() << (uint32_t)&ack_counters_base_l1_ptr[0] << ENDL();
                        ack_counters_base_l1_ptr[0] = ack_counters[0];
                    } else {
                        // Second word (channels 4-7)
                        uint8_t* bytes = reinterpret_cast<uint8_t*>(&ack_counters[1]);
                        uint8_t old_val = bytes[src_id - CREDITS_PER_WORD];
                        bytes[src_id - CREDITS_PER_WORD] += static_cast<uint8_t>(packed_count);
                        // DPRINT << "RECV_ACK: ch=" << (uint32_t)src_id << " +=" << packed_count
                        //        << " byte[" << (uint32_t)(src_id - CREDITS_PER_WORD) << "]=" << (uint32_t)old_val << "->" << (uint32_t)bytes[src_id - CREDITS_PER_WORD]
                        //        << " packed_word1=" << HEX() << ack_counters[1]
                        //        << " L1_addr=" << HEX() << (uint32_t)&ack_counters_base_l1_ptr[1] << ENDL();
                        ack_counters_base_l1_ptr[1] = ack_counters[1];
                    }
                }
    
                update_sender_side_credits();
            } else {
                ack_counters[src_id] += packed_count;
                ack_counters_base_l1_ptr[src_id] = ack_counters[src_id];
                update_sender_side_credits();
            }
        }
    
        // Send packed ACK credits - add packed value directly (formats must match!)
        // Used for batch ACK sending when first-level ACK is enabled
        // ASSUMES: Sender->receiver and receiver->sender use SAME packing format (8-bit on BH)
        template<bool wait_for_txq, typename PackedValueType>
        FORCE_INLINE void send_packed_ack_credits(const PackedValueType& packed_value) {
            if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
                // Both formats use 8-bit byte-aligned packing on BH, so we can add directly
                constexpr size_t CREDITS_PER_WORD = 4;
    
                if constexpr (NUM_SENDER_CHANNELS <= CREDITS_PER_WORD) {
                    // ≤4 channels: single word
                    uint32_t packed_val = static_cast<uint32_t>(packed_value.get());
                    uint32_t old_counter = ack_counters[0];
                    ack_counters[0] += packed_val;
                    ack_counters_base_l1_ptr[0] = ack_counters[0];
                    // DPRINT << "RECV_WRITE_ACK_L1: packed_delta=" << HEX() << packed_val
                    //        << " old=" << old_counter
                    //        << " new=" << ack_counters[0]
                    //        << " L1_addr=" << (uint32_t)ack_counters_base_l1_ptr << ENDL();
                } else {
                    // 5-8 channels: two words
                    uint64_t packed_val = packed_value.get();
                    uint32_t lower_word = static_cast<uint32_t>(packed_val & 0xFFFFFFFFULL);
                    uint32_t old_lower = ack_counters[0];
                    ack_counters[0] += lower_word;
                    ack_counters_base_l1_ptr[0] = ack_counters[0];
    
                    uint32_t upper_word = static_cast<uint32_t>(packed_val >> 32);
                    uint32_t old_upper = ack_counters[1];
                    ack_counters[1] += upper_word;
                    ack_counters_base_l1_ptr[1] = ack_counters[1];
                    // DPRINT << "RECV_WRITE_ACK_L1: packed_delta=" << HEX() << packed_val
                    //        << " lower: old=" << old_lower << " new=" << ack_counters[0]
                    //        << " upper: old=" << old_upper << " new=" << ack_counters[1]
                    //        << " L1_addr=" << (uint32_t)ack_counters_base_l1_ptr << ENDL();
                }
    
                // Wait for eth queue if requested (safer but slower)
                
                if constexpr (wait_for_txq) {
                    while (internal_::eth_txq_is_busy(receiver_txq_id)) {}
                }
                update_sender_side_credits();
            }
        }
    
        volatile uint32_t* tt_l1_ptr completion_counter_l1_ptr;
        volatile uint32_t* tt_l1_ptr ack_counters_base_l1_ptr;
    
        // Ack counters: packed storage (4 credits per word, matching original behavior)
        // 1 word for ≤4 channels, 2 words for 5-8 channels
        static constexpr size_t ACK_PACKED_WORDS = (NUM_SENDER_CHANNELS + 3) / 4;
        std::array<uint32_t, ACK_PACKED_WORDS> ack_counters;
        
        // Completion counters: unpacked, one per receiver channel (fast local memory)
        // With first-level acks enabled, completions are tied to receiver channels not sender channels
        uint32_t completion_counter;
    
private:
    FORCE_INLINE void update_sender_side_credits() const {
        internal_::eth_send_packet_bytes_unsafe(
            receiver_txq_id,
            local_receiver_credits_base_address,
            to_senders_credits_base_address,
            total_number_of_receiver_to_sender_credit_num_bytes);
    }
};

template <bool enable_first_level_ack>
struct ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender {
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender() {
        for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
            // Packing behavior: when enable_first_level_ack is true, all sender channels
            // pack into stream register 0. Otherwise, each uses its own register.
            if constexpr (enable_first_level_ack) {
                sender_channel_packets_completed_stream_ids[i] = to_receiver_packets_sent_streams[0];
                // All sender channels pack into the first ack stream register (register 0)
                sender_channel_packets_ack_stream_ids[i] = to_sender_packets_acked_streams[0];
            } else {
                sender_channel_packets_completed_stream_ids[i] = to_sender_packets_completed_streams[i];
                sender_channel_packets_ack_stream_ids[i] = to_sender_packets_acked_streams[i];
            }
        }
    }

    FORCE_INLINE void send_completion_credit() {
        uint32_t stream_id = sender_channel_packets_completed_stream_ids[0];
        WATCHER_RING_BUFFER_PUSH(0xFCC00000 | (0 << 16) | stream_id);
        // DPRINT << "RECV_COMPL_STREAM: ch=" << (uint32_t)src_id
        //        << " stream_id=" << stream_id
        //        << " remote_update +1" << ENDL();
        remote_update_ptr_val<receiver_txq_id>(stream_id, 1);
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id, int count = 1) {
        uint32_t stream_id = sender_channel_packets_ack_stream_ids[src_id];
        WATCHER_RING_BUFFER_PUSH(0xFAA00000 | (src_id << 16) | stream_id);
        // DPRINT << "RECV_ACK_STREAM: ch=" << (uint32_t)src_id
        //        << " stream_id=" << stream_id
        //        << " remote_update +" << count << ENDL();
        remote_update_ptr_val<receiver_txq_id>(stream_id, count);
    }

    // Send packed ACK credits directly to the shared stream register
    // Used for batch ACK sending when first-level ACK is enabled
    template<typename PackedValueType>
    FORCE_INLINE void send_packed_ack_credits(const PackedValueType& packed_value) {
        if constexpr (enable_first_level_ack) {
            // All channels use register 0 when enable_first_level_ack is true
            uint32_t stream_id = sender_channel_packets_ack_stream_ids[0];
            // packed_value format already matches what sender expects - just write it
            // For ≤32 bits (≤4 channels), write as uint32_t
            // For >32 bits (5 channels), this needs multi-register handling (TODO if needed)
            static_assert(sizeof(typename PackedValueType::storage_type) <= 8,
                         "Packed value storage too large");
            if constexpr (sizeof(typename PackedValueType::storage_type) <= 4) {
                remote_update_ptr_val<receiver_txq_id>(stream_id, static_cast<uint32_t>(packed_value.get()));
            } else {
                // Multi-register case - need to write to both stream_ids[0] and stream_ids[1]
                // For now, assume this is handled elsewhere or not needed
                static_assert(sizeof(typename PackedValueType::storage_type) <= 4,
                             "Multi-register packed ACK credits not yet implemented for stream registers");
            }
        }
    }

    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_packets_completed_stream_ids;
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_packets_ack_stream_ids;
};

// Type alias for stream-register-based credit sender using global packing policy
// Both VCs use the same packing behavior (determined by USE_PACKED_* flags derived from VC0)
using ReceiverChannelStreamRegisterCreditSender =
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender<USE_PACKED_FIRST_LEVEL_ACK_CREDITS>;

using ReceiverChannelResponseCreditSender = typename std::conditional_t<
    multi_txq_enabled,
    ReceiverChannelCounterBasedResponseCreditSender,
    ReceiverChannelStreamRegisterCreditSender>;

template <typename T = void>
struct init_receiver_channel_response_credit_senders_impl;

// Implementation for ReceiverChannelStreamRegisterCreditSender
template <>
struct init_receiver_channel_response_credit_senders_impl<ReceiverChannelStreamRegisterCreditSender> {
    template <uint8_t NUM_RECEIVER_CHANNELS>
    static constexpr auto init() -> std::array<ReceiverChannelStreamRegisterCreditSender, NUM_RECEIVER_CHANNELS> {
        std::array<ReceiverChannelStreamRegisterCreditSender, NUM_RECEIVER_CHANNELS> credit_senders;
        for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
            credit_senders[i] = ReceiverChannelStreamRegisterCreditSender();
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
    SenderChannelFromReceiverCounterBasedCreditsReceiver(size_t receiver_channel_index) :
        acks_received_counter_ptr(
            reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address)),  // Packed: all channels read same address
        acks_received_and_processed(0) {
        // Note: Completion tracking is now in OutboundReceiverChannelPointers (per-receiver-channel)
        // instead of here (per-sender-channel) since completions are shared across all senders to same receiver
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t sender_channel_index>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        // For counter-based (BH), ack counters are 8-bit packed and wrap at 256
        // Must use uint8_t arithmetic to handle wraparound correctly
        // Return value must be in packed form (byte in correct position)
        constexpr uint32_t byte_shift = sender_channel_index * 8;
        uint32_t raw_value = *acks_received_counter_ptr;
        uint8_t received = static_cast<uint8_t>(raw_value >> byte_shift);
        uint8_t processed = static_cast<uint8_t>(acks_received_and_processed >> byte_shift);
        uint8_t diff = received - processed;  // uint8_t subtraction wraps correctly
        uint32_t result = static_cast<uint32_t>(diff) << byte_shift;
        return result;  // Return in packed position
    }

    template <uint8_t sender_channel_index>
    FORCE_INLINE void increment_num_processed_acks(size_t packed_num_acks) {
        if constexpr (USE_PACKED_FIRST_LEVEL_ACK_CREDITS) {
            // Extract this channel's byte from both values, add with wraparound, then pack back
            constexpr uint32_t byte_shift = sender_channel_index * 8;
            constexpr uint32_t byte_mask = 0xFF << byte_shift;

            uint32_t current_byte = acks_received_and_processed & byte_mask;
            uint32_t delta_byte = packed_num_acks & byte_mask;
            uint32_t new_byte = (current_byte + delta_byte) & byte_mask;  // Addition with mask for wraparound
            uint8_t old_unpacked = static_cast<uint8_t>(current_byte >> byte_shift);
            uint8_t delta_unpacked = static_cast<uint8_t>(delta_byte >> byte_shift);
            uint8_t new_unpacked = static_cast<uint8_t>(new_byte >> byte_shift);

            // DPRINT << "SEND_PROC_ACK: ch=" << (uint32_t)sender_channel_index
            //        << " old_packed=" << HEX() << acks_received_and_processed
            //        << " byte[" << (uint32_t)sender_channel_index << "]=" << DEC() << (uint32_t)old_unpacked << "->" << (uint32_t)new_unpacked
            //        << " delta=" << (uint32_t)delta_unpacked
            //        << " new_packed=" << HEX() << ((acks_received_and_processed & ~byte_mask) | new_byte) << ENDL();

            acks_received_and_processed = (acks_received_and_processed & ~byte_mask) | new_byte;
        } else {
            // For counter-based (BH), ack counters are 8-bit and wrap at 256
            // Must use uint8_t semantics to match receiver's wraparound behavior
            uint32_t old_val = acks_received_and_processed;
            acks_received_and_processed = static_cast<uint32_t>(
                static_cast<uint8_t>(acks_received_and_processed) + static_cast<uint8_t>(packed_num_acks)
            );
            // DPRINT << "SEND_PROC_ACK: ch=" << (uint32_t)sender_channel_index
            //        << " old=" << (uint32_t)static_cast<uint8_t>(old_val)
            //        << " delta=" << (uint32_t)static_cast<uint8_t>(packed_num_acks)
            //        << " new=" << (uint32_t)static_cast<uint8_t>(acks_received_and_processed) << ENDL();
        }
    }

    // Completion methods removed - now in OutboundReceiverChannelPointers

    volatile uint32_t* acks_received_counter_ptr;
    uint32_t acks_received_and_processed = 0;
};

template <bool enable_first_level_ack>
struct SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver {
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver() = default;

    // Packing behavior: when enable_first_level_ack is true, all sender channels use
    // register 0 for packed credits. Otherwise, each uses its own register.
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver(size_t sender_channel_index) :
        to_sender_packets_acked_stream(
            // enable_first_level_ack
                // ? 
                to_sender_packets_acked_streams[0]  // All channels use register 0 for packed credits
                // : to_sender_packets_acked_streams[sender_channel_index]
            ),
        to_sender_packets_completed_stream(
            // enable_first_level_ack ? 
            to_receiver_packets_sent_streams[0]
                                //    : to_sender_packets_completed_streams[sender_channel_index]
            ) {}

    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t sender_channel_index>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        uint32_t acks = get_ptr_val(to_sender_packets_acked_stream);
        // if (acks > 0) {
        //     DPRINT << "SEND_GET_ACK_STREAM: ch=" << (uint32_t)sender_channel_index
        //            << " stream_id=" << to_sender_packets_acked_stream
        //            << " acks=" << acks << ENDL();
        // }
        return acks;
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) {
        // DPRINT << "SEND_PROC_ACK_STREAM: stream_id=" << to_sender_packets_acked_stream
        //        << " decrement -" << num_acks << ENDL();
        increment_local_update_ptr_val(to_sender_packets_acked_stream, -num_acks);
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        uint32_t completions = get_ptr_val(to_sender_packets_completed_stream);
        // if (completions > 0) {
        //     DPRINT << "SEND_GET_COMPL_STREAM: ch=" << (uint32_t)sender_channel_index
        //            << " stream_id=" << to_sender_packets_completed_stream
        //            << " completions=" << completions << ENDL();
        // }
        return completions;
    }

    FORCE_INLINE void increment_num_processed_completions(size_t num_completions) {
        // DPRINT << "SEND_PROC_COMPL_STREAM: stream_id=" << to_sender_packets_completed_stream
        //        << " decrement -" << num_completions << ENDL();
        increment_local_update_ptr_val(to_sender_packets_completed_stream, -num_completions);
    }

    uint32_t to_sender_packets_acked_stream;
    uint32_t to_sender_packets_completed_stream;
};

// Type alias for stream-register-based credits receiver using global packing policy
// All sender channels use the same packing behavior (determined by USE_PACKED_* flags)
using SenderChannelFromReceiverStreamRegisterCreditsReceiver =
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver<USE_PACKED_FIRST_LEVEL_ACK_CREDITS>;

template <typename T = void>
struct init_sender_channel_from_receiver_credits_flow_controllers_impl;

// Implementation for SenderChannelFromReceiverStreamRegisterCreditsReceiver
template <>
struct init_sender_channel_from_receiver_credits_flow_controllers_impl<
    SenderChannelFromReceiverStreamRegisterCreditsReceiver> {
    template <uint8_t NUM_SENDER_CHANNELS>
    static constexpr auto init()
        -> std::array<SenderChannelFromReceiverStreamRegisterCreditsReceiver, NUM_SENDER_CHANNELS> {
        std::array<SenderChannelFromReceiverStreamRegisterCreditsReceiver, NUM_SENDER_CHANNELS> flow_controllers;
        for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
            new (&flow_controllers[i]) SenderChannelFromReceiverStreamRegisterCreditsReceiver(i);
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
            // Map sender channel to receiver channel (VC0 or VC1)
            size_t receiver_channel = (i < ACTUAL_VC0_SENDER_CHANNELS) ? VC0_RECEIVER_CHANNEL : VC1_RECEIVER_CHANNEL;
            new (&flow_controllers[i]) SenderChannelFromReceiverCounterBasedCreditsReceiver(receiver_channel);
        }
        return flow_controllers;
    }
};

using SenderChannelFromReceiverCredits = typename std::conditional_t<
    multi_txq_enabled,
    SenderChannelFromReceiverCounterBasedCreditsReceiver,
    SenderChannelFromReceiverStreamRegisterCreditsReceiver>;

// SFINAE-based overload for !multi_txq_enabled case (WH with stream registers)
template <uint8_t NUM_SENDER_CHANNELS>
constexpr FORCE_INLINE auto init_sender_channel_from_receiver_credits_flow_controllers()
    -> std::enable_if_t<!multi_txq_enabled, std::array<SenderChannelFromReceiverCredits, NUM_SENDER_CHANNELS>> {
    return init_sender_channel_from_receiver_credits_flow_controllers_impl<
        SenderChannelFromReceiverStreamRegisterCreditsReceiver>::template init<NUM_SENDER_CHANNELS>();
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
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender) {
    if constexpr (CHECK_BUSY) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        };
    }
    receiver_channel_response_credit_sender.send_completion_credit();
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
 * - WH with enable_first_level_ack: Uses packed credits (multiple channels in one register)
 * - BH with multi_txq: Uses counter-based (one counter per channel, no packing)
 * - Non-enable_first_level_ack: Uses individual stream registers (no packing)
 */

/**
 * Extract individual sender channel ACKs from packed value.
 * WH: Unpacks from shared register. BH: Pass-through (already unpacked).
 */
template <uint8_t sender_channel_index>
FORCE_INLINE uint32_t extract_sender_channel_acks(uint32_t packed_acks) {
    if constexpr (USE_PACKED_FIRST_LEVEL_ACK_CREDITS) {
        if constexpr (multi_txq_enabled) {
            return (packed_acks >> (sender_channel_index * 8)) & 0xFF;
        } else {
            // WH: Stream register with packing - need to extract this channel's credits
            using PackedCreditsType = tt::tt_fabric::PackedCredits<
                NUM_SENDER_CHANNELS,
                tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS,
                to_sender_packets_acked_streams[0]>;
            auto packed_acks_named = typename PackedCreditsType::PackedCreditValueType{packed_acks};
            return PackedCreditsType::template get_value<sender_channel_index>(packed_acks_named);
        }
    } else {
        // BH or !enable_first_level_ack: Counter-based, already unpacked (one counter per channel)
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
        if constexpr (multi_txq_enabled) {
            return acks_count << (sender_channel_index * 8);
        } else {
            // WH: Pack into register position for this channel
            return tt::tt_fabric::PackedCredits<
                    NUM_SENDER_CHANNELS,
                    tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS,
                    to_sender_packets_acked_streams[0]>::template pack_value<sender_channel_index>(acks_count)
                .get();
        }
    } else {
        // BH or !enable_first_level_ack: Direct value, no packing
        return acks_count;
    }
}

// ============================================================================
// Receiver Packet Credits - Configuration
// ============================================================================

/**
 * Determine credit width for receiver packet credits (matching legacy PackedCredits behavior)
 */
namespace receiver_credits_config {
    constexpr uint32_t MIN_BITS = tt::tt_fabric::log2_ceil(tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS + 1);

    // Prefer 8-bit (byte-aligned) when it fits in available registers
    // Check if 8-bit packing fits in single 24-bit register
    constexpr bool byte_aligned_fits_in_single_reg = (NUM_SENDER_CHANNELS * 8) <= 24;
    // Check if 8-bit packing fits in two registers (max 6 channels × 8 = 48 bits < 64 bits)
    constexpr bool byte_aligned_fits_in_two_regs = (NUM_SENDER_CHANNELS * 8) <= 64;

    // Always prefer 8-bit for consistency when possible
    constexpr bool credits_are_byte_aligned = byte_aligned_fits_in_two_regs;
    constexpr uint8_t CREDIT_WIDTH = credits_are_byte_aligned ? 8 : MIN_BITS;
    constexpr uint32_t TOTAL_BITS = NUM_SENDER_CHANNELS * CREDIT_WIDTH;
    constexpr bool NEEDS_MULTI_REGISTER = TOTAL_BITS > 24;
    constexpr size_t PACKED_WORDS_COUNT = NEEDS_MULTI_REGISTER ? 2 : 1;
}

/**
 * Determine which stream register ID to write to for a given sender channel.
 * In multi-register cases (4-5 channels):
 * - Channels 0-1 → reg0 (base stream ID)
 * - Channels 2-4 → reg1 (base stream ID + 1)
 * In single-register cases (1-3 channels): always base stream ID
 */
template <uint8_t sender_channel_index, uint32_t base_stream_id>
constexpr uint32_t get_sender_target_stream_id() {
    if constexpr (receiver_credits_config::NEEDS_MULTI_REGISTER) {
        // Multi-register layout: reg0 has channels 0-1, reg1 has channels 2-4
        constexpr uint8_t CHANNELS_IN_REG0 = 2;
        if constexpr (sender_channel_index < CHANNELS_IN_REG0) {
            return base_stream_id;
        } else {
            return base_stream_id + 1;
        }
    } else {
        // Single register: all channels use base stream ID
        return base_stream_id;
    }
}

/**
 * Build packet forward increment value for sender channel.
 * WH: Pack into register position (supports channels 0-4 at appropriate offsets).
 * BH: Always 1 (no packing).
 *
 * For packed credits, this packs a value of 1 at the bit offset corresponding to
 * sender_channel_index (0-4). In multi-register cases (4-5 channels), this returns
 * only the portion relevant to the register this channel maps to:
 * - Channels 0-1 → write to reg0 (to_receiver_pkts_sent_id)
 * - Channels 2-4 → write to reg1 (to_receiver_pkts_sent_id + 1)
 */
template <uint8_t sender_channel_index, uint32_t to_receiver_pkts_sent_id>
FORCE_INLINE constexpr uint32_t build_packet_forward_value() {
    if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
        // WH: Pack 1 credit into this channel's position using generic credit packing
        using Packing = tt::tt_fabric::CreditPacking<NUM_SENDER_CHANNELS, receiver_credits_config::CREDIT_WIDTH>;
        static_assert(sender_channel_index < NUM_SENDER_CHANNELS, "Sender channel index out of bounds");

        // Pack value of 1 at the bit offset for sender_channel_index
        auto packed = Packing::template pack_channel<sender_channel_index>(1);

        // For multi-register case, extract only the relevant register's portion
        if constexpr (receiver_credits_config::NEEDS_MULTI_REGISTER) {
            // Multi-register layout: reg0 has channels 0-1, reg1 has channels 2-4
            constexpr uint8_t CHANNELS_IN_REG0 = 2;
            constexpr uint32_t reg1_shift = CHANNELS_IN_REG0 * receiver_credits_config::CREDIT_WIDTH;

            if constexpr (sender_channel_index < CHANNELS_IN_REG0) {
                // Channels 0-1: extract lower 16 bits for reg0
                return static_cast<uint32_t>(packed.get() & ((1ULL << reg1_shift) - 1));
            } else {
                // Channels 2-4: extract and shift down upper bits for reg1
                return static_cast<uint32_t>(packed.get() >> reg1_shift);
            }
        } else {
            // Single register case: return the full packed value
            return static_cast<uint32_t>(packed.get());
        }
    } else {
        // BH or !enable_first_level_ack: Always 1 (counter-based, no packing needed)
        return 1;
    }
}

// ============================================================================
// Receiver Packet Credits - Generic Type Definitions
// ============================================================================

/**
 * Template helper to resolve receiver packet credit view type based on stream ID(s).
 * Automatically selects single or multi-register storage based on total bits needed.
 *
 * Usage:
 *   using MyCredits = ReceiverPacketCreditsViewFor<my_stream_id>;
 *   MyCredits credits;
 */
template <size_t stream_id_0, size_t stream_id_1 = stream_id_0 + 1>
using ReceiverPacketCreditsViewFor = std::conditional_t<
    receiver_credits_config::NEEDS_MULTI_REGISTER,
    // Multi-register case (>24 bits): 4-5 channels with 8-bit packing
    tt::tt_fabric::MultiOverlayRegCreditView<
        NUM_SENDER_CHANNELS,
        receiver_credits_config::CREDIT_WIDTH,
        stream_id_0,
        stream_id_1>,
    // Single-register case (≤24 bits): 1-3 channels, or 4 channels with tight packing
    tt::tt_fabric::OverlayRegCreditView<
        NUM_SENDER_CHANNELS,
        receiver_credits_config::CREDIT_WIDTH,
        stream_id_0>>;

/**
 * Template helper to resolve receiver packet credit updater type based on stream ID(s).
 * Provides write operations including decrement_packed() for clearing credits.
 *
 * Usage:
 *   using MyCreditsUpdater = ReceiverPacketCreditsUpdaterFor<my_stream_id>;
 *   MyCreditsUpdater updater;
 *   updater.decrement_packed(value);
 */
template <size_t stream_id_0, size_t stream_id_1 = stream_id_0 + 1>
using ReceiverPacketCreditsUpdaterFor = std::conditional_t<
    receiver_credits_config::NEEDS_MULTI_REGISTER,
    // Multi-register case (>24 bits): 4-5 channels with 8-bit packing
    tt::tt_fabric::MultiOverlayRegCreditUpdater<
        NUM_SENDER_CHANNELS,
        receiver_credits_config::CREDIT_WIDTH,
        stream_id_0,
        stream_id_1>,
    // Single-register case (≤24 bits): 1-3 channels, or 4 channels with tight packing
    tt::tt_fabric::OverlayRegCreditUpdater<
        NUM_SENDER_CHANNELS,
        receiver_credits_config::CREDIT_WIDTH,
        stream_id_0>>;

/**
 * Default receiver packet credit view type using to_sender_packets_completed_streams.
 * For legacy compatibility.
 */
using ReceiverPacketCreditsView = ReceiverPacketCreditsViewFor<
    to_sender_packets_completed_streams[0],
    to_sender_packets_completed_streams[1]>;

/**
 * Accumulate receiver channel credits from packed value.
 * WH: Unpack and sum all channels. BH: Direct value (no packing).
 */
FORCE_INLINE uint32_t accumulate_receiver_channel_credits(uint32_t packed_value) {
    if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
        // WH: Unpack and sum across all sender channels using generic packing layer
        using Packing = tt::tt_fabric::CreditPacking<NUM_SENDER_CHANNELS, receiver_credits_config::CREDIT_WIDTH>;
        typename Packing::PackedValueType packed{packed_value};
        return Packing::sum_all_channels(packed);
    } else {
        // BH or !enable_first_level_ack: Already unpacked (direct count)
        return packed_value;
    }
}
