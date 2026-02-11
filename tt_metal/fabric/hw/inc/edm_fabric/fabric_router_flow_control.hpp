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
// - BH (multi_txq_enabled): Uses counter-based mechanism regardless of ENABLE_FIRST_LEVEL_ACK_VC0
//
constexpr bool USE_PACKED_PACKET_SENT_CREDITS = ENABLE_FIRST_LEVEL_ACK_VC0;
constexpr bool USE_PACKED_FIRST_LEVEL_ACK_CREDITS = ENABLE_FIRST_LEVEL_ACK_VC0;
constexpr bool USE_PACKED_COMPLETION_ACK_CREDITS = ENABLE_FIRST_LEVEL_ACK_VC0;

// Validation: If VC1 enables first-level ack, VC0 must also have it enabled
// because the packing policy (USE_PACKED_*) is derived from VC0's setting.
static_assert(
    !ENABLE_FIRST_LEVEL_ACK_VC1 || ENABLE_FIRST_LEVEL_ACK_VC0,
    "If VC1 has first-level ack enabled, VC0 must also have it enabled "
    "(global packing policy is governed by VC0's setting)");

template <bool enable_first_level_ack>
struct ReceiverChannelCounterBasedResponseCreditSender {
    ReceiverChannelCounterBasedResponseCreditSender() = default;
    ReceiverChannelCounterBasedResponseCreditSender(size_t receiver_channel_index) :
        completion_counter_l1_ptr(
            reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counters_base_address) +
            (enable_first_level_ack ? receiver_channel_index : 0)),
        ack_counters_base_l1_ptr(reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counters_base_address)),
        ack_counters({}) {
        // Initialize completion counters (local and L1)
        if constexpr (enable_first_level_ack) {
            // With first-level acks, completions are per receiver channel, not per sender channel
            completion_counter = 0;
            *completion_counter_l1_ptr = 0;  // Initialize L1 memory!
        } else {
            // Without first-level acks, completions are per sender channel
            for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
                completion_counter[i] = 0;
                completion_counter_l1_ptr[i] = 0;  // Initialize L1 memory!
            }
        }
        // Initialize packed ack counters (local and L1)
        for (size_t i = 0; i < ACK_PACKED_WORDS; i++) {
            ack_counters[i] = 0;
            ack_counters_base_l1_ptr[i] = 0;  // Initialize L1 memory!
        }
    }

    FORCE_INLINE void send_completion_credit(uint8_t src_id = 0) {
        // Increment completion counter
        if constexpr (enable_first_level_ack) {
            // Receiver-channel-based: single scalar counter (src_id ignored)
            // This represents free buffer space on the receiver side (shared by all senders in this VC)
            completion_counter++;
            *completion_counter_l1_ptr = completion_counter;
        } else {
            // Sender-channel-based: array indexed by src_id (matches pre-cdfbd972cde working implementation)
            completion_counter[src_id]++;
            completion_counter_l1_ptr[src_id] = completion_counter[src_id];
        }
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
                    ack_counters_base_l1_ptr[0] = ack_counters[0];
                } else {
                    // 5-8 channels: span 2 words, use branch to select word
                    if (src_id < CREDITS_PER_WORD) {
                        // First word (channels 0-3)
                        uint8_t* bytes = reinterpret_cast<uint8_t*>(&ack_counters[0]);
                        uint8_t old_val = bytes[src_id];
                        bytes[src_id] += static_cast<uint8_t>(packed_count);
                        ack_counters_base_l1_ptr[0] = ack_counters[0];
                    } else {
                        // Second word (channels 4-7)
                        uint8_t* bytes = reinterpret_cast<uint8_t*>(&ack_counters[1]);
                        uint8_t old_val = bytes[src_id - CREDITS_PER_WORD];
                        bytes[src_id - CREDITS_PER_WORD] += static_cast<uint8_t>(packed_count);
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
        template <size_t VC_NUM_SENDER_CHANNELS, bool wait_for_txq, typename PackedValueType>
        FORCE_INLINE void send_packed_ack_credits(const PackedValueType& packed_value) {
            if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
                // Use safe CreditPacking helper to add packed values without carries between channels
                // This replaces the previous manual byte-by-byte loop with optimized, tested code
                using PackingType = tt::tt_fabric::CreditPacking<NUM_SENDER_CHANNELS, 8>;  // 8-bit credits

                constexpr size_t CREDITS_PER_WORD = 4;

                if constexpr (NUM_SENDER_CHANNELS <= CREDITS_PER_WORD) {
                    // ≤4 channels: single word
                    // Safe multi-channel addition using masked-pair addition (branch-free)
                    auto current = PackingType::PackedValueType{ack_counters[0]};
                    auto delta = PackingType::PackedValueType{static_cast<uint32_t>(packed_value.get())};

                    uint32_t current_val = current.value;
                    uint32_t delta_val = delta.value;

                    // Masked-pair addition: prevents carries between bytes
                    // Add to bytes 0,2: any carry goes to bytes 1,3 which we mask out
                    uint32_t b0_2 = current_val + (delta_val & 0x00FF00FFu);
                    // Add to bytes 1,3: any carry goes to bytes 2,4 which we mask out
                    uint32_t b1_3 = current_val + (delta_val & 0xFF00FF00u);
                    // Combine: take bytes 0,2 from b0_2 and bytes 1,3 from b1_3
                    uint32_t result = (b0_2 & 0x00FF00FFu) | (b1_3 & 0xFF00FF00u);

                    ack_counters[0] = result;
                    ack_counters_base_l1_ptr[0] = result;
                } else {
                    // 5 channels: two words (4 in lower, 1 in upper)
                    // Max channels per VC is 5
                    static_assert(NUM_SENDER_CHANNELS == 5, "Multi-word case expects exactly 5 channels");

                    uint64_t packed_val = packed_value.get();

                    // Lower word: 4 channels (bytes 0-3) - use masked-pair addition
                    uint32_t current_lower = ack_counters[0];
                    uint32_t delta_lower = static_cast<uint32_t>(packed_val & 0xFFFFFFFFULL);

                    // Masked-pair addition for lower 4 channels
                    uint32_t b0_2_lower = current_lower + (delta_lower & 0x00FF00FFu);
                    uint32_t b1_3_lower = current_lower + (delta_lower & 0xFF00FF00u);
                    uint32_t result_lower = (b0_2_lower & 0x00FF00FFu) | (b1_3_lower & 0xFF00FF00u);

                    ack_counters[0] = result_lower;
                    ack_counters_base_l1_ptr[0] = result_lower;

                    // Upper word: 1 channel (byte 4) - single byte, no carries possible
                    uint8_t current_upper = static_cast<uint8_t>(ack_counters[1] & 0xFF);
                    uint8_t delta_upper = static_cast<uint8_t>((packed_val >> 32) & 0xFF);
                    uint8_t result_upper = current_upper + delta_upper;  // uint8_t wraps naturally

                    ack_counters[1] = result_upper;
                    ack_counters_base_l1_ptr[1] = result_upper;
                }
                // Wait for eth queue if requested (safer but slower)

                if constexpr (wait_for_txq) {
                    while (internal_::eth_txq_is_busy(receiver_txq_id)) {}
                }
                // update_sender_side_credits();
            }
        }

        volatile tt_l1_ptr uint32_t* completion_counter_l1_ptr;
        volatile tt_l1_ptr uint32_t* ack_counters_base_l1_ptr;

        // Ack counters: packed storage (4 credits per word, matching original behavior)
        // 1 word for ≤4 channels, 2 words for 5-8 channels
        static constexpr size_t ACK_PACKED_WORDS = (NUM_SENDER_CHANNELS + 3) / 4;
        std::array<uint32_t, ACK_PACKED_WORDS> ack_counters;

        // Completion counters: conditional type based on enable_first_level_ack
        // With first-level acks: scalar per receiver channel
        // Without first-level acks: array per sender channel
        std::conditional_t<
            enable_first_level_ack,
            uint32_t,                                  // Scalar for first-level ack
            std::array<uint32_t, NUM_SENDER_CHANNELS>  // Array for old behavior
            >
            completion_counter;

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
                // Packed mode: use receiver-side stream for completions (matches original working packed
                // implementation)
                sender_channel_packets_completed_stream_ids[i] = to_receiver_packets_sent_streams[0];
                sender_channel_packets_ack_stream_ids[i] = to_sender_packets_acked_streams[0];
            } else {
                // Unpacked mode: each sender uses dedicated per-channel stream
                sender_channel_packets_completed_stream_ids[i] = to_sender_packets_completed_streams[i];
                sender_channel_packets_ack_stream_ids[i] = to_sender_packets_acked_streams[i];
            }
        }
    }

    FORCE_INLINE void send_completion_credit(uint8_t src_id = 0) {
        if constexpr (enable_first_level_ack) {
            // Packed mode: all senders use shared register 0 (src_id ignored)
            remote_update_ptr_val<receiver_txq_id>(sender_channel_packets_completed_stream_ids[0], 1);
        } else {
            WATCHER_RING_BUFFER_PUSH(0xcc200000 | src_id);
            WATCHER_RING_BUFFER_PUSH(sender_channel_packets_completed_stream_ids[src_id]);
            // Unpacked mode: each sender uses dedicated register (sender-channel-based)
            // Matches pre-cdfbd972cde working implementation
            remote_update_ptr_val<receiver_txq_id>(sender_channel_packets_completed_stream_ids[src_id], 1);
        }
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id, int count = 1) {
        uint32_t stream_id = sender_channel_packets_ack_stream_ids[src_id];
        remote_update_ptr_val<receiver_txq_id>(stream_id, count);
    }

    // Send packed ACK credits directly to the shared stream register
    // Used for batch ACK sending when first-level ACK is enabled
    template <size_t VC_NUM_SENDER_CHANNELS, bool wait_for_txq, typename PackedValueType>
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
                if constexpr (wait_for_txq) {
                    while (internal_::eth_txq_is_busy(receiver_txq_id)) {
                    }
                }
                if constexpr (VC_NUM_SENDER_CHANNELS <= 2) {  // TODO: generalize for multi-VC
                    WATCHER_RING_BUFFER_PUSH(0xdeadbeef);
                    remote_update_ptr_val<receiver_txq_id>(stream_id, static_cast<uint32_t>(packed_value.get()));
                } else {
                    WATCHER_RING_BUFFER_PUSH(0xa4000000 | (packed_value.get() & 0xFFFF));
                    remote_update_ptr_val<receiver_txq_id>(stream_id, static_cast<uint32_t>(packed_value.get()));
                    auto stream_id_1 = stream_id + 1;

                    if constexpr (wait_for_txq) {
                        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
                        }
                    }
                    WATCHER_RING_BUFFER_PUSH(0xa4100000 | ((packed_value.get() >> 16) & 0xFFFF));
                    while (internal_::eth_txq_is_busy(receiver_txq_id)) {
                    }
                    remote_update_ptr_val<receiver_txq_id>(stream_id, static_cast<uint32_t>(packed_value.get()));
                }
            } else {
                // uint32_t reg0_val = packed_value.get() & 0x0000FFFF;
                // if constexpr (wait_for_txq) {
                //     while (internal_::eth_txq_is_busy(receiver_txq_id)) {}
                // }
                // remote_update_ptr_val<receiver_txq_id>(stream_id, reg0_val);

                // auto stream_id_1 = stream_id + 1;
                // uint32_t reg1_val = packed_value.get() >> 16;
                // while (internal_::eth_txq_is_busy(receiver_txq_id)) {}
                // remote_update_ptr_val<receiver_txq_id>(stream_id + 1, reg1_val);
                // // Multi-register case - need to write to both stream_ids[0] and stream_ids[1]
                // // For now, assume this is handled elsewhere or not needed
                static_assert(
                    sizeof(typename PackedValueType::storage_type) <= 8,
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

// Type alias for counter-based credit sender using global packing policy
using ReceiverChannelCounterCreditSender =
    ReceiverChannelCounterBasedResponseCreditSender<USE_PACKED_FIRST_LEVEL_ACK_CREDITS>;

using ReceiverChannelResponseCreditSender = typename std::
    conditional_t<multi_txq_enabled, ReceiverChannelCounterCreditSender, ReceiverChannelStreamRegisterCreditSender>;

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

// Implementation for ReceiverChannelCounterCreditSender
template <>
struct init_receiver_channel_response_credit_senders_impl<ReceiverChannelCounterCreditSender> {
    template <uint8_t NUM_RECEIVER_CHANNELS>
    static constexpr auto init() -> std::array<ReceiverChannelCounterCreditSender, NUM_RECEIVER_CHANNELS> {
        std::array<ReceiverChannelCounterCreditSender, NUM_RECEIVER_CHANNELS> credit_senders;
        for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
            credit_senders[i] = ReceiverChannelCounterCreditSender(i);
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
template <bool enable_first_level_ack>
struct SenderChannelFromReceiverCounterBasedCreditsReceiver {
    SenderChannelFromReceiverCounterBasedCreditsReceiver() = default;
    SenderChannelFromReceiverCounterBasedCreditsReceiver(size_t sender_channel_index) :
        acks_received_counter_ptr(reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address)),
        acks_received_and_processed(0) {
        *acks_received_counter_ptr = 0;

        // Initialize completion tracking for unpacked mode
        if constexpr (!enable_first_level_ack) {
            completions_received_counter_ptr =
                reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counters_base_address) +
                sender_channel_index;
            completions_received_and_processed = 0;
        }
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) { acks_received_and_processed += num_acks; }

    // Completion tracking methods (only for unpacked mode where !enable_first_level_ack)
    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        if constexpr (!enable_first_level_ack) {
            return *completions_received_counter_ptr - completions_received_and_processed;
        }
    }

    FORCE_INLINE void increment_num_processed_completions(size_t num_completions) {
        if constexpr (!enable_first_level_ack) {
            completions_received_and_processed += num_completions;
        }
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t sender_channel_index>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        if constexpr (!enable_first_level_ack) {
            return *acks_received_counter_ptr - acks_received_and_processed;
        } else {
            router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();

            // Use safe CreditPacking helper to compute difference without borrows
            // This extracts only the target channel's byte and subtracts with correct wraparound
            using PackingType = tt::tt_fabric::CreditPacking<NUM_SENDER_CHANNELS, 8>;  // 8-bit credits

            uint32_t raw_value = *acks_received_counter_ptr;
            auto raw_packed = PackingType::PackedValueType{raw_value};
            auto processed_packed = PackingType::PackedValueType{acks_received_and_processed};

            // Safe diff: extracts bytes, subtracts with uint8_t wrap
            // Performance: ~3 instructions (2 extractions + 1 subtraction)
            uint8_t diff = PackingType::template diff_channels<sender_channel_index>(raw_packed, processed_packed);

            // Return PACKED value (shifted to channel position) for compatibility with calling code
            // The caller in fabric_erisc_router.cpp:1786 masks with (0xFF << (channel_idx * 8))
            // and expects the value to be at that bit position
            return static_cast<uint32_t>(diff) << (sender_channel_index * 8);
        }
    }

    template <uint8_t sender_channel_index>
    FORCE_INLINE void increment_num_processed_acks(size_t packed_num_acks) {
        if constexpr (USE_PACKED_FIRST_LEVEL_ACK_CREDITS) {
            // Use safe CreditPacking helper to add to single channel without carries
            using PackingType = tt::tt_fabric::CreditPacking<NUM_SENDER_CHANNELS, 8>;  // 8-bit credits

            auto current = PackingType::PackedValueType{acks_received_and_processed};
            auto updated = PackingType::template add_to_channel<sender_channel_index>(
                current,
                static_cast<uint8_t>(packed_num_acks)
            );

            acks_received_and_processed = updated.get();
            // Performance: ~5 instructions (Zbb optimized), fully inlined
        } else {
            // For counter-based (BH), ack counters are 8-bit and wrap at 256
            // Must use uint8_t semantics to match receiver's wraparound behavior
            uint32_t old_val = acks_received_and_processed;
            acks_received_and_processed = static_cast<uint32_t>(
                static_cast<uint8_t>(acks_received_and_processed) + static_cast<uint8_t>(packed_num_acks)
            );
        }
    }

    // Completion methods removed - now in OutboundReceiverChannelPointers

    volatile uint32_t* acks_received_counter_ptr;
    uint32_t acks_received_and_processed = 0;

    // Conditionally include completion tracking (only for unpacked mode where !enable_first_level_ack)
    // When enable_first_level_ack=true, completions are tracked in OutboundReceiverChannelPointers instead
    using CompletionPtrType = std::conditional_t<!enable_first_level_ack, volatile uint32_t*, std::monostate>;
    using CompletionCounterType = std::conditional_t<!enable_first_level_ack, uint32_t, std::monostate>;

    [[no_unique_address]] CompletionPtrType completions_received_counter_ptr;
    [[no_unique_address]] CompletionCounterType completions_received_and_processed;
};

template <bool enable_first_level_ack>
struct SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver {
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver() = default;

    // Packing behavior: when enable_first_level_ack is true, all sender channels use
    // register 0 for packed credits. Otherwise, each uses its own register.
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver(size_t sender_channel_index) {
        // For unpacked mode: members initialized in initializer list above
        // For packed mode: reassign to use shared registers
        if constexpr (enable_first_level_ack) {
            to_sender_packets_acked_streams =
                std::array<size_t, 2>{::to_sender_packets_acked_streams[0], ::to_sender_packets_acked_streams[1]};
            to_sender_packets_completed_stream = ::to_receiver_packets_sent_streams[0];
        } else {
            to_sender_packets_acked_streams = ::to_sender_packets_acked_streams[sender_channel_index];
            to_sender_packets_completed_stream = ::to_sender_packets_completed_streams[sender_channel_index];
        }
    }

    // returns the packed value for the channel, in isolation, but in the "packed" location of the register
    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t sender_channel_index>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        if constexpr (!enable_first_level_ack) {
            // Unpacked mode: register value is for this channel only
            return get_ptr_val(to_sender_packets_acked_streams);
        } else {
            // Packed mode: extract this channel's credits (8-bit per channel)
            // Channel 0 or 1: read from first register
            // Channel 2+: read from second register
            // In reg1, channels are at offsets (ch_id - CHANNELS_IN_REG0) * CREDIT_WIDTH
            constexpr uint8_t CREDIT_WIDTH = 8;
            constexpr uint8_t CHANNELS_IN_REG0 = 2;
            constexpr size_t stream_reg_index = sender_channel_index < CHANNELS_IN_REG0 ? 0 : 1;
            constexpr size_t slot_in_reg = sender_channel_index < CHANNELS_IN_REG0 ? sender_channel_index : sender_channel_index - CHANNELS_IN_REG0;
            constexpr size_t shift_amount = slot_in_reg * CREDIT_WIDTH;
            constexpr size_t mask = 0xff << shift_amount;

            uint32_t reg_val = get_ptr_val(to_sender_packets_acked_streams[stream_reg_index]);
            uint32_t channel_acks = reg_val & mask;
            return static_cast<uint32_t>(channel_acks);
        }
    }

    template <uint8_t sender_channel_index>
    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) {
        if constexpr (!enable_first_level_ack) {
            // Unpacked mode: simple decrement
            increment_local_update_ptr_val(to_sender_packets_acked_streams, -num_acks);
        } else {
            // Packed mode: pack decrement value into this channel's position and write to correct register
            constexpr uint8_t CREDIT_WIDTH = 8;
            constexpr uint8_t CHANNELS_IN_REG0 = 2;
            constexpr size_t stream_reg_index = sender_channel_index < CHANNELS_IN_REG0 ? 0 : 1;
            constexpr size_t slot_in_reg = sender_channel_index < CHANNELS_IN_REG0 ? sender_channel_index : sender_channel_index - CHANNELS_IN_REG0;
            constexpr size_t shift_amount = slot_in_reg * CREDIT_WIDTH;

            uint32_t packed_decrement = static_cast<uint32_t>(num_acks) << (shift_amount);
            increment_local_update_ptr_val(
                to_sender_packets_acked_streams[stream_reg_index], -static_cast<int32_t>(packed_decrement));
        }
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        uint32_t completions = get_ptr_val(this->to_sender_packets_completed_stream);
        return completions;
    }

    FORCE_INLINE void increment_num_processed_completions(size_t num_completions) {
        increment_local_update_ptr_val(to_sender_packets_completed_stream, -num_completions);
    }

    // For packed mode with >3 channels, need array to access both registers
    // For unpacked mode, single register is sufficient
    std::conditional_t<
        enable_first_level_ack,
        std::array<size_t, 2>,  // Packed: array for multi-register access
        size_t>
        to_sender_packets_acked_streams;  // Note: plural for array case

    size_t to_sender_packets_completed_stream;  // Completions always unpacked
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

// Type alias for BH counter-based credits receiver
using SenderChannelFromReceiverCounterCreditsReceiver =
    SenderChannelFromReceiverCounterBasedCreditsReceiver<USE_PACKED_FIRST_LEVEL_ACK_CREDITS>;

// Implementation for SenderChannelFromReceiverCounterCreditsReceiver
template <>
struct init_sender_channel_from_receiver_credits_flow_controllers_impl<
    SenderChannelFromReceiverCounterCreditsReceiver> {
    template <uint8_t NUM_SENDER_CHANNELS>
    static constexpr auto init() -> std::array<SenderChannelFromReceiverCounterCreditsReceiver, NUM_SENDER_CHANNELS> {
        std::array<SenderChannelFromReceiverCounterCreditsReceiver, NUM_SENDER_CHANNELS> flow_controllers;
        for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
            // Pass sender channel index (needed for unpacked mode)
            new (&flow_controllers[i]) SenderChannelFromReceiverCounterCreditsReceiver(i);
        }
        return flow_controllers;
    }
};

using SenderChannelFromReceiverCredits = typename std::conditional_t<
    multi_txq_enabled,
    SenderChannelFromReceiverCounterCreditsReceiver,
    SenderChannelFromReceiverStreamRegisterCreditsReceiver>;

// SFINAE-based overload for !multi_txq_enabled case (WH with stream registers)
template <uint8_t NUM_SENDER_CHANNELS>
constexpr FORCE_INLINE auto init_sender_channel_from_receiver_credits_flow_controllers()
    -> std::enable_if_t<!multi_txq_enabled, std::array<SenderChannelFromReceiverCredits, NUM_SENDER_CHANNELS>> {
    return init_sender_channel_from_receiver_credits_flow_controllers_impl<
        SenderChannelFromReceiverStreamRegisterCreditsReceiver>::template init<NUM_SENDER_CHANNELS>();
}

// SFINAE-based overload for multi_txq_enabled case (BH)
template <uint8_t NUM_SENDER_CHANNELS>
constexpr FORCE_INLINE auto init_sender_channel_from_receiver_credits_flow_controllers()
    -> std::enable_if_t<multi_txq_enabled, std::array<SenderChannelFromReceiverCredits, NUM_SENDER_CHANNELS>> {
    return init_sender_channel_from_receiver_credits_flow_controllers_impl<
        SenderChannelFromReceiverCounterCreditsReceiver>::template init<NUM_SENDER_CHANNELS>();
}

template <bool CHECK_BUSY>
FORCE_INLINE void receiver_send_completion_ack(
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender, uint8_t src_id) {
    if constexpr (CHECK_BUSY) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        };
    }
    receiver_channel_response_credit_sender.send_completion_credit(src_id);
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
FORCE_INLINE uint8_t extract_sender_channel_acks(uint32_t packed_acks) {
    if constexpr (USE_PACKED_FIRST_LEVEL_ACK_CREDITS) {
        if constexpr (multi_txq_enabled) {
            constexpr uint32_t shift = sender_channel_index * 8;
            return (packed_acks >> shift) & 0xFF;
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
    if constexpr (USE_PACKED_PACKET_SENT_CREDITS && receiver_credits_config::NEEDS_MULTI_REGISTER) {
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
        constexpr auto packed = Packing::template pack_channel<sender_channel_index>(1);

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
