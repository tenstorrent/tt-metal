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

// ============================================================================
// Receiver Packet Credits - Configuration (must be defined early for ack structs)
// ============================================================================
namespace receiver_credits_config {
constexpr uint32_t MIN_BITS = tt::tt_fabric::log2_ceil(tt::tt_fabric::MAX_SENDER_BUFFER_SLOTS + 1);

// For ≤2 sender channels, 8-bit packing fits in a single 24-bit register (2×8=16 ≤ 24)
// and enables fast byte-aligned operations.
// For >2 sender channels, use MIN_BITS packing to fit more channels per register:
//   - 4 channels × 6-bit = 24 bits → single register (was 2 registers with 8-bit)
//   - 5 channels × 6-bit = 30 bits → 2 registers with 4+1 layout
constexpr bool credits_are_byte_aligned = (NUM_SENDER_CHANNELS <= 2);
constexpr uint8_t CREDIT_WIDTH = credits_are_byte_aligned ? 8 : MIN_BITS;
constexpr uint32_t CREDIT_MASK = (1u << CREDIT_WIDTH) - 1;
constexpr uint32_t TOTAL_BITS = NUM_SENDER_CHANNELS * CREDIT_WIDTH;
constexpr bool NEEDS_MULTI_REGISTER = TOTAL_BITS > 24;
constexpr size_t PACKED_WORDS_COUNT = NEEDS_MULTI_REGISTER ? 2 : 1;

// L1 packing mirrors overlay register format (same CREDIT_WIDTH)
constexpr size_t CREDITS_PER_L1_WORD = 32 / CREDIT_WIDTH;
}  // namespace receiver_credits_config

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

    template <bool SEND_CREDITS_DURING_CALL>
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
        if constexpr (SEND_CREDITS_DURING_CALL) {
            update_sender_side_credits();
        }
    }

        // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
        FORCE_INLINE void send_ack_credit(uint8_t src_id, int packed_count = 1) {
            if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
                static_assert(NUM_SENDER_CHANNELS <= 8, "NUM_SENDER_CHANNELS must be less than or equal to 8");

                constexpr uint8_t CW = receiver_credits_config::CREDIT_WIDTH;
                constexpr uint32_t CM = receiver_credits_config::CREDIT_MASK;
                constexpr size_t CREDITS_PER_WORD = receiver_credits_config::CREDITS_PER_L1_WORD;

                if constexpr (NUM_SENDER_CHANNELS <= CREDITS_PER_WORD) {
                    // All channels fit in single word
                    uint32_t shift = src_id * CW;
                    uint32_t old_val = (ack_counters[0] >> shift) & CM;
                    uint32_t new_val = (old_val + packed_count) & CM;
                    ack_counters[0] = (ack_counters[0] & ~(CM << shift)) | (new_val << shift);
                    ack_counters_base_l1_ptr[0] = ack_counters[0];
                } else {
                    // Multi-word: select correct word for this channel
                    size_t word_index = src_id / CREDITS_PER_WORD;
                    size_t channel_in_word = src_id % CREDITS_PER_WORD;
                    uint32_t shift = channel_in_word * CW;
                    uint32_t old_val = (ack_counters[word_index] >> shift) & CM;
                    uint32_t new_val = (old_val + packed_count) & CM;
                    ack_counters[word_index] = (ack_counters[word_index] & ~(CM << shift)) | (new_val << shift);
                    ack_counters_base_l1_ptr[word_index] = ack_counters[word_index];
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
        template <
            size_t VC_NUM_SENDER_CHANNELS,
            bool wait_for_txq,
            bool SEND_CREDITS_DURING_CALL,
            typename PackedValueType>
        FORCE_INLINE void send_packed_ack_credits(const PackedValueType& packed_value) {
            if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
                constexpr uint8_t CW = receiver_credits_config::CREDIT_WIDTH;
                constexpr size_t CREDITS_PER_WORD = receiver_credits_config::CREDITS_PER_L1_WORD;

                // SIMD-in-register addition: add packed delta to packed accumulator without
                // cross-channel carries. XOR-MSB trick works for any field width where
                // delta per channel < 2^(CW-1).
                // MSB_MASK has bit (CW-1) set in each channel's field.
                auto sum_packed_word_credits = [](uint32_t current_val, uint32_t delta_val) -> uint32_t {
                    if constexpr (VC_NUM_SENDER_CHANNELS == 1) {
                        return (current_val + delta_val) & receiver_credits_config::CREDIT_MASK;
                    } else {
                        constexpr uint32_t MSB_MASK =
                            tt::tt_fabric::construct_packed_credit_sum_mask<VC_NUM_SENDER_CHANNELS, CW>() << (CW - 1);
                        return ((current_val & ~MSB_MASK) + delta_val) ^ (current_val & MSB_MASK);
                    }
                };

                if constexpr (VC_NUM_SENDER_CHANNELS <= CREDITS_PER_WORD) {
                    // All channels fit in single word
                    uint32_t result =
                        sum_packed_word_credits(ack_counters[0], static_cast<uint32_t>(packed_value.get()));
                    ack_counters[0] = result;
                    ack_counters_base_l1_ptr[0] = result;
                } else {
                    // Multi-word: split at CREDITS_PER_WORD boundary
                    constexpr uint32_t lower_bits = CREDITS_PER_WORD * CW;
                    constexpr uint32_t lower_mask = static_cast<uint32_t>((1ull << lower_bits) - 1);

                    uint64_t packed_val = packed_value.get();

                    // Lower word
                    uint32_t result_lower =
                        sum_packed_word_credits(ack_counters[0], static_cast<uint32_t>(packed_val & lower_mask));
                    ack_counters[0] = result_lower;
                    ack_counters_base_l1_ptr[0] = result_lower;

                    // Upper word: remaining channels
                    uint32_t current_upper = ack_counters[1];
                    uint32_t delta_upper = static_cast<uint32_t>(packed_val >> lower_bits);
                    // For single remaining channel, simple masked add
                    uint32_t result_upper = (current_upper + delta_upper) & receiver_credits_config::CREDIT_MASK;
                    ack_counters[1] = result_upper;
                    ack_counters_base_l1_ptr[1] = result_upper;
                }

                if constexpr (SEND_CREDITS_DURING_CALL) {
                    if constexpr (wait_for_txq) {
                        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
                        }
                    }
                    update_sender_side_credits();
                }
            }
        }

        volatile tt_l1_ptr uint32_t* completion_counter_l1_ptr;
        volatile tt_l1_ptr uint32_t* ack_counters_base_l1_ptr;

        // Ack counters: packed storage (CREDITS_PER_L1_WORD credits per word)
        // With 6-bit packing: 5 per word (1 word for ≤5 channels)
        // With 8-bit packing: 4 per word (1 word for ≤4 channels)
        static constexpr size_t ACK_PACKED_WORDS =
            (NUM_SENDER_CHANNELS + receiver_credits_config::CREDITS_PER_L1_WORD - 1) /
            receiver_credits_config::CREDITS_PER_L1_WORD;
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

    template <bool SEND_CREDITS_DURING_CALL>
    FORCE_INLINE void send_completion_credit(uint8_t src_id = 0) {
        if constexpr (enable_first_level_ack) {
            // Packed mode: all senders use shared register 0 (src_id ignored)
            if constexpr (SEND_CREDITS_DURING_CALL) {
                remote_update_ptr_val<receiver_txq_id>(sender_channel_packets_completed_stream_ids[0], 1);
            }
        } else {
            WATCHER_RING_BUFFER_PUSH(0xcc200000 | src_id);
            WATCHER_RING_BUFFER_PUSH(sender_channel_packets_completed_stream_ids[src_id]);
            // Unpacked mode: each sender uses dedicated register (sender-channel-based)
            // Matches pre-cdfbd972cde working implementation
            if constexpr (SEND_CREDITS_DURING_CALL) {
                remote_update_ptr_val<receiver_txq_id>(sender_channel_packets_completed_stream_ids[src_id], 1);
            }
        }
    }

    // Assumes !eth_txq_is_busy() -- PLEASE CHECK BEFORE CALLING
    FORCE_INLINE void send_ack_credit(uint8_t src_id, int count = 1) {
        uint32_t stream_id = sender_channel_packets_ack_stream_ids[src_id];
        remote_update_ptr_val<receiver_txq_id>(stream_id, count);
    }

    // Send packed ACK credits directly to the shared stream register
    // Used for batch ACK sending when first-level ACK is enabled
    template <size_t VC_NUM_SENDER_CHANNELS, bool wait_for_txq, bool SEND_CREDITS_DURING_CALL, typename PackedValueType>
    FORCE_INLINE void send_packed_ack_credits(const PackedValueType& packed_value) {
        static_assert(SEND_CREDITS_DURING_CALL, "SEND_CREDITS_DURING_CALL must be true");
        if constexpr (enable_first_level_ack) {
            // All channels use register 0 when enable_first_level_ack is true
            uint32_t stream_id = sender_channel_packets_ack_stream_ids[0];
            // packed_value format already matches what sender expects - just write it
            // For ≤32 bits (≤4 channels), write as uint32_t
            // For >32 bits (5 channels), this needs multi-register handling (TODO if needed)
            static_assert(sizeof(typename PackedValueType::storage_type) <= 8,
                         "Packed value storage too large");
            if constexpr (sizeof(typename PackedValueType::storage_type) <= 4) {
                constexpr uint8_t ACK_CHANNELS_PER_REG = MAX_ACK_CREDITS_PER_OVERLAY_REGISTER;
                if constexpr (wait_for_txq) {
                    while (internal_::eth_txq_is_busy(receiver_txq_id)) {
                    }
                }
                if constexpr (VC_NUM_SENDER_CHANNELS <= ACK_CHANNELS_PER_REG) {
                    // All channels fit in single register
                    WATCHER_RING_BUFFER_PUSH(0xdeadbeef);
                    remote_update_ptr_val<receiver_txq_id>(stream_id, static_cast<uint32_t>(packed_value.get()));
                } else {
                    // Multi-register: split at ACK_CHANNELS_PER_REG boundary
                    constexpr uint32_t reg0_bits = PackedValueType::CREDIT_WIDTH * ACK_CHANNELS_PER_REG;
                    constexpr uint32_t reg0_mask = (1u << reg0_bits) - 1;
                    WATCHER_RING_BUFFER_PUSH(0xa4000000 | (packed_value.get() & 0xFFFF));
                    remote_update_ptr_val<receiver_txq_id>(
                        stream_id, static_cast<uint32_t>(packed_value.get() & reg0_mask));
                    auto stream_id_1 = stream_id + 1;

                    if constexpr (wait_for_txq) {
                        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
                        }
                    }
                    remote_update_ptr_val<receiver_txq_id>(
                        stream_id_1, static_cast<uint32_t>(packed_value.get() >> reg0_bits));
                }
            } else {
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
// Members container for SenderChannelFromReceiverCounterBasedCreditsReceiver
// Packed mode (enable_first_level_ack=true): only ack tracking
template <bool enable_first_level_ack>
struct SenderChannelFromReceiverCounterBasedCreditsReceiverMembers {
    volatile uint32_t* acks_received_counter_ptr;
    uint32_t acks_received_and_processed;
    // No completion tracking in packed mode
};

// Unpacked mode (enable_first_level_ack=false): ack tracking + completion tracking
template <>
struct SenderChannelFromReceiverCounterBasedCreditsReceiverMembers<false> {
    volatile uint32_t* acks_received_counter_ptr;
    uint32_t acks_received_and_processed;
    volatile uint32_t* completions_received_counter_ptr;
    uint32_t completions_received_and_processed;
};

template <bool enable_first_level_ack>
struct SenderChannelFromReceiverCounterBasedCreditsReceiver {
    SenderChannelFromReceiverCounterBasedCreditsReceiverMembers<enable_first_level_ack> m;

    SenderChannelFromReceiverCounterBasedCreditsReceiver() = default;
    SenderChannelFromReceiverCounterBasedCreditsReceiver(size_t sender_channel_index) {
        m.acks_received_counter_ptr = reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address);
        m.acks_received_and_processed = 0;
        *m.acks_received_counter_ptr = 0;

        // Initialize completion tracking for unpacked mode
        if constexpr (!enable_first_level_ack) {
            m.completions_received_counter_ptr =
                reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counters_base_address) +
                sender_channel_index;
            m.completions_received_and_processed = 0;
        }
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) { m.acks_received_and_processed += num_acks; }

    // Completion tracking methods (only for unpacked mode where !enable_first_level_ack)
    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_completions_from_receiver() {
        if constexpr (!enable_first_level_ack) {
            router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
            return *m.completions_received_counter_ptr - m.completions_received_and_processed;
        }
        return 0;  // Won't be called when enable_first_level_ack=true, but satisfies compiler
    }

    FORCE_INLINE void increment_num_processed_completions(size_t num_completions) {
        if constexpr (!enable_first_level_ack) {
            m.completions_received_and_processed += num_completions;
        }
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t sender_channel_index>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        if constexpr (!enable_first_level_ack) {
            return *m.acks_received_counter_ptr - m.acks_received_and_processed;
        } else {
            router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();

            // Determine VC size based on sender channel index
            // Sender channels 0..ACTUAL_VC0_SENDER_CHANNELS-1 belong to VC0
            // Remaining channels belong to VC1
            constexpr size_t VC_SENDER_CHANNELS = (sender_channel_index < ACTUAL_VC0_SENDER_CHANNELS)
                                                      ? ACTUAL_VC0_SENDER_CHANNELS
                                                      : ACTUAL_VC1_SENDER_CHANNELS;

            constexpr uint8_t CW = receiver_credits_config::CREDIT_WIDTH;
            // Use safe CreditPacking helper to compute difference without borrows
            using PackingType = tt::tt_fabric::CreditPacking<VC_SENDER_CHANNELS, CW>;

            uint32_t raw_value = *m.acks_received_counter_ptr;
            auto raw_packed = typename PackingType::PackedValueType{raw_value};
            auto processed_packed = typename PackingType::PackedValueType{m.acks_received_and_processed};

            // Adjust sender_channel_index to be relative to its VC (0-based within the VC)
            constexpr size_t vc_relative_index = (sender_channel_index < ACTUAL_VC0_SENDER_CHANNELS)
                                                     ? sender_channel_index
                                                     : (sender_channel_index - ACTUAL_VC0_SENDER_CHANNELS);

            // Safe diff: extracts channel, subtracts with wraparound
            uint8_t diff = PackingType::template diff_channels<vc_relative_index>(raw_packed, processed_packed);

            // Return PACKED value (shifted to channel position) for compatibility with calling code
            return static_cast<uint32_t>(diff) << (sender_channel_index * CW);
        }
    }

    template <uint8_t sender_channel_index>
    FORCE_INLINE void increment_num_processed_acks(size_t packed_num_acks) {
        if constexpr (USE_PACKED_FIRST_LEVEL_ACK_CREDITS) {
            // Determine VC size based on sender channel index
            constexpr size_t VC_SENDER_CHANNELS = (sender_channel_index < ACTUAL_VC0_SENDER_CHANNELS)
                                                      ? ACTUAL_VC0_SENDER_CHANNELS
                                                      : ACTUAL_VC1_SENDER_CHANNELS;

            // Adjust sender_channel_index to be relative to its VC
            constexpr size_t vc_relative_index = (sender_channel_index < ACTUAL_VC0_SENDER_CHANNELS)
                                                     ? sender_channel_index
                                                     : (sender_channel_index - ACTUAL_VC0_SENDER_CHANNELS);

            constexpr uint8_t CW = receiver_credits_config::CREDIT_WIDTH;
            // Use safe CreditPacking helper to add to single channel without carries
            using PackingType = tt::tt_fabric::CreditPacking<VC_SENDER_CHANNELS, CW>;

            auto current = typename PackingType::PackedValueType{m.acks_received_and_processed};
            auto updated =
                PackingType::template add_to_channel<vc_relative_index>(current, static_cast<uint8_t>(packed_num_acks));

            m.acks_received_and_processed = updated.get();
        } else {
            // Ack counters wrap at 2^CREDIT_WIDTH
            constexpr uint32_t CM = receiver_credits_config::CREDIT_MASK;
            m.acks_received_and_processed =
                static_cast<uint32_t>((m.acks_received_and_processed + packed_num_acks) & CM);
        }
    }
};

template <bool enable_first_level_ack>
struct SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver {
    static constexpr uint8_t CREDIT_WIDTH = receiver_credits_config::CREDIT_WIDTH;
    static constexpr uint8_t CHANNELS_IN_REG0 = MAX_ACK_CREDITS_PER_OVERLAY_REGISTER;
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

    template <uint8_t sender_channel_index>
    static constexpr uint32_t get_credit_slot_in_reg() {
        constexpr size_t slot_in_reg =
            sender_channel_index < CHANNELS_IN_REG0 ? sender_channel_index : sender_channel_index - CHANNELS_IN_REG0;
        return slot_in_reg;
    }

    template <uint8_t sender_channel_index>
    static constexpr uint32_t get_stream_reg_index() {
        constexpr size_t stream_reg_index = sender_channel_index < CHANNELS_IN_REG0 ? 0 : 1;
        return stream_reg_index;
    }

    template <uint8_t sender_channel_index>
    static constexpr uint32_t get_in_reg_shift_amount() {
        constexpr size_t shift_amount = get_credit_slot_in_reg<sender_channel_index>() * CREDIT_WIDTH;
        return shift_amount;
    }

    // returns the packed value for the channel, in isolation, but in the "packed" location of the register
    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t sender_channel_index>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        if constexpr (!enable_first_level_ack) {
            // Unpacked mode: register value is for this channel only
            return get_ptr_val(to_sender_packets_acked_streams);
        } else {
            // Packed mode: extract this channel's credits
            // In reg1, channels are at offsets (ch_id - CHANNELS_IN_REG0) * CREDIT_WIDTH
            constexpr size_t shift_amount = get_in_reg_shift_amount<sender_channel_index>();
            constexpr size_t mask = static_cast<size_t>(receiver_credits_config::CREDIT_MASK) << shift_amount;

            uint32_t reg_val =
                get_ptr_val(to_sender_packets_acked_streams[get_stream_reg_index<sender_channel_index>()]);
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
            constexpr size_t shift_amount = get_in_reg_shift_amount<sender_channel_index>();

            uint32_t packed_decrement = static_cast<uint32_t>(num_acks) << (shift_amount);
            increment_local_update_ptr_val(
                to_sender_packets_acked_streams[get_stream_reg_index<sender_channel_index>()],
                -static_cast<int32_t>(packed_decrement));
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

template <bool CHECK_BUSY, bool SEND_CREDITS_DURING_CALL>
FORCE_INLINE void receiver_send_completion_ack(
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender, uint8_t src_id) {
    if constexpr (CHECK_BUSY) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        };
    }
    receiver_channel_response_credit_sender.template send_completion_credit<SEND_CREDITS_DURING_CALL>(src_id);
}

// MUST CHECK !is_eth_txq_busy() before calling
template <bool CHECK_BUSY, bool SEND_CREDITS_DURING_CALL>
FORCE_INLINE void receiver_send_completion_ack(
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender) {
    if constexpr (CHECK_BUSY) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        };
    }
    receiver_channel_response_credit_sender.template send_completion_credit<SEND_CREDITS_DURING_CALL>();
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
        constexpr uint8_t CW = receiver_credits_config::CREDIT_WIDTH;
        constexpr uint32_t shift = sender_channel_index * CW;
        return (packed_acks >> shift) & receiver_credits_config::CREDIT_MASK;
    } else {
        // !enable_first_level_ack: Counter-based, already unpacked (one counter per channel)
        return packed_acks;
    }
}

/**
 * Build ACK decrement value for sender channel.
 * Pack into register/L1 position. Direct value when no packing.
 */
template <uint8_t sender_channel_index>
FORCE_INLINE uint32_t build_ack_decrement_value(uint32_t acks_count) {
    if constexpr (USE_PACKED_FIRST_LEVEL_ACK_CREDITS) {
        constexpr uint8_t CW = receiver_credits_config::CREDIT_WIDTH;
        return acks_count << (sender_channel_index * CW);
    } else {
        // !enable_first_level_ack: Direct value, no packing
        return acks_count;
    }
}

/**
 * Determine which stream register ID to write to for a given sender channel.
 * In multi-register cases (5+ channels):
 * - Channels 0-3 → reg0 (base stream ID)
 * - Channels 4+  → reg1 (base stream ID + 1)
 * In single-register cases (1-4 channels): always base stream ID
 */
template <uint8_t sender_channel_index, uint32_t base_stream_id>
constexpr uint32_t get_sender_target_stream_id() {
    if constexpr (USE_PACKED_PACKET_SENT_CREDITS && receiver_credits_config::NEEDS_MULTI_REGISTER) {
        constexpr uint8_t CHANNELS_IN_REG0 = MAX_PACKETS_RECEIVED_CREDITS_PER_OVERLAY_REGISTER;
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
 * sender_channel_index (0-4). In multi-register cases (5+ channels), this returns
 * only the portion relevant to the register this channel maps to:
 * - Channels 0-3 → write to reg0 (to_receiver_pkts_sent_id)
 * - Channels 4+  → write to reg1 (to_receiver_pkts_sent_id + 1)
 */
template <uint8_t sender_channel_index, uint32_t to_receiver_pkts_sent_id>
FORCE_INLINE constexpr uint32_t build_packet_forward_value() {
    if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
        // Determine VC size based on sender channel index
        constexpr size_t VC_SENDER_CHANNELS = (sender_channel_index < ACTUAL_VC0_SENDER_CHANNELS)
                                                  ? ACTUAL_VC0_SENDER_CHANNELS
                                                  : ACTUAL_VC1_SENDER_CHANNELS;
        constexpr size_t vc_relative_index = (sender_channel_index < ACTUAL_VC0_SENDER_CHANNELS)
                                                 ? sender_channel_index
                                                 : (sender_channel_index - ACTUAL_VC0_SENDER_CHANNELS);

        // WH: Pack 1 credit into this channel's position using generic credit packing
        using Packing = tt::tt_fabric::CreditPacking<VC_SENDER_CHANNELS, receiver_credits_config::CREDIT_WIDTH>;
        static_assert(sender_channel_index < NUM_SENDER_CHANNELS, "Sender channel index out of bounds");

        // Pack value of 1 at the bit offset for this channel (relative to its VC)
        constexpr auto packed = Packing::template pack_channel<vc_relative_index>(1);

        // Recompute NEEDS_MULTI_REGISTER for this specific VC
        constexpr bool VC_NEEDS_MULTI_REGISTER = (VC_SENDER_CHANNELS * receiver_credits_config::CREDIT_WIDTH) > 24;

        // For multi-register case, extract only the relevant register's portion
        if constexpr (VC_NEEDS_MULTI_REGISTER) {
            // Multi-register layout: reg0 has channels 0-1, reg1 has channels 2-4
            constexpr uint8_t CHANNELS_IN_REG0 = MAX_PACKETS_RECEIVED_CREDITS_PER_OVERLAY_REGISTER;
            constexpr uint32_t reg1_shift = CHANNELS_IN_REG0 * receiver_credits_config::CREDIT_WIDTH;

            // Use VC-relative index to determine which register within this VC's register pair
            if constexpr (vc_relative_index < CHANNELS_IN_REG0) {
                // Channels 0-1 (VC-relative): extract lower 16 bits for reg0
                return static_cast<uint32_t>(packed.get() & ((1ULL << reg1_shift) - 1));
            } else {
                // Channels 2-4 (VC-relative): extract and shift down upper bits for reg1
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
 * Returns the number of sender channels visible to a specific receiver channel.
 * Uses actual configured channel counts from compile-time args (populated by builder).
 *
 * Topology-dependent mapping:
 * - Receiver channel 0 (VC0): ACTUAL_VC0_SENDER_CHANNELS sender channels
 * - Receiver channel 1 (VC1): ACTUAL_VC1_SENDER_CHANNELS sender channels
 *
 * Each receiver channel tracks credits only for its own subset of sender channels,
 * not the global NUM_SENDER_CHANNELS total.
 */
constexpr size_t get_num_sender_channels_for_receiver(size_t receiver_channel) {
    return (receiver_channel == 0) ? ACTUAL_VC0_SENDER_CHANNELS : ACTUAL_VC1_SENDER_CHANNELS;
}

/**
 * Determine if a receiver channel needs multi-register storage (>24 bits total).
 * This must be computed per receiver channel, not globally, because each receiver
 * channel tracks a different number of sender channels.
 */
constexpr bool receiver_channel_needs_multi_register(size_t receiver_channel) {
    return (get_num_sender_channels_for_receiver(receiver_channel) * receiver_credits_config::CREDIT_WIDTH) > 24;
}

/**
 * Template helper to resolve receiver packet credit view type based on receiver channel
 * and stream ID(s). Automatically selects single or multi-register storage based on
 * the number of sender channels visible to this specific receiver channel.
 *
 * Usage:
 *   using MyCredits = ReceiverPacketCreditsViewFor<receiver_channel, my_stream_id>;
 *   MyCredits credits;
 *
 * @tparam receiver_channel Which receiver channel (VC): 0 for VC0, 1 for VC1
 * @tparam stream_id_0 Primary stream register ID
 * @tparam stream_id_1 Secondary stream register ID (for multi-register case)
 */
template <size_t receiver_channel, size_t stream_id_0, size_t stream_id_1 = stream_id_0 + 1>
using ReceiverPacketCreditsViewFor = std::conditional_t<
    receiver_channel_needs_multi_register(receiver_channel),
    // Multi-register case (>24 bits): 5+ channels
    tt::tt_fabric::MultiOverlayRegCreditView<
        get_num_sender_channels_for_receiver(receiver_channel),
        receiver_credits_config::CREDIT_WIDTH,
        stream_id_0,
        stream_id_1>,
    // Single-register case (≤24 bits): 1-3 channels, or 4 channels with tight packing
    tt::tt_fabric::OverlayRegCreditView<
        get_num_sender_channels_for_receiver(receiver_channel),
        receiver_credits_config::CREDIT_WIDTH,
        stream_id_0>>;

/**
 * Template helper to resolve receiver packet credit updater type based on receiver channel
 * and stream ID(s). Provides write operations including decrement_packed() for clearing credits.
 *
 * Usage:
 *   using MyCreditsUpdater = ReceiverPacketCreditsUpdaterFor<receiver_channel, my_stream_id>;
 *   MyCreditsUpdater updater;
 *   updater.decrement_packed(value);
 *
 * @tparam receiver_channel Which receiver channel (VC): 0 for VC0, 1 for VC1
 * @tparam stream_id_0 Primary stream register ID
 * @tparam stream_id_1 Secondary stream register ID (for multi-register case)
 */
template <size_t receiver_channel, size_t stream_id_0, size_t stream_id_1 = stream_id_0 + 1>
using ReceiverPacketCreditsUpdaterFor = std::conditional_t<
    receiver_channel_needs_multi_register(receiver_channel),
    // Multi-register case (>24 bits): 5+ channels
    tt::tt_fabric::MultiOverlayRegCreditUpdater<
        get_num_sender_channels_for_receiver(receiver_channel),
        receiver_credits_config::CREDIT_WIDTH,
        stream_id_0,
        stream_id_1>,
    // Single-register case (≤24 bits): 1-3 channels, or 4 channels with tight packing
    tt::tt_fabric::OverlayRegCreditUpdater<
        get_num_sender_channels_for_receiver(receiver_channel),
        receiver_credits_config::CREDIT_WIDTH,
        stream_id_0>>;

/**
//  * Accumulate receiver channel credits from packed value.
//  * WH: Unpack and sum all channels. BH: Direct value (no packing).
//  */
// FORCE_INLINE uint32_t accumulate_receiver_channel_credits(uint32_t packed_value) {
//     if constexpr (USE_PACKED_PACKET_SENT_CREDITS) {
//         // WH: Unpack and sum across all sender channels using generic packing layer
//         using Packing = tt::tt_fabric::CreditPacking<NUM_SENDER_CHANNELS, receiver_credits_config::CREDIT_WIDTH>;
//         typename Packing::PackedValueType packed{packed_value};
//         return Packing::sum_all_channels(packed);
//     } else {
//         // BH or !enable_first_level_ack: Already unpacked (direct count)
//         return packed_value;
//     }
// }
