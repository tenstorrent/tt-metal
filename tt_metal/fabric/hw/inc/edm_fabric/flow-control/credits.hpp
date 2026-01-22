// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/risc_attribs.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/named_types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#include <cstdint>

#ifndef STREAM_REG_CFG_DATA_WIDTH
#define STREAM_REG_CFG_DATA_WIDTH 24
#endif

namespace tt::tt_fabric {

// Compile-time log2 ceiling
constexpr uint32_t log2_ceil(uint32_t n) {
    if (n <= 1) {
        return 0;
    }
    uint32_t log = 0;
    uint32_t value = n - 1;
    while (value > 0) {
        value >>= 1;
        log++;
    }
    return log;
}

template <uint8_t NUM_CHANNELS_IN_THIS_VC, uint32_t MAX_BUFFER_SLOTS>
using PackedCreditValue = NamedType<uint32_t, struct PackedCreditValueType>;
/**
 * Packed credits for multiple sender channels.
 * Automatically determines credit width and packing based on channel count and buffer slots.
 *
 * Template params:
 * - NUM_CHANNELS: Number of sender channels (up to 4 per VC)
 * - MAX_BUFFER_SLOTS: Max buffer slot count (determines credit bit width)
 * - base_stream_id: The base stream ID to use for the packet credits pool, assumes we have NUM_REGISTERS of contig regs
 * available
 */
template <uint8_t NUM_CHANNELS_IN_THIS_VC, uint32_t MAX_BUFFER_SLOTS, size_t stream_id>
struct PackedCredits {
    // Calculate credit bit width (byte-aligned when possible)
    static constexpr uint32_t MAX_SLOTS_PER_CHANNEL = 64;  // let's us pack 4 credits per register
    static constexpr bool credits_are_byte_aligned = NUM_CHANNELS_IN_THIS_VC <= 3;
    static constexpr uint32_t MIN_BITS = log2_ceil(MAX_BUFFER_SLOTS + 1);
    static constexpr uint32_t CREDIT_WIDTH = credits_are_byte_aligned ? 8 : MIN_BITS;
    static constexpr uint32_t TWO_CREDIT_WIDTHS = 2 * CREDIT_WIDTH;
    static_assert(MAX_BUFFER_SLOTS < (1u << CREDIT_WIDTH), "MAX_BUFFER_SLOTS exceeds credit width");
    static constexpr uint32_t CREDIT_MASK = (1u << CREDIT_WIDTH) - 1;

    static constexpr uint32_t TOTAL_BITS = NUM_CHANNELS_IN_THIS_VC * CREDIT_WIDTH;
    static_assert(TOTAL_BITS <= STREAM_REG_CFG_DATA_WIDTH, "Exceeds max register capacity");

    using PackedCreditValueType = PackedCreditValue<NUM_CHANNELS_IN_THIS_VC, MAX_BUFFER_SLOTS>;

    union PackedCreditData {
        uint32_t packed;
        uint8_t bytes[4];
        uint16_t halfs[2];
    };

    static constexpr uint32_t bit_offset(uint8_t channel) { return channel * CREDIT_WIDTH; }

    template <uint8_t CHANNEL>
    static constexpr uint32_t bit_offset() {
        return bit_offset(CHANNEL);
    }

    FORCE_INLINE static PackedCreditValueType get_packed() { return PackedCreditValueType(get_ptr_val(stream_id)); }

    FORCE_INLINE static uint32_t get_sum(PackedCreditValueType value) {
        if constexpr (NUM_CHANNELS_IN_THIS_VC == 1) {
            return value.get();
        }

        PackedCreditData data{value.get()};
        if constexpr (credits_are_byte_aligned) {
            if constexpr (NUM_CHANNELS_IN_THIS_VC == 2) {
                return data.bytes[0] + data.bytes[1];
            } else {
                static_assert(MAX_SLOTS_PER_CHANNEL <= 64, "This implementation assumes a max of 6 bits per credit");
                static_assert(NUM_CHANNELS_IN_THIS_VC <= 4, "Only 1-4 channels supported");
                // since we have >= 3 credits, then we are have atleast one credit on the top 16 bits
                // we break the sum down into two parts:
                // 1. sum the upper and lower 16 bits. This leaves a partial sum
                //    upper half contains c2 and possibly c3
                //    lower half contains c0 and c1
                //    result-> byte0 = (c0 + c2), byte1 = (c1 + c3)
                // 2. sum the lower two bytes of the partial sum
                //    result-> byte0 {c0 + c2} + byte1 {c1 + c3}
                //
                // not that we are guaranteed to have enough carry bits here when credit is byte sized
                // because max slot count fits in 6 bits
                auto partial = PackedCreditData{.packed = (data.halfs[0] + data.halfs[1])};
                return partial.bytes[0] + data.bytes[1];
            }
        } else {
            // Non-byte-aligned credits - tree-based shift-and-add reduction
            static_assert(NUM_CHANNELS_IN_THIS_VC == 4, "Non-byte-aligned only for 4 channels");
            static constexpr uint32_t CREDIT_WIDTH_MASK = (1u << CREDIT_WIDTH) - 1;
            static constexpr uint32_t CREDIT_WIDTH_P1_MASK = (CREDIT_WIDTH_MASK << 1) | 1;
            static constexpr uint32_t CREDIT_WIDTH_P2_MASK = (CREDIT_WIDTH_P1_MASK << 1) | 1;
            static constexpr uint32_t spare_bits_for_carry_over = CREDIT_WIDTH - MIN_BITS;
            // a similar approach is taken as above, bit with a little extra complexity because we
            // don't necessarily have this two carry bits free guarantee.
            // To account for this, we at compile time evaluate how many bits free we have to minimize the
            // amount of masking we need to do.
            if constexpr (spare_bits_for_carry_over == 0) {
                constexpr uint32_t c0_shift = 0 * CREDIT_WIDTH;
                constexpr uint32_t c1_shift = 1 * CREDIT_WIDTH;
                constexpr uint32_t c2_shift = 2 * CREDIT_WIDTH;
                constexpr uint32_t c3_shift = 3 * CREDIT_WIDTH;

                uint32_t c0 = (data.packed >> c0_shift) & CREDIT_WIDTH_MASK;
                uint32_t c1 = (data.packed >> c1_shift) & CREDIT_WIDTH_MASK;
                uint32_t c2 = (data.packed >> c2_shift) & CREDIT_WIDTH_MASK;
                uint32_t c3 = (data.packed >> c3_shift) & CREDIT_WIDTH_MASK;

                return (c0 + c1) + (c2 + c3);

            } else if constexpr (spare_bits_for_carry_over == 1) {
                auto sum_of_halfs_with_garbage_at_ends = data.packed + (data.packed >> TWO_CREDIT_WIDTHS);
                // Need to mask the lower partial sum of halfs because the next higher bit may have non-zero values
                // which would corrupt the addition with the top credit in the sum_of_halfs, so we clear any garbage
                // bits the upper half doesn't need
                auto final_sum_with_garbage_at_ends =
                    (sum_of_halfs_with_garbage_at_ends & CREDIT_WIDTH_P1_MASK) +
                    ((sum_of_halfs_with_garbage_at_ends >> CREDIT_WIDTH) & CREDIT_WIDTH_P1_MASK);
                return final_sum_with_garbage_at_ends & CREDIT_WIDTH_P2_MASK;
            } else {
                // We have >= 2 spare bits so we only need to mask out the very final result
                auto sum_of_halfs = data.packed + (data.packed >> TWO_CREDIT_WIDTHS);
                auto final_sum_with_garbage_at_ends = sum_of_halfs + (sum_of_halfs >> CREDIT_WIDTH);
                return final_sum_with_garbage_at_ends & CREDIT_WIDTH_P2_MASK;
            }
        }
    }

    FORCE_INLINE static uint32_t get_value(uint8_t channel_id, PackedCreditValueType value) {
        PackedCreditData data{value.get()};
        if constexpr (credits_are_byte_aligned) {
            return data.bytes[channel_id];
        } else {
            return (data.packed >> bit_offset(channel_id)) & CREDIT_MASK;
        }
    }

    template <uint8_t CHANNEL>
    FORCE_INLINE static uint32_t get_value(PackedCreditValueType value) {
        return get_value(CHANNEL, value);
    }

    FORCE_INLINE static PackedCreditValueType pack_value(uint8_t channel_id, uint32_t value) {
        return PackedCreditValueType(value << bit_offset(channel_id));
    }

    template <uint32_t channel_id>
    FORCE_INLINE static PackedCreditValueType pack_value(uint32_t value) {
        return pack_value(channel_id, value);
    }

    FORCE_INLINE static uint32_t get_sum() { return get_sum(get_packed()); }
};

}  // namespace tt::tt_fabric
