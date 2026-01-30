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
    // 8-bit credits are always byte-aligned, regardless of channel count (even 5 channels uses 2 words)
    static constexpr bool credits_are_byte_aligned = (NUM_CHANNELS_IN_THIS_VC <= 3);
    static constexpr uint32_t MIN_BITS = log2_ceil(MAX_BUFFER_SLOTS + 1);
    static constexpr uint32_t CREDIT_WIDTH = credits_are_byte_aligned ? 8 : MIN_BITS;
    static constexpr uint32_t TWO_CREDIT_WIDTHS = 2 * CREDIT_WIDTH;
    static_assert(MAX_BUFFER_SLOTS < (1u << CREDIT_WIDTH), "MAX_BUFFER_SLOTS exceeds credit width");
    static constexpr uint32_t CREDIT_MASK = (1u << CREDIT_WIDTH) - 1;

    static constexpr uint32_t TOTAL_BITS = NUM_CHANNELS_IN_THIS_VC * CREDIT_WIDTH;
    static_assert(TOTAL_BITS <= STREAM_REG_CFG_DATA_WIDTH, "Exceeds max register capacity");

    using PackedCreditValueType = NamedType<uint32_t, struct PackedCreditValueTag>;

    union PackedCreditData {
        uint32_t packed;
        uint8_t bytes[4];
        uint16_t halfs[2];
    };

    static constexpr uint32_t bit_offset(uint8_t channel) { return channel * CREDIT_WIDTH; }

    template <uint8_t CHANNEL>
    FORCE_INLINE static constexpr uint32_t bit_offset() {
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
                auto partial = PackedCreditData{
                    .packed = (static_cast<uint32_t>(data.halfs[0]) + static_cast<uint32_t>(data.halfs[1]))};
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

// ============================================================================
// GENERIC PACKED CREDITS IMPLEMENTATION
// ============================================================================

/**
 * USAGE EXAMPLES:
 *
 * 1. Read-only view of overlay register credits (4 channels, 6-bit packing):
 *    using MyCredits = OverlayRegCreditView<4, 6, STREAM_ID_5>;
 *    MyCredits credits;  // Zero-size object!
 *    uint32_t total = credits.get_total();
 *    uint32_t ch2 = credits.get_channel(2);
 *
 * 2. Updatable overlay register credits (3 channels, 8-bit packing):
 *    using MyCredits = OverlayRegCreditUpdater<3, 8, STREAM_ID_10>;
 *    MyCredits credits;
 *    credits.increment_channel<0>(5);  // Atomic increment
 *
 * 3. Memory-backed credits (2 channels, 10-bit packing):
 *    using MyCredits = MemoryCreditUpdater<2, 10>;
 *    volatile uint32_t buffer;
 *    MyCredits credits(&buffer);
 *    credits.set_channel<1>(42);  // Read-modify-write
 *    uint32_t total = credits.get_total();
 *
 * 4. Read-only memory view:
 *    using MyCredits = ConstMemoryCreditView<4, 8>;
 *    const volatile uint32_t* buffer_ptr = get_buffer();
 *    MyCredits credits(buffer_ptr);
 *    uint32_t sum = credits.get_total();
 *
 * 5. Custom configuration with different credit widths:
 *    // 4 channels × 6 bits = 24 bits total (fits in single overlay register)
 *    using TightPacked = OverlayRegCreditView<4, 6, STREAM_ID_0>;
 *
 *    // 3 channels × 8 bits = 24 bits (byte-aligned, faster operations)
 *    using ByteAligned = OverlayRegCreditView<3, 8, STREAM_ID_1>;
 *
 * 6. Multi-register overlay credits (when >24 bits needed):
 *    // 5 channels × 8 bits = 40 bits (needs 2 registers with efficient reduction layout)
 *    // Layout: reg0={ch0,ch1}, reg1={ch2,ch3,ch4}
 *    using MultiRegCredits = MultiOverlayRegCreditView<5, 8, STREAM_ID_0, STREAM_ID_1>;
 *    MultiRegCredits credits;
 *    uint32_t total = credits.get_total();  // Reads both registers, efficient 2-step sum
 *
 *    // For updates:
 *    using MultiRegUpdater = MultiOverlayRegCreditUpdater<5, 8, STREAM_ID_0, STREAM_ID_1>;
 *    MultiRegUpdater updater;
 *    updater.increment_channel<4>(3);  // Automatically writes to reg1
 *
 * 7. 4-channel multi-register (2D case):
 *    // 4 channels × 8 bits = 32 bits, layout: reg0={ch0,ch1}, reg1={ch2,ch3}
 *    using TwoByTwoCredits = MultiOverlayRegCreditView<4, 8, STREAM_ID_0, STREAM_ID_1>;
 *
 * KEY FEATURES:
 * - Compile-time configuration: All parameters known at compile time
 * - Zero overhead: Stateless backends compile to zero size with [[no_unique_address]]
 * - Automatic optimization: Byte-aligned packing uses faster sum algorithms
 * - Mix and match: Any channel count + credit width + storage type
 * - Type-safe: Can't accidentally mix different configurations
 */

// ============================================================================
// LAYER 1: CREDIT PACKING POLICIES
// ============================================================================

/**
 * Packed credit container - type-safe, ZERO-OVERHEAD wrapper for packed credit data.
 * Automatically selects uint32_t or uint64_t storage based on total bits needed.
 *
 * For ≤32 bits (e.g., 4 channels × 8 bits): uses uint32_t
 * For >32 bits (e.g., 5 channels × 8 bits = 40 bits): uses uint64_t
 *
 * ZERO-OVERHEAD GUARANTEE:
 * - Trivial aggregate type (no user-defined constructors/destructors)
 * - Standard layout (POD-like)
 * - Same size as underlying storage_type (sizeof(Container) == sizeof(uint32_t) or sizeof(uint64_t))
 * - No initialization instructions generated (uninitialized = uninitialized, no hidden zero-init)
 * - Trivially copyable (memcpy-safe)
 * - Compiler can treat as raw integer in optimizations
 *
 * USAGE:
 *   PackedCreditContainer<4, 8> credits{0};  // Aggregate initialization
 *   auto value = credits.get();              // Extract raw value
 *   credits.value = 42;                      // Direct access
 */
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH_BITS>
struct PackedCreditContainer {
private:
    static constexpr uint32_t TOTAL_BITS = NUM_CHANNELS * CREDIT_WIDTH_BITS;
    static_assert(TOTAL_BITS <= 64, "Total packed bits exceeds 64-bit capacity");

public:
    // Select storage type based on total bits
    using storage_type = std::conditional_t<(TOTAL_BITS <= 32), uint32_t, uint64_t>;

    // Public data member - makes this a trivial aggregate type
    storage_type value;

    // Static factory functions (no constructors = trivial type)
    static constexpr PackedCreditContainer make(storage_type v) {
        return PackedCreditContainer{v};
    }

    // Get the raw value
    constexpr storage_type get() const { return value; }

    // Get as uint32_t (for compatibility, only valid if fits)
    constexpr uint32_t get_u32() const {
        static_assert(TOTAL_BITS <= 32, "Cannot convert >32-bit packed value to uint32_t");
        return static_cast<uint32_t>(value);
    }

    // Get as uint64_t (always valid)
    constexpr uint64_t get_u64() const {
        return static_cast<uint64_t>(value);
    }

    // Comparison operators
    constexpr bool operator==(const PackedCreditContainer& other) const {
        return value == other.value;
    }
    constexpr bool operator!=(const PackedCreditContainer& other) const {
        return value != other.value;
    }
    constexpr bool operator==(storage_type raw) const {
        return value == raw;
    }
    constexpr bool operator!=(storage_type raw) const {
        return value != raw;
    }

    // Arithmetic operators
    constexpr PackedCreditContainer operator+(const PackedCreditContainer& other) const {
        return PackedCreditContainer{value + other.value};
    }
    constexpr PackedCreditContainer operator-(const PackedCreditContainer& other) const {
        return PackedCreditContainer{value - other.value};
    }
};

// Static assertions to verify zero-overhead properties
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH_BITS>
struct PackedCreditContainerTraits {
    using Container = PackedCreditContainer<NUM_CHANNELS, CREDIT_WIDTH_BITS>;
    using storage_type = typename Container::storage_type;

    // Verify zero-overhead properties
    static_assert(sizeof(Container) == sizeof(storage_type), "Container must be same size as storage_type");
    static_assert(std::is_trivially_copyable<Container>::value, "Container must be trivially copyable");
    static_assert(std::is_standard_layout<Container>::value, "Container must be standard layout");
};

/**
 * Credit packing policy - handles bit-level packing and unpacking of credits.
 * This is a stateless policy class containing only static methods.
 *
 * Template parameters:
 * - NUM_CHANNELS: Number of channels to pack (1-5)
 * - CREDIT_WIDTH_BITS: Bits per credit (e.g., 6, 8, 10)
 *
 * Constraints:
 * - Total bits (NUM_CHANNELS * CREDIT_WIDTH_BITS) must fit in 64 bits
 */
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH_BITS>
struct CreditPacking {
    static_assert(NUM_CHANNELS >= 1 && NUM_CHANNELS <= 5, "Only 1-5 channels supported");
    static_assert(CREDIT_WIDTH_BITS >= 1 && CREDIT_WIDTH_BITS <= 16, "Credit width must be 1-16 bits");

    static constexpr uint8_t NUM_CHANNELS_VALUE = NUM_CHANNELS;
    static constexpr uint8_t CREDIT_WIDTH = CREDIT_WIDTH_BITS;
    static constexpr uint32_t CREDIT_MASK = (1u << CREDIT_WIDTH) - 1;
    static constexpr uint32_t TWO_CREDIT_WIDTHS = 2 * CREDIT_WIDTH;
    static constexpr uint32_t TOTAL_BITS = NUM_CHANNELS * CREDIT_WIDTH;
    static_assert(TOTAL_BITS <= 64, "Total packed bits exceeds 64-bit capacity");

    using PackedValueType = PackedCreditContainer<NUM_CHANNELS, CREDIT_WIDTH_BITS>;
    using storage_type = typename PackedValueType::storage_type;

    // Check if credits are byte-aligned (for optimization)
    // 8-bit credits are always byte-aligned, regardless of channel count
    static constexpr bool credits_are_byte_aligned = (CREDIT_WIDTH == 8);

private:
    // Helper union for byte/half access optimization (used locally in functions)
    union PackedDataView {
        storage_type packed;
        uint8_t bytes[sizeof(storage_type)];
        uint16_t halfs[sizeof(storage_type) / 2];
    };

    /**
     * Helper: Sum 2 byte-aligned channels from packed value
     */
    FORCE_INLINE static uint8_t sum_2_byte_aligned_channels(storage_type packed_value) {
        PackedDataView data{packed_value};
        return data.bytes[0] + data.bytes[1];
    }

    /**
     * Helper: Sum 3 byte-aligned channels from packed value
     * Uses optimized half-word reduction (from original PackedCredits)
     *
     * How it works:
     * - halfs[0] contains c0 (bits 0-7) and c1 (bits 8-15)
     * - halfs[1] contains c2 (bits 16-23) and 0 (bits 24-31)
     * - Adding halfs gives: byte0 = (c0 + c2), byte1 = c1
     * - Final sum: (c0 + c2) + c1
     */
    FORCE_INLINE static uint8_t sum_3_byte_aligned_channels(storage_type packed_value) {        
        PackedDataView data{packed_value};
        // Sum upper and lower 16 bits
        auto partial = PackedDataView{
            .packed = static_cast<storage_type>((static_cast<uint32_t>(data.halfs[0]) + static_cast<uint32_t>(data.halfs[1])))};
        // For 3 channels: partial.bytes[1] == data.bytes[1] (both are c1)
        return partial.bytes[0] + data.bytes[1];
    }

    /**
     * Helper: Sum 4 byte-aligned channels from packed value
     * Uses optimized half-word reduction (from original PackedCredits)
     *
     * How it works:
     * - halfs[0] contains c0 (bits 0-7) and c1 (bits 8-15)
     * - halfs[1] contains c2 (bits 16-23) and c3 (bits 24-31)
     * - Adding halfs gives: byte0 = (c0 + c2), byte1 = (c1 + c3)
     * - Final sum: (c0 + c2) + (c1 + c3)
     */
    FORCE_INLINE static uint8_t sum_4_byte_aligned_channels(const storage_type& packed_value) {
        return static_cast<uint8_t>((static_cast<uint32_t>(packed_value) * 0x01010101u) >> 24);
    }

    /**
     * Helper: Sum 2 non-byte-aligned channels from packed value
     */
    FORCE_INLINE static uint8_t sum_2_non_byte_aligned_channels(storage_type packed_value) {
        uint32_t c0 = static_cast<uint32_t>(packed_value) & CREDIT_MASK;
        uint32_t c1 = (static_cast<uint32_t>(packed_value) >> CREDIT_WIDTH) & CREDIT_MASK;
        return c0 + c1;
    }

    /**
     * Helper: Sum 3 non-byte-aligned channels from packed value
     */
    FORCE_INLINE static uint8_t sum_3_non_byte_aligned_channels(storage_type packed_value) {
        uint32_t c0 = static_cast<uint32_t>(packed_value) & CREDIT_MASK;
        uint32_t c1 = (static_cast<uint32_t>(packed_value) >> CREDIT_WIDTH) & CREDIT_MASK;
        uint32_t c2 = (static_cast<uint32_t>(packed_value) >> (2 * CREDIT_WIDTH)) & CREDIT_MASK;
        return c0 + c1 + c2;
    }

    /**
     * Helper: Sum 4 non-byte-aligned channels from packed value
     * Uses tree-based reduction with compile-time optimization based on spare bits
     */
    FORCE_INLINE static uint32_t sum_4_non_byte_aligned_channels(storage_type packed_value) {
        static constexpr uint32_t MAX_CREDIT_VALUE = (1u << CREDIT_WIDTH) - 1;
        static constexpr uint32_t MIN_BITS_NEEDED = log2_ceil(MAX_CREDIT_VALUE + 1);
        static constexpr uint32_t spare_bits_for_carry_over = CREDIT_WIDTH - MIN_BITS_NEEDED;

        static constexpr uint32_t CREDIT_WIDTH_MASK = CREDIT_MASK;
        static constexpr uint32_t CREDIT_WIDTH_P1_MASK = (CREDIT_WIDTH_MASK << 1) | 1;
        static constexpr uint32_t CREDIT_WIDTH_P2_MASK = (CREDIT_WIDTH_P1_MASK << 1) | 1;

        PackedDataView data{packed_value};

        if constexpr (spare_bits_for_carry_over == 0) {
            // No spare bits - must extract and add individually
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
            // 1 spare bit - need careful masking
            auto sum_of_halfs_with_garbage_at_ends = data.packed + (data.packed >> TWO_CREDIT_WIDTHS);
            auto final_sum_with_garbage_at_ends =
                (sum_of_halfs_with_garbage_at_ends & CREDIT_WIDTH_P1_MASK) +
                ((sum_of_halfs_with_garbage_at_ends >> CREDIT_WIDTH) & CREDIT_WIDTH_P1_MASK);
            return final_sum_with_garbage_at_ends & CREDIT_WIDTH_P2_MASK;
        } else {
            // >= 2 spare bits - only mask final result
            auto sum_of_halfs = data.packed + (data.packed >> TWO_CREDIT_WIDTHS);
            auto final_sum_with_garbage_at_ends = sum_of_halfs + (sum_of_halfs >> CREDIT_WIDTH);
            return final_sum_with_garbage_at_ends & CREDIT_WIDTH_P2_MASK;
        }
    }

public:
    /**
     * Get the bit offset for a given channel
     */
    static constexpr uint32_t bit_offset(uint8_t channel) {
        return channel * CREDIT_WIDTH;
    }

    template <uint8_t CHANNEL>
    static constexpr uint32_t bit_offset() {
        static_assert(CHANNEL < NUM_CHANNELS, "Channel index out of bounds");
        return bit_offset(CHANNEL);
    }

    /**
     * Extract a single channel's credit value from packed container
     */
    FORCE_INLINE static uint32_t extract_channel(const PackedValueType& packed, uint8_t channel_id) {
        PackedDataView data{packed.value};
        if constexpr (credits_are_byte_aligned) {
            return data.bytes[channel_id];
        } else {
            return (data.packed >> bit_offset(channel_id)) & CREDIT_MASK;
        }
    }

    template <uint8_t CHANNEL>
    FORCE_INLINE static uint32_t extract_channel(const PackedValueType& packed) {
        return extract_channel(packed, CHANNEL);
    }

    /**
     * Pack a single channel's credit value into a new container
     */
    FORCE_INLINE static constexpr PackedValueType pack_channel(uint8_t channel_id, uint32_t value) {
        return PackedValueType{static_cast<storage_type>(value) << bit_offset(channel_id)};
    }

    template <uint8_t CHANNEL>
    FORCE_INLINE static constexpr PackedValueType pack_channel(uint32_t value) {
        return pack_channel(CHANNEL, value);
    }

    /**
     * Pack a single channel's credit value into an existing container
     * Returns the modified container
     *
     * IMPORTANT: This REPLACES the channel value, not ORs it.
     * This is critical for unbounded counters where the same channel is updated multiple times.
     */
    FORCE_INLINE static PackedValueType pack_channel(PackedValueType& packed, uint8_t channel_id, uint32_t value) {
        if constexpr (credits_are_byte_aligned) {
            // Byte-aligned: use efficient byte store
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed.value);
            bytes[channel_id] = static_cast<uint8_t>(value);
        } else {
            // Non-byte-aligned: mask out old value and set new value
            storage_type channel_mask = static_cast<storage_type>(CREDIT_MASK) << bit_offset(channel_id);
            packed.value = (packed.value & ~channel_mask) | (static_cast<storage_type>(value) << bit_offset(channel_id));
        }
        return packed;
    }

    template <uint8_t CHANNEL>
    FORCE_INLINE static PackedValueType pack_channel(PackedValueType& packed, uint32_t value) {
        if constexpr (credits_are_byte_aligned) {
            // Byte-aligned: use efficient byte store
            uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed.value);
            bytes[CHANNEL] = static_cast<uint8_t>(value);
        } else {
            // Non-byte-aligned: mask out old value and set new value
            constexpr storage_type channel_mask = static_cast<storage_type>(CREDIT_MASK) << bit_offset<CHANNEL>();
            packed.value = (packed.value & ~channel_mask) | (static_cast<storage_type>(value) << bit_offset<CHANNEL>());
        }
        return packed;
    }

    /**
     * Sum all channels in the packed value
     * Uses optimized algorithms based on channel count and alignment
     *
     * For 5 channels, uses 2-step reduction matching multi-register layout:
     *   - Step 1: Sum channels 0-1 (first 2 channels, bits 0-15)
     *   - Step 2: Sum channels 2-4 (next 3 channels, bits 16-39)
     *   - Step 3: Add partial sums
     */
    FORCE_INLINE static uint8_t sum_all_channels(const PackedValueType& packed) {
        if constexpr (NUM_CHANNELS == 1) {
            return static_cast<uint8_t>(packed.value & CREDIT_MASK);
        }
        static_assert(credits_are_byte_aligned);
        if constexpr (credits_are_byte_aligned) {
            // Byte-aligned optimization
            if constexpr (NUM_CHANNELS == 1) {
                return packed.value & ((1 << CREDIT_WIDTH) - 1);
            } else if constexpr (NUM_CHANNELS == 2) {
                return sum_2_byte_aligned_channels(packed.value);
            } else if constexpr (NUM_CHANNELS == 3) {
                return sum_3_byte_aligned_channels(packed.value);
            } else if constexpr (NUM_CHANNELS == 4) {
                return sum_4_byte_aligned_channels(packed.value);
            } else if constexpr (NUM_CHANNELS == 5) {
                // 5 byte-aligned channels: 3-step reduction
                // Step 1: partial1 = (data & 0xffffffff) + (data >> 32)
                //         This adds c4 (from upper 8 bits) to c0 (lower byte of lower 32 bits)
                //         Result bytes: [c0+c4, c1, c2, c3]
                // Step 2: partial2 = partial1.halfs[0] + partial1.halfs[1]
                //         Result bytes: [(c0+c4)+c2, c1+c3, ...]
                // Step 3: final = partial2.bytes[0] + partial2.bytes[1]
                static_assert(NUM_CHANNELS == 5, "This path is only for 5 channels");
                static_assert(CREDIT_WIDTH == 8, "5-channel reduction assumes 8-bit credits");
                static_assert(sizeof(storage_type) == 8, "5 channels requires uint64_t storage");

                uint32_t reg0 = static_cast<uint32_t>(packed.value & 0xffffffffULL);  // bits 0-31: c0, c1, c2, c3
                uint32_t reg1 = static_cast<uint32_t>(packed.value >> 32);            // bits 32-39: c4

                PackedDataView partial1{static_cast<storage_type>(reg0 + reg1)};  // bytes: [c0+c4, c1, c2, c3]
                PackedDataView partial2{static_cast<storage_type>(
                    static_cast<uint32_t>(partial1.halfs[0]) + static_cast<uint32_t>(partial1.halfs[1]))};  // bytes: [(c0+c4)+c2, c1+c3, ...]

                return partial2.bytes[0] + partial2.bytes[1];  // ((c0+c4)+c2) + (c1+c3) = c0+c1+c2+c3+c4
            } else {
                static_assert(NUM_CHANNELS <= 5, "Unsupported channel count for byte-aligned sum");
                return 0; // Should never reach here due to static_assert
            }
        } else {
            // Non-byte-aligned credits - tree-based shift-and-add reduction
            if constexpr (NUM_CHANNELS == 1) {
                return packed.value & ((1 << CREDIT_WIDTH) - 1);
            } else if constexpr (NUM_CHANNELS == 2) {
                return sum_2_non_byte_aligned_channels(packed.value);
            } else if constexpr (NUM_CHANNELS == 3) {
                return sum_3_non_byte_aligned_channels(packed.value);
            } else if constexpr (NUM_CHANNELS == 4) {
                return sum_4_non_byte_aligned_channels(packed.value);
            } else if constexpr (NUM_CHANNELS == 5) {
                // 5 non-byte-aligned channels: 2-step reduction matching multi-register layout
                // Step 1: Sum channels 0-1 from reg0 equivalent (bits 0 to 2*CREDIT_WIDTH)
                // Step 2: Sum channels 2-4 from reg1 equivalent (bits 2*CREDIT_WIDTH to 5*CREDIT_WIDTH)
                // Step 3: Add partial sums
                constexpr uint32_t reg1_shift = 2 * CREDIT_WIDTH;
                uint32_t sum_ch0_ch1 = sum_2_non_byte_aligned_channels(packed.value);
                uint32_t sum_ch2_ch3_ch4 = sum_3_non_byte_aligned_channels(packed.value >> reg1_shift);
                return sum_ch0_ch1 + sum_ch2_ch3_ch4;
            } else {
                static_assert(NUM_CHANNELS <= 5, "Unsupported channel count for non-byte-aligned sum");
                return 0; // Should never reach here due to static_assert
            }
        }
    }

    // ========================================================================
    // SAFE ARITHMETIC OPERATIONS (Phase 1: Prevent Carry/Borrow Bugs)
    // ========================================================================

    /**
     * Safe single-channel addition - prevents carries from corrupting adjacent channels
     *
     * PERFORMANCE: ~5 instructions with Zbb (RISC-V bit manipulation extension)
     * - Extract: 1 shift + 1 mask (2 inst)
     * - Add: 1 instruction (uint8_t wraps at 256)
     * - Clear old: 1 'andn' instruction (with Zbb)
     * - Repack: 1 shift + 1 OR (2 inst)
     *
     * @tparam CHANNEL Channel index to modify
     * @param packed_value Current packed value
     * @param delta Amount to add (will wrap at 256)
     * @return New packed value with channel incremented
     */
    template <uint8_t CHANNEL>
    FORCE_INLINE static PackedValueType add_to_channel(
        const PackedValueType& packed_value,
        uint8_t delta) {

        static_assert(CHANNEL < NUM_CHANNELS, "Channel index out of bounds");

        // Extract current value - single shift + mask (2 instructions)
        uint8_t old_val = extract_channel<CHANNEL>(packed_value);
        uint8_t new_val = old_val + delta;  // uint8_t wraps at 256 (1 instruction)

        // Efficiently clear old byte and insert new byte
        // Using Zbb 'andn': result = a & ~b (1 instruction instead of 2)
        constexpr storage_type mask = static_cast<storage_type>(CREDIT_MASK) << bit_offset<CHANNEL>();
        storage_type raw = packed_value.value;

        // Compiler will emit 'andn' instruction with Zbb:
        //   andn t0, raw, mask      # Clear old byte
        //   slli t1, new_val, shift # Position new byte
        //   or   t0, t0, t1         # Insert new byte
        storage_type cleared = raw & ~mask;  // Zbb: andn (1 inst)
        storage_type positioned = static_cast<storage_type>(new_val) << bit_offset<CHANNEL>();
        return PackedValueType{cleared | positioned};

        // Total: ~5 instructions, all inlined, no branches, no memory access
    }

    /**
     * Safe single-channel subtraction - prevents borrows from corrupting adjacent channels
     *
     * PERFORMANCE: ~5 instructions (identical to add_to_channel)
     *
     * @tparam CHANNEL Channel index to modify
     * @param packed_value Current packed value
     * @param delta Amount to subtract (will wrap with borrow)
     * @return New packed value with channel decremented
     */
    template <uint8_t CHANNEL>
    FORCE_INLINE static PackedValueType subtract_from_channel(
        const PackedValueType& packed_value,
        uint8_t delta) {

        static_assert(CHANNEL < NUM_CHANNELS, "Channel index out of bounds");

        uint8_t old_val = extract_channel<CHANNEL>(packed_value);
        uint8_t new_val = old_val - delta;  // uint8_t wraps with borrow

        constexpr storage_type mask = static_cast<storage_type>(CREDIT_MASK) << bit_offset<CHANNEL>();
        storage_type cleared = packed_value.value & ~mask;  // Zbb: andn
        return PackedValueType{cleared | (static_cast<storage_type>(new_val) << bit_offset<CHANNEL>())};

        // Total: ~5 instructions, identical to add_to_channel
    }

    /**
     * Safe difference between two packed values for a specific channel
     * Used to compute unprocessed acks: acks_received - acks_processed
     *
     * PERFORMANCE: ~3 instructions (2 extractions + 1 subtraction)
     *
     * @tparam CHANNEL Channel index to compare
     * @param minuend Value to subtract from
     * @param subtrahend Value to subtract
     * @return Difference for this channel (wraps correctly)
     */
    template <uint8_t CHANNEL>
    FORCE_INLINE static uint8_t diff_channels(
        const PackedValueType& minuend,
        const PackedValueType& subtrahend) {

        static_assert(CHANNEL < NUM_CHANNELS, "Channel index out of bounds");

        // Extract and subtract in one step - compiler optimizes to 2 shifts + 1 sub
        uint8_t a = extract_channel<CHANNEL>(minuend);
        uint8_t b = extract_channel<CHANNEL>(subtrahend);
        return a - b;  // uint8_t wraps correctly

        // Total: ~3 instructions (2 extractions + 1 subtraction)
    }

    /**
     * Safe multi-channel addition - adds two packed values without carries between channels
     *
     * This is the HOT PATH for receiver ack processing!
     * Uses template specialization to provide custom optimized implementations per channel count.
     *
     * PERFORMANCE (in-order ERISC, with Zbb):
     * - 1 channel: 4 instructions
     * - 2 channels: 8 instructions
     * - 3 channels: 10 instructions
     * - 4 channels: 13 instructions
     * - 5+ channels: ~6*N instructions (generic fallback)
     *
     * All implementations are fully unrolled (no loops, no branches).
     *
     * @param a First packed value
     * @param b Second packed value
     * @return Sum with per-channel wrapping (no carries between channels)
     */
    FORCE_INLINE static PackedValueType add_packed(
        const PackedValueType& a,
        const PackedValueType& b) {

        // Template specialization provides optimal code for each channel count
        if constexpr (NUM_CHANNELS == 1) {
            // Single byte addition - trivial case
            // Compiler optimizes to 4 instructions:
            //   andi  a0, a0, 0xFF      # Mask to byte
            //   andi  a1, a1, 0xFF      # Mask to byte
            //   add   a0, a0, a1        # Add (uint8_t wraps automatically)
            //   andi  a0, a0, 0xFF      # Mask result
            uint8_t val_a = static_cast<uint8_t>(a.value & CREDIT_MASK);
            uint8_t val_b = static_cast<uint8_t>(b.value & CREDIT_MASK);
            return PackedValueType{static_cast<storage_type>((val_a + val_b) & CREDIT_MASK)};

        } else if constexpr (NUM_CHANNELS == 2) {
            // 2 channels: fully unrolled
            storage_type av = a.value;
            storage_type bv = b.value;

            // Channel 0 (byte 0)
            uint8_t ch0_a = static_cast<uint8_t>(av & CREDIT_MASK);
            uint8_t ch0_b = static_cast<uint8_t>(bv & CREDIT_MASK);
            uint8_t ch0_sum = ch0_a + ch0_b;  // Wraps at 256

            // Channel 1 (byte 1) - extract directly from shifted position
            uint8_t ch1_a = static_cast<uint8_t>((av >> CREDIT_WIDTH) & CREDIT_MASK);
            uint8_t ch1_b = static_cast<uint8_t>((bv >> CREDIT_WIDTH) & CREDIT_MASK);
            uint8_t ch1_sum = ch1_a + ch1_b;  // Wraps at 256

            // Repack (no OR needed for ch0, already in position)
            storage_type result = static_cast<storage_type>(ch0_sum) |
                                  (static_cast<storage_type>(ch1_sum) << CREDIT_WIDTH);
            return PackedValueType{result};

            // Total: ~8 instructions with Zbb, fully inlined, no branches

        } else if constexpr (NUM_CHANNELS == 3) {
            // 3 channels: fully unrolled
            storage_type av = a.value;
            storage_type bv = b.value;

            // Three independent byte additions
            uint8_t ch0_sum = static_cast<uint8_t>((av & CREDIT_MASK) + (bv & CREDIT_MASK));
            uint8_t ch1_sum = static_cast<uint8_t>(((av >> CREDIT_WIDTH) & CREDIT_MASK) +
                                                     ((bv >> CREDIT_WIDTH) & CREDIT_MASK));
            uint8_t ch2_sum = static_cast<uint8_t>(((av >> (2 * CREDIT_WIDTH)) & CREDIT_MASK) +
                                                     ((bv >> (2 * CREDIT_WIDTH)) & CREDIT_MASK));

            // Repack all three bytes
            storage_type result = static_cast<storage_type>(ch0_sum) |
                                  (static_cast<storage_type>(ch1_sum) << CREDIT_WIDTH) |
                                  (static_cast<storage_type>(ch2_sum) << (2 * CREDIT_WIDTH));
            return PackedValueType{result};

            // Total: ~10 instructions with Zbb, fully inlined

        } else if constexpr (NUM_CHANNELS == 4) {
            // 4 channels: optimized for in-order ERISC
            storage_type av = a.value;
            storage_type bv = b.value;

            // Four independent byte operations - fully unrolled, no loop overhead
            // Each operation: extract (shift+mask), add (uint8_t wraps), position (shift)
            uint8_t ch0 = static_cast<uint8_t>((av & CREDIT_MASK) + (bv & CREDIT_MASK));
            uint8_t ch1 = static_cast<uint8_t>(((av >> CREDIT_WIDTH) & CREDIT_MASK) +
                                                ((bv >> CREDIT_WIDTH) & CREDIT_MASK));
            uint8_t ch2 = static_cast<uint8_t>(((av >> (2 * CREDIT_WIDTH)) & CREDIT_MASK) +
                                                ((bv >> (2 * CREDIT_WIDTH)) & CREDIT_MASK));
            uint8_t ch3 = static_cast<uint8_t>(((av >> (3 * CREDIT_WIDTH)) & CREDIT_MASK) +
                                                ((bv >> (3 * CREDIT_WIDTH)) & CREDIT_MASK));

            // Repack into result - compiler optimizes OR sequence efficiently
            storage_type result = static_cast<storage_type>(ch0) |
                                  (static_cast<storage_type>(ch1) << CREDIT_WIDTH) |
                                  (static_cast<storage_type>(ch2) << (2 * CREDIT_WIDTH)) |
                                  (static_cast<storage_type>(ch3) << (3 * CREDIT_WIDTH));
            return PackedValueType{result};

            // Total: ~13 instructions with Zbb
            // Dependency chain depth: ~8 cycles on in-order core
            // Better than loop: no branch prediction, no loop counter

        } else {
            // 5+ channels: generic unrolled loop for unusual channel counts
            storage_type result = 0;

            #pragma unroll
            for (uint8_t ch = 0; ch < NUM_CHANNELS; ++ch) {
                uint8_t val_a = extract_channel(a, ch);
                uint8_t val_b = extract_channel(b, ch);
                uint8_t sum = val_a + val_b;  // Wraps at 256
                result |= (static_cast<storage_type>(sum) << (ch * CREDIT_WIDTH));
            }

            return PackedValueType{result};

            // Compiler fully unrolls for compile-time NUM_CHANNELS
        }
    }
};

// ============================================================================
// LAYER 2: STORAGE BACKENDS
// ============================================================================

/**
 * Storage backend for overlay registers (increment-on-write, 24-bit words)
 * Stateless - stream_id is known at compile time
 * Single register variant - for up to 24 bits of packed credits
 *
 * ZERO-OVERHEAD GUARANTEE:
 * - No data members (only static constexpr members and static methods)
 * - Empty type: sizeof = 1 byte (C++ minimum)
 * - With [[no_unique_address]], occupies 0 bytes in parent class
 * - Default constructor is trivial and generates NO code
 * - No runtime state, no initialization cost, no destructor cost
 */
template <size_t stream_id>
struct OverlayRegStorage {
    static constexpr uint32_t WORD_WIDTH = 24;
    static constexpr size_t STREAM_ID = stream_id;
    static constexpr size_t NUM_REGISTERS = 1;
    using storage_type = uint32_t;  // Single register always uses uint32_t

    FORCE_INLINE static storage_type read() {
        return get_ptr_val(stream_id);
    }

    /**
     * Atomic increment for overlay registers
     * The packed_delta contains the values to add to each channel
     */
    FORCE_INLINE static void atomic_increment(storage_type packed_delta) {
        increment_local_update_ptr_val(stream_id, packed_delta);
    }

    /**
     * Decrement by packed value - writes negative delta for atomic decrement
     */
    FORCE_INLINE static void decrement_packed(storage_type packed_value) {
        // DPRINT << "OVERLAY_DECR: stream=" << stream_id
        //        << " packed=" << HEX() << packed_value
        //        << " delta=" << -packed_value << ENDL();
        increment_local_update_ptr_val(stream_id, -packed_value);
    }
};

/**
 * Storage backend for multiple overlay registers (increment-on-write, 24-bit words each)
 * Stateless - stream_ids are known at compile time
 * Use this when total packed bits exceed 24 bits (e.g., 5 channels × 8 bits = 40 bits needs 2 registers)
 *
 * EFFICIENT REDUCTION-OPTIMIZED REGISTER LAYOUT:
 * - Register 0: channels 0-1 (2 credits at bits 0-15)
 * - Register 1: remaining channels 2-N (filling up to 24 bits)
 *
 * This layout enables efficient 2-step reduction:
 *   1. Sum credits in reg0: credit0 + credit1
 *   2. Sum credits in reg1: credit2 + credit3 [+ credit4]
 *   3. Add partial sums: sum_reg0 + sum_reg1
 *
 * Examples:
 * - 4 channels × 8 bits = 32 bits:
 *   - Reg0: ch0 (bits 0-7), ch1 (bits 8-15)
 *   - Reg1: ch2 (bits 0-7), ch3 (bits 8-15)
 *
 * - 5 channels × 8 bits = 40 bits:
 *   - Reg0: ch0 (bits 0-7), ch1 (bits 8-15)
 *   - Reg1: ch2 (bits 0-7), ch3 (bits 8-15), ch4 (bits 16-23)
 *
 * Note: CreditPacking still uses linear bit layout (ch_i at offset i*CREDIT_WIDTH).
 * This storage backend handles remapping to/from the register layout.
 */
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH, size_t... stream_ids>
struct MultiOverlayRegStorage {
    static constexpr uint32_t WORD_WIDTH = 24;
    static constexpr size_t NUM_REGISTERS = sizeof...(stream_ids);
    static constexpr std::array<size_t, NUM_REGISTERS> STREAM_IDS = {stream_ids...};

    static_assert(NUM_REGISTERS == 2, "Currently only 2 registers supported for multi-register storage");
    static_assert(NUM_CHANNELS >= 4 && NUM_CHANNELS <= 5, "Multi-register storage for 4-5 channels");

    static constexpr uint32_t TOTAL_BITS = NUM_CHANNELS * CREDIT_WIDTH;
    // Use uint64_t for storage when total bits > 32
    using storage_type = std::conditional_t<(TOTAL_BITS <= 32), uint32_t, uint64_t>;

private:
    // Channel assignment: reg0 gets 2 channels, reg1 gets the rest
    static constexpr uint8_t CHANNELS_IN_REG0 = 2;
    static constexpr uint8_t CHANNELS_IN_REG1 = NUM_CHANNELS - CHANNELS_IN_REG0;

    // Helper to read a specific register by index
    template <size_t idx>
    FORCE_INLINE static uint32_t read_register() {
        static_assert(idx < NUM_REGISTERS, "Register index out of bounds");
        return get_ptr_val(STREAM_IDS[idx]);
    }

    // Helper to write a specific register by index
    template <size_t idx>
    FORCE_INLINE static void write_register(int32_t value) {
        static_assert(idx < NUM_REGISTERS, "Register index out of bounds");
        increment_local_update_ptr_val(STREAM_IDS[idx], value);
    }

public:
    /**
     * Read and combine values from all registers, remapping to linear layout
     * Returns a value where channel i is at bit offset i * CREDIT_WIDTH
     */
    FORCE_INLINE static storage_type read() {
        // Read both registers
        uint32_t reg0_val = read_register<0>();  // Contains ch0, ch1 at bits 0-15
        uint32_t reg1_val = read_register<1>();  // Contains ch2, ch3[, ch4] at bits 0-23

        // Remap to linear layout: ch0 at 0, ch1 at 8, ch2 at 16, ch3 at 24, ch4 at 32
        // reg0 already has ch0 and ch1 in the right place (bits 0-15)
        // reg1 has ch2, ch3, ch4 at bits 0-23, need to shift to bits 16-39
        constexpr uint32_t reg1_shift = CHANNELS_IN_REG0 * CREDIT_WIDTH;
        // Use storage_type to handle both 32-bit and 64-bit cases
        return static_cast<storage_type>(reg0_val) | (static_cast<storage_type>(reg1_val) << reg1_shift);
    }

    /**
     * Atomic increment across multiple registers
     * Takes linear packed value and splits it into register layout
     */
    FORCE_INLINE static void atomic_increment(storage_type packed_delta) {
        // Extract channels 0-1 for reg0 (bits 0-15 of packed_delta)
        constexpr storage_type reg0_mask = (static_cast<storage_type>(1) << (CHANNELS_IN_REG0 * CREDIT_WIDTH)) - 1;
        uint32_t reg0_delta = static_cast<uint32_t>(packed_delta & reg0_mask);

        // Extract channels 2-N for reg1 (remaining bits, shifted down)
        constexpr uint32_t reg1_shift = CHANNELS_IN_REG0 * CREDIT_WIDTH;
        uint32_t reg1_delta = static_cast<uint32_t>(packed_delta >> reg1_shift);

        // Write to both registers
        write_register<0>(static_cast<int32_t>(reg0_delta));
        write_register<1>(static_cast<int32_t>(reg1_delta));
    }

    /**
     * Decrement all channels by splitting packed value and writing negative deltas
     * This acknowledges/consumes the credits represented by packed_value
     */
    FORCE_INLINE static void decrement_packed(storage_type packed_value) {
        // Split the positive value into register portions
        constexpr storage_type reg0_mask = (static_cast<storage_type>(1) << (CHANNELS_IN_REG0 * CREDIT_WIDTH)) - 1;
        uint32_t reg0_part = static_cast<uint32_t>(packed_value & reg0_mask);

        constexpr uint32_t reg1_shift = CHANNELS_IN_REG0 * CREDIT_WIDTH;
        uint32_t reg1_part = static_cast<uint32_t>(packed_value >> reg1_shift);

        // DPRINT << "MULTI_OVERLAY_DECR: packed=" << HEX() << packed_value
        //        << " reg0_part=" << HEX() << reg0_part << " delta=" << -static_cast<int32_t>(reg0_part)
        //        << " reg1_part=" << HEX() << reg1_part << " delta=" << -static_cast<int32_t>(reg1_part) << ENDL();

        // Write negative deltas to decrement atomically
        write_register<0>(-static_cast<int32_t>(reg0_part));
        write_register<1>(-static_cast<int32_t>(reg1_part));
    }
};

/**
 * Storage backend for regular memory (32-bit words, non-atomic)
 * Stateful - contains a pointer to the memory location
 */
struct MemoryStorage {
    static constexpr uint32_t WORD_WIDTH = 32;

    volatile uint32_t* ptr;

    FORCE_INLINE MemoryStorage(volatile uint32_t* p) : ptr(p) {}

    FORCE_INLINE uint32_t read() const {
        return *ptr;
    }

    FORCE_INLINE void write(uint32_t value) {
        *ptr = value;
    }
};

/**
 * Storage backend for const memory (32-bit words, read-only)
 * Stateful - contains a pointer to the memory location
 */
struct ConstMemoryStorage {
    static constexpr uint32_t WORD_WIDTH = 32;

    const volatile uint32_t* ptr;

    FORCE_INLINE ConstMemoryStorage(const volatile uint32_t* p) : ptr(p) {}

    FORCE_INLINE uint32_t read() const {
        return *ptr;
    }
};

// ============================================================================
// LAYER 3: UNIFIED CREDIT INTERFACE
// ============================================================================

/**
 * Read-only view of packed credits
 * Zero-overhead abstraction when using stateless storage backends
 *
 * Template parameters:
 * - NUM_CHANNELS: Number of channels
 * - CREDIT_WIDTH_BITS: Bits per credit
 * - StorageBackend: Storage implementation (OverlayRegStorage or MemoryStorage)
 */
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH_BITS, typename StorageBackend>
class PackedCreditView {
protected:
    using Packing = CreditPacking<NUM_CHANNELS, CREDIT_WIDTH_BITS>;
    [[no_unique_address]] StorageBackend storage;

public:
    static constexpr uint8_t NUM_CHANNELS_VALUE = NUM_CHANNELS;
    static constexpr uint8_t CREDIT_WIDTH = CREDIT_WIDTH_BITS;

    // Default constructor for stateless backends - guaranteed zero overhead
    FORCE_INLINE constexpr PackedCreditView() = default;

    // Constructor for stateful backends (memory pointers)
    template <typename... Args>
    FORCE_INLINE constexpr PackedCreditView(Args&&... args) : storage(args...) {}

    // Zero-overhead guarantees for stateless backends:
    // - Trivially default constructible (no constructor code generated)
    // - Trivially destructible (no destructor code generated)
    // - Standard layout (predictable memory layout)
    static_assert(std::is_trivially_default_constructible<StorageBackend>::value ||
                  sizeof(StorageBackend) > 1,
                  "Stateless storage backends must be trivially default constructible");

    /**
     * Get the packed value as a container
     */
    FORCE_INLINE PackedCreditContainer<NUM_CHANNELS, CREDIT_WIDTH_BITS> get_packed() const {
        return PackedCreditContainer<NUM_CHANNELS, CREDIT_WIDTH_BITS>{storage.read()};
    }

    // /**
    //  * Get the raw packed value (for legacy compatibility)
    //  */
    // FORCE_INLINE typename PackedCreditContainer<NUM_CHANNELS, CREDIT_WIDTH_BITS>::storage_type get_packed_raw() const
    // {
    //     return storage.read();
    // }

    // /**
    //  * Access the underlying storage (for advanced usage)
    //  */
    // FORCE_INLINE const StorageBackend& get_storage() const {
    //     return storage;
    // }
};

/**
 * Updatable packed credits
 * Supports both atomic increment (overlay regs) and read-modify-write (memory)
 */
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH_BITS, typename StorageBackend>
class PackedCreditUpdater : public PackedCreditView<NUM_CHANNELS, CREDIT_WIDTH_BITS, StorageBackend> {
private:
    using Base = PackedCreditView<NUM_CHANNELS, CREDIT_WIDTH_BITS, StorageBackend>;
    using Packing = typename Base::Packing;

    // Type trait to detect OverlayRegStorage (single or multi-register)
    template <typename T>
    struct is_overlay_storage : std::false_type {};

    template <size_t stream_id>
    struct is_overlay_storage<OverlayRegStorage<stream_id>> : std::true_type {};

    template <uint8_t N_CHANNELS, uint8_t CREDIT_W, size_t... stream_ids>
    struct is_overlay_storage<MultiOverlayRegStorage<N_CHANNELS, CREDIT_W, stream_ids...>> : std::true_type {};

public:
    using Base::Base;  // Inherit constructors

    /**
     * Increment a specific channel by delta
     * Behavior depends on storage type:
     * - OverlayReg: atomic increment
     * - Memory: read-modify-write
     */
    FORCE_INLINE void increment_channel(uint8_t channel_id, uint32_t delta) {
        typename Packing::storage_type packed_delta = Packing::pack_channel(channel_id, delta).get();

        if constexpr (is_overlay_storage<StorageBackend>::value) {
            // Atomic increment for overlay registers
            this->storage.atomic_increment(static_cast<uint32_t>(packed_delta));
        } else {
            // Read-modify-write for memory
            typename Packing::storage_type current = this->storage.read();
            typename Packing::storage_type new_value = current + packed_delta;
            this->storage.write(static_cast<uint32_t>(new_value));
        }
    }

    template <uint8_t CHANNEL>
    FORCE_INLINE void increment_channel(uint32_t delta) {
        increment_channel(CHANNEL, delta);
    }

    /**
     * Decrement all channels by the packed amount
     * For overlay registers: writes negative delta (atomic decrement via increment-on-write)
     * For memory: read-modify-write subtraction
     *
     * This is used to acknowledge/consume credits. Takes a type-safe PackedCreditContainer.
     */
    FORCE_INLINE void decrement_packed(const PackedCreditContainer<NUM_CHANNELS, CREDIT_WIDTH_BITS>& packed_value) {
        if constexpr (is_overlay_storage<StorageBackend>::value) {
            // Overlay registers: Call backend's decrement_packed which handles
            // single vs multi-register splitting automatically
            this->storage.decrement_packed(packed_value.get());
        } else {
            // Memory: Read-modify-write subtraction
            auto current = this->storage.read();
            auto new_value = current - packed_value.get();
            this->storage.write(new_value);
        }
    }

    /**
     * Decrement all channels by raw packed value (for legacy compatibility)
     */
    FORCE_INLINE void decrement_packed_raw(typename PackedCreditContainer<NUM_CHANNELS, CREDIT_WIDTH_BITS>::storage_type packed_value) {
        decrement_packed(PackedCreditContainer<NUM_CHANNELS, CREDIT_WIDTH_BITS>{packed_value});
    }

    /**
     * Set a specific channel to a value (memory only)
     * For overlay registers, this doesn't make sense as they're increment-only
     */
    template <typename T = StorageBackend>
    FORCE_INLINE typename std::enable_if<!is_overlay_storage<T>::value, void>::type
    set_channel(uint8_t channel_id, uint32_t value) {
        typename Packing::storage_type current = this->storage.read();
        typename Packing::storage_type channel_mask = static_cast<typename Packing::storage_type>(Packing::CREDIT_MASK) << Packing::bit_offset(channel_id);
        typename Packing::storage_type new_value = (current & ~channel_mask) | Packing::pack_channel(channel_id, value).get();
        this->storage.write(new_value);
    }

    template <uint8_t CHANNEL, typename T = StorageBackend>
    FORCE_INLINE typename std::enable_if<!is_overlay_storage<T>::value, void>::type
    set_channel(uint32_t value) {
        set_channel(CHANNEL, value);
    }

    /**
     * Write entire packed value (memory only)
     */
    template <typename T = StorageBackend>
    FORCE_INLINE typename std::enable_if<!is_overlay_storage<T>::value, void>::type
    set_packed(uint32_t packed_value) {
        this->storage.write(packed_value);
    }

    /**
     * Access the underlying storage (for advanced usage)
     */
    FORCE_INLINE StorageBackend& get_storage() {
        return this->storage;
    }
};

// ============================================================================
// CONVENIENCE TYPE ALIASES
// ============================================================================

// Single overlay register aliases (for up to 24 bits of packed credits)
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH, size_t stream_id>
using OverlayRegCreditView = PackedCreditView<NUM_CHANNELS, CREDIT_WIDTH, OverlayRegStorage<stream_id>>;

template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH, size_t stream_id>
using OverlayRegCreditUpdater = PackedCreditUpdater<NUM_CHANNELS, CREDIT_WIDTH, OverlayRegStorage<stream_id>>;

// Multi-register overlay aliases (for >24 bits of packed credits, e.g., 5 channels × 8 bits = 40 bits)
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH, size_t... stream_ids>
using MultiOverlayRegCreditView = PackedCreditView<NUM_CHANNELS, CREDIT_WIDTH, MultiOverlayRegStorage<NUM_CHANNELS, CREDIT_WIDTH, stream_ids...>>;

template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH, size_t... stream_ids>
using MultiOverlayRegCreditUpdater = PackedCreditUpdater<NUM_CHANNELS, CREDIT_WIDTH, MultiOverlayRegStorage<NUM_CHANNELS, CREDIT_WIDTH, stream_ids...>>;

// Memory aliases (common configurations)
template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH>
using MemoryCreditView = PackedCreditView<NUM_CHANNELS, CREDIT_WIDTH, MemoryStorage>;

template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH>
using MemoryCreditUpdater = PackedCreditUpdater<NUM_CHANNELS, CREDIT_WIDTH, MemoryStorage>;

template <uint8_t NUM_CHANNELS, uint8_t CREDIT_WIDTH>
using ConstMemoryCreditView = PackedCreditView<NUM_CHANNELS, CREDIT_WIDTH, ConstMemoryStorage>;

}  // namespace tt::tt_fabric
