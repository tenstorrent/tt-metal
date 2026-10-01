// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <limits>
#include <tt-metalium/bfloat16.hpp>

#include <tt-logger/tt-logger.hpp>

namespace tt::test_utils {

//! Generic Library of templated packing/unpacking functions.
//! Custom ValueType is supported as long as it is trivially copyable and its
//! sizeof matches one of {1, 2, 4, 8} bytes (so std::bit_cast can shuffle the
//! raw bits in/out of the matching unsigned integer). Both pack_vector and
//! unpack_vector use the same bit_cast path, so no per-type special cases are
//! needed (bfloat16, float8_e4m3, df::float32, etc. all flow through the same
//! template instantiation).

// Maps a byte width N in {1, 2, 4, 8} to the matching unsigned integer type
// (uint8_t / uint16_t / uint32_t / uint64_t). Shared by pack_vector and
// unpack_vector so they bit_cast through the same intermediate integer type.
constexpr auto sized_unsigned = []<std::size_t N>() {
    if constexpr (N == 1) {
        return std::uint8_t{};
    } else if constexpr (N == 2) {
        return std::uint16_t{};
    } else if constexpr (N == 4) {
        return std::uint32_t{};
    } else if constexpr (N == 8) {
        return std::uint64_t{};
    } else {
        static_assert(N == 1 || N == 2 || N == 4 || N == 8, "unsupported size");
    }
};

template <typename PackType, typename ValueType>
std::vector<PackType> pack_vector(const std::vector<ValueType>& values) {
    static_assert(
        std::is_integral_v<PackType>,
        "Packed Type must be an integral type we are packing to -- uint8_t/uint16_t/uint32_t...");
    static_assert(
        std::is_trivially_copyable_v<ValueType>,
        "ValueType must be trivially copyable so std::bit_cast can extract bits");
    TT_FATAL(
        sizeof(PackType) >= sizeof(ValueType),
        "sizeof(PackType)={} >= sizeof(ValueType))={}",
        sizeof(PackType),
        sizeof(ValueType));
    TT_FATAL(
        (sizeof(PackType) % sizeof(ValueType)) == 0,
        "sizeof(PackType)={} % sizeof(ValueType)={} must equal 0",
        sizeof(PackType),
        sizeof(ValueType));
    using bits_type = decltype(sized_unsigned.template operator()<sizeof(ValueType)>());
    constexpr unsigned int num_values_to_pack = sizeof(PackType) / sizeof(ValueType);
    TT_FATAL(
        (values.size() % num_values_to_pack) == 0,
        "Number of values must evenly divide into the final packed type... no padding assumed");
    std::vector<PackType> results(values.size() / num_values_to_pack, 0);
    unsigned int index = 0;
    std::for_each(results.begin(), results.end(), [&](PackType& result) {
        for (unsigned j = 0; j < num_values_to_pack; j++) {
            const auto bits = std::bit_cast<bits_type>(values[index]);
            result |= static_cast<PackType>(bits) << (j * sizeof(ValueType) * CHAR_BIT);
            index++;
        }
        return result;
    });
    return results;
}

template <typename ValueType, typename PackType>
std::vector<ValueType> unpack_vector(const std::vector<PackType>& values) {
    static_assert(
        std::is_integral_v<PackType>,
        "Packed Type must be an integral type we are packing to -- uint8_t/uint16_t/uint32_t...");
    static_assert(
        std::is_trivially_copyable_v<ValueType>,
        "ValueType must be trivially copyable so std::bit_cast can construct it");
    TT_FATAL(
        sizeof(PackType) > sizeof(ValueType),
        "sizeof(PackType)={} > sizeof(ValueType))={}",
        sizeof(PackType),
        sizeof(ValueType));
    TT_FATAL(
        (sizeof(PackType) % sizeof(ValueType)) == 0,
        "sizeof(PackType)={} % sizeof(ValueType)={} must equal 0",
        sizeof(PackType),
        sizeof(ValueType));

    using bits_type = decltype(sized_unsigned.template operator()<sizeof(ValueType)>());
    constexpr unsigned int num_values_to_unpack = sizeof(PackType) / sizeof(ValueType);
    std::vector<ValueType> results = {};
    // Width-safe mask: shifting by sizeof(ValueType)*CHAR_BIT is UB when it
    // equals the width of the operand type (e.g. ValueType=uint32_t ⇒ shift by 32).
    constexpr PackType bitmask = static_cast<PackType>(std::numeric_limits<bits_type>::max());
    std::for_each(values.begin(), values.end(), [&](const PackType& value) {
        PackType current_value = value;
        for (unsigned j = 0; j < num_values_to_unpack; j++) {
            bits_type bits_of_value = static_cast<bits_type>(current_value & bitmask);
            results.push_back(std::bit_cast<ValueType>(bits_of_value));
            current_value = current_value >> (sizeof(ValueType) * CHAR_BIT);
        }
    });
    return results;
}

}  // namespace tt::test_utils
