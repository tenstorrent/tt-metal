// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <random>

#include <tt-metalium/logger.hpp>

namespace tt {
namespace test_utils {

//! Generic Library of templated packing/unpacking functions.
//! Custom type is supported as long as the custom type supports the following custom functions
//! static SIZEOF - indicates byte size of custom type
//! to_float() - get float value from custom type
//! to_packed() - get packed (into an integral type that is of the bitwidth specified by SIZEOF)
//! Constructor(float in) - constructor with a float as the initializer
//! Constructor(uint32_t in) - constructor with a uint32_t as the initializer -- only lower bits needed

// Assumes ValueType has a .to_packed() function and static SIZEOF field
template <typename PackType, typename ValueType>
std::vector<PackType> pack_vector(const std::vector<ValueType>& values) {
    static_assert(
        std::is_integral<PackType>::value,
        "Packed Type must be an integral type we are packing to -- uint8_t/uint16_t/uint32_t...");
    TT_FATAL(
        sizeof(PackType) >= ValueType::SIZEOF,
        "sizeof(PackType)={} >= ValueType::SIZEOF)={}",
        sizeof(PackType),
        ValueType::SIZEOF);
    TT_FATAL(
        (sizeof(PackType) % ValueType::SIZEOF) == 0,
        "sizeof(PackType)={} % ValueType::SIZEOF={} must equal 0",
        sizeof(PackType),
        ValueType::SIZEOF);
    constexpr unsigned int num_values_to_pack = sizeof(PackType) / ValueType::SIZEOF;
    TT_FATAL(
        (values.size() % num_values_to_pack) == 0,
        "Number of values must evenly divide into the final packed type... no padding assumed");
    std::vector<PackType> results(values.size() / num_values_to_pack, 0);
    unsigned int index = 0;
    std::for_each(results.begin(), results.end(), [&](PackType& result) {
        for (unsigned j = 0; j < num_values_to_pack; j++) {
            result |= values[index].to_packed() << (j * ValueType::SIZEOF * CHAR_BIT);
            index++;
        }
        return result;
    });
    return results;
}

template <typename ValueType, typename PackType>
std::vector<ValueType> unpack_vector(const std::vector<PackType>& values) {
    static_assert(
        std::is_integral<PackType>::value,
        "Packed Type must be an integral type we are packing to -- uint8_t/uint16_t/uint32_t...");
    TT_FATAL(
        sizeof(PackType) > ValueType::SIZEOF,
        "sizeof(PackType)={} > ValueType::SIZEOF)={}",
        sizeof(PackType),
        ValueType::SIZEOF);
    TT_FATAL(
        (sizeof(PackType) % ValueType::SIZEOF) == 0,
        "sizeof(PackType)={} % ValueType::SIZEOF={} must equal 0",
        sizeof(PackType),
        ValueType::SIZEOF);
    constexpr unsigned int num_values_to_unpack = sizeof(PackType) / ValueType::SIZEOF;
    std::vector<ValueType> results = {};
    constexpr unsigned long bitmask = (1 << (ValueType::SIZEOF * CHAR_BIT)) - 1;
    std::for_each(values.begin(), values.end(), [&](const PackType& value) {
        PackType current_value = value;
        for (unsigned j = 0; j < num_values_to_unpack; j++) {
            results.push_back(ValueType(static_cast<uint32_t>(current_value & bitmask)));
            current_value = current_value >> (ValueType::SIZEOF * CHAR_BIT);
        }
    });
    return results;
}

}  // namespace test_utils
}  // namespace tt
