// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <random>

#include <tt-metalium/logger.hpp>
#include "tt_metal/test_utils/packing.hpp"

namespace tt {
namespace test_utils {

//! Generic Library of templated stimulus generation + packing/unpacking functions.
//! Custom type is supported as long as the custom type supports the following custom functions
//! static SIZEOF - indicates byte size of custom type
//! to_float() - get float value from custom type
//! to_packed() - get packed (into an integral type that is of the bitwidth specified by SIZEOF)
//! Constructor(float in) - constructor with a float as the initializer
//! Constructor(uint32_t in) - constructor with a uint32_t as the initializer -- only lower bits needed

// Setup a vector as follows:
// For the following offsets, corresponding values below
// [   0,    1, ...   offset, offset + 1, ... offset + stride, offset + stride + 1, ...]
// [init, init, ... assigned,       init, ...        assigned,                init, ...]
template <typename ValueType>
std::vector<ValueType> generate_strided_vector(
    const ValueType& init, const ValueType& assigned, const size_t& stride, const size_t& offset, const size_t& numel) {
    std::vector<ValueType> results(numel, init);
    for (unsigned int index = offset; index < numel; index = index + stride) {
        results.at(index) = assigned;
    }
    return results;
}

template <typename ValueType>
std::vector<ValueType> generate_constant_vector(const ValueType& constant, const size_t& numel) {
    std::vector<ValueType> results(numel, constant);
    return results;
}

template <typename ValueType>
std::vector<ValueType> generate_increment_vector(
    const ValueType& init,
    const size_t& numel,
    const float increment = 1.0,
    const float start = 0.0,
    const int count = 16,
    const bool slide = true) {
    std::vector<ValueType> results(numel, init);
    float start_value = start;
    float value = start_value;
    for (unsigned int index = 0; index < numel; ++index) {
        if (index % count == 0 && index > 0) {
            if (slide) {
                start_value += increment;
            }
            value = start_value;
        }
        results.at(index) = value;
        value += increment;
    }
    return results;
}

template <typename ValueType>
std::vector<ValueType> generate_uniform_random_vector(
    ValueType min, ValueType max, const size_t numel, const uint32_t seed = 0) {
    std::mt19937 gen(seed);
    std::vector<ValueType> results(numel);
    if constexpr (std::is_integral<ValueType>::value) {
        std::uniform_int_distribution<ValueType> dis(min, max);
        std::generate(results.begin(), results.end(), [&]() { return dis(gen); });
    } else if constexpr (std::is_floating_point<ValueType>::value) {
        std::uniform_real_distribution<ValueType> dis(min, max);
        std::generate(results.begin(), results.end(), [&]() { return dis(gen); });
    } else {
        std::uniform_real_distribution<float> dis(min.to_float(), max.to_float());
        std::generate(results.begin(), results.end(), [&]() { return ValueType(dis(gen)); });
    }
    return results;
}

template <typename ValueType>
std::vector<ValueType> generate_normal_random_vector(
    ValueType mean, ValueType stdev, const size_t numel, const uint32_t seed = 0) {
    std::mt19937 gen(seed);
    std::vector<ValueType> results(numel);
    if constexpr (std::is_integral<ValueType>::value or std::is_floating_point<ValueType>::value) {
        std::normal_distribution<ValueType> dis(mean, stdev);
        std::generate(results.begin(), results.end(), [&]() { return dis(gen); });
    } else {
        std::normal_distribution<float> dis(mean.to_float(), stdev.to_float());
        std::generate(results.begin(), results.end(), [&]() { return ValueType(dis(gen)); });
    }
    return results;
}

// Will randomize values in the generated vector from the input vector
template <typename ValueType>
std::vector<ValueType> generate_random_vector_from_vector(
    std::vector<ValueType>& possible_values, const size_t numel, const uint32_t seed = 0) {
    TT_FATAL(possible_values.size(), "possible_values.size()={} > 0", possible_values.size());
    std::mt19937 gen(seed);
    std::vector<ValueType> results(numel);
    std::uniform_int_distribution<unsigned int> dis(0, possible_values.size() - 1);
    std::generate(results.begin(), results.end(), [&]() { return possible_values.at(dis(gen)); });
    return results;
}

template <typename PackType, typename ValueType>
std::vector<PackType> generate_packed_uniform_random_vector(
    ValueType min, ValueType max, const size_t numel, const uint32_t seed = 0) {
    return pack_vector<PackType, ValueType>(generate_uniform_random_vector(min, max, numel, seed));
}

template <typename PackType, typename ValueType>
std::vector<PackType> generate_packed_normal_random_vector(
    ValueType mean, ValueType stdev, const size_t numel, const uint32_t seed = 0) {
    return pack_vector<PackType, ValueType>(generate_normal_random_vector(mean, stdev, numel, seed));
}

template <typename PackType, typename ValueType>
std::vector<PackType> generate_packed_random_vector_from_vector(
    std::vector<ValueType>& possible_values, const size_t numel, const uint32_t seed = 0) {
    return pack_vector<PackType, ValueType>(generate_random_vector_from_vector(possible_values, numel, seed));
}

template <typename PackType, typename ValueType>
std::vector<PackType> generate_packed_strided_vector(
    const ValueType& init, const ValueType& assigned, const size_t& stride, const size_t& offset, const size_t& numel) {
    return pack_vector<PackType, ValueType>(generate_strided_vector(init, assigned, stride, offset, numel));
}

template <typename PackType, typename ValueType>
std::vector<PackType> generate_packed_constant_vector(const ValueType& constant, const size_t& numel) {
    return pack_vector<PackType, ValueType>(generate_constant_vector(constant, numel));
}

template <typename PackType, typename ValueType>
std::vector<PackType> generate_packed_increment_vector(
    const ValueType& init,
    const size_t& numel,
    float increment = 1.0,
    float start = 0.0,
    int count = 16,
    bool slide = true) {
    return pack_vector<PackType, ValueType>(generate_increment_vector(init, numel, increment, start, count, slide));
}

}  // namespace test_utils
}  // namespace tt
