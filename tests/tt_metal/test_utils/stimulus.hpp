#pragma once
#include <algorithm>
#include <random>

#include "common/bfloat16.hpp"
#include "common/utils.hpp"

namespace tt {
namespace test_utils {
template <typename ValueType>
std::vector<ValueType> generate_uniform_random_vector(
    ValueType min, ValueType max, const size_t numel, const float seed = 0) {
    std::random_device rd;
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
    ValueType mean, ValueType stdev, const size_t numel, const float seed = 0) {
    std::random_device rd;
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
// Assumes ValueType has a .to_packed() function and static SIZEOF field
template <typename PackType, typename ValueType>
std::vector<PackType> pack_vector(const std::vector<ValueType>& values) {
    static_assert(
        std::is_integral<PackType>::value,
        "Packed Type must be an integral type we are packing to -- uint8_t/uint16_t/uint32_t...");
    tt::log_assert(
        sizeof(PackType) > ValueType::SIZEOF,
        "sizeof(PackType)={} > ValueType::SIZEOF)={}",
        sizeof(PackType),
        ValueType::SIZEOF);
    tt::log_assert(
        (sizeof(PackType) % ValueType::SIZEOF) == 0,
        "sizeof(PackType)={} % ValueType::SIZEOF={} must equal 0",
        sizeof(PackType),
        ValueType::SIZEOF);
    constexpr unsigned int num_values_to_pack = sizeof(PackType) / ValueType::SIZEOF;
    tt::log_assert(
        (values.size() % num_values_to_pack) == 0,
        "Number of values must evenly divide into the final packed type... no padding assumed");
    std::vector<PackType> results(values.size() / num_values_to_pack, 0);
    unsigned int index = 0;
    std::for_each(results.begin(), results.end(), [&](PackType& result) {
        for (unsigned j = 0; j < num_values_to_pack; j++) {
            result |= values[index].to_packed() << (index * ValueType::SIZEOF * CHAR_BIT);
            index++;
        }
        return result;
    });
    return results;
}

template <typename PackType, typename ValueType>
std::vector<PackType> generate_packed_uniform_random_vector(
    ValueType min, ValueType max, const size_t numel, const float seed = 0) {
    return pack_vector<PackType, ValueType>(generate_uniform_random_vector(min, max, numel, seed));
}

template <typename PackType, typename ValueType>
std::vector<PackType> generate_packed_normal_random_vector(
    ValueType mean, ValueType stdev, const size_t numel, const float seed = 0) {
    return pack_vector<PackType, ValueType>(generate_normal_random_vector(mean, stdev, numel, seed));
}

template <typename ValueType, typename PackType>
std::vector<ValueType> unpack_vector(const std::vector<PackType>& values) {
    static_assert(
        std::is_integral<PackType>::value,
        "Packed Type must be an integral type we are packing to -- uint8_t/uint16_t/uint32_t...");
    tt::log_assert(
        sizeof(PackType) > ValueType::SIZEOF,
        "sizeof(PackType)={} > ValueType::SIZEOF)={}",
        sizeof(PackType),
        ValueType::SIZEOF);
    tt::log_assert(
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
