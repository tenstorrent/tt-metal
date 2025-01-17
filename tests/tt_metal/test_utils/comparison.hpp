// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/logger.hpp>
#include "tt_metal/test_utils/packing.hpp"

namespace tt {
namespace test_utils {

//! Generic Library of templated comparison functions.
//! Custom type is supported as long as the custom type supports the following custom functions
//! static SIZEOF - indicates byte size of custom type
//! to_float() - get float value from custom type
//! to_packed() - get packed (into an integral type that is of the bitwidth specified by SIZEOF)
//! Constructor(float in) - constructor with a float as the initializer
//! Constructor(uint32_t in) - constructor with a uint32_t as the initializer -- only lower bits needed
//
// this follows the implementation of numpy's is_close
template <typename ValueType>
bool is_close(const ValueType a, const ValueType b, float rtol = 0.01f, float atol = 0.001f) {
    float af = 0.0f;
    float bf = 0.0f;
    if constexpr (std::is_integral<ValueType>::value or std::is_floating_point<ValueType>::value) {
        af = static_cast<float>(a);
        bf = static_cast<float>(b);
    } else {
        af = a.to_float();
        bf = b.to_float();
    }
    // the idea is near zero we want absolute tolerance since relative doesn't make sense
    // (consider 1e-6f and 1.1e-6f)
    // elsewhere (not near zero) we want relative tolerance
    auto absdiff = fabsf(af - bf);
    auto reldenom = fmaxf(fabsf(af), fabsf(bf));
    auto result = (absdiff <= atol) || (absdiff <= rtol * reldenom);
    if (result != true) {
        tt::log_error(tt::LogTest, "Discrepacy: A={}, B={}", af, bf);
        tt::log_error(tt::LogTest, "   absdiff={}, atol={}", absdiff, atol);
        tt::log_error(tt::LogTest, "   reldiff={}, rtol={}", absdiff / (reldenom + 1e-6f), rtol);
    }
    return result;
}
template <typename ValueType>
bool is_close_vectors(
    const std::vector<ValueType>& vec_a,
    const std::vector<ValueType>& vec_b,
    std::function<bool(ValueType, ValueType)> comparison_function,
    int* argfail = nullptr) {
    TT_FATAL(
        vec_a.size() == vec_b.size(),
        "is_close_vectors -- vec_a.size()={} == vec_b.size()={}",
        vec_a.size(),
        vec_b.size());

    for (unsigned int i = 0; i < vec_a.size(); i++) {
        if (not comparison_function(vec_a.at(i), vec_b.at(i))) {
            if (argfail) {
                *argfail = i;
            }
            return false;
        }
    }
    return true;
}

template <typename ValueType, typename PackType>
bool is_close_packed_vectors(
    const std::vector<PackType>& vec_a,
    const std::vector<PackType>& vec_b,
    std::function<bool(ValueType, ValueType)> comparison_function,
    int* argfail = nullptr) {
    return is_close_vectors(
        unpack_vector<ValueType, PackType>(vec_a),
        unpack_vector<ValueType, PackType>(vec_b),
        comparison_function,
        argfail);
}

}  // namespace test_utils
}  // namespace tt
