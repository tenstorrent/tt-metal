// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include "tt_metal/test_utils/packing.hpp"

namespace tt::test_utils {

//! Generic Library of templated comparison functions.
//! Custom ValueType is supported as long as it is convertible to float (used
//! by is_close to compute absolute / relative error). The unpack-and-compare
//! helpers additionally require trivial-copyability for std::bit_cast through
//! the byte-shuffling layer in packing.hpp.
//
// this follows the implementation of numpy's is_close
template <typename ValueType>
bool is_close(const ValueType a, const ValueType b, float rtol = 0.01f, float atol = 0.001f) {
    auto af = static_cast<float>(a);
    auto bf = static_cast<float>(b);
    // the idea is near zero we want absolute tolerance since relative doesn't make sense
    // (consider 1e-6f and 1.1e-6f)
    // elsewhere (not near zero) we want relative tolerance
    auto absdiff = fabsf(af - bf);
    auto reldenom = fmaxf(fabsf(af), fabsf(bf));
    auto result = (absdiff <= atol) || (absdiff <= rtol * reldenom);
    if (!result) {
        log_error(tt::LogTest, "Discrepacy: A={}, B={}", af, bf);
        log_error(tt::LogTest, "   absdiff={}, atol={}", absdiff, atol);
        log_error(tt::LogTest, "   reldiff={}, rtol={}", absdiff / (reldenom + 1e-6f), rtol);
    }
    return result;
}
template <typename ValueType>
bool is_close_vectors(
    const std::vector<ValueType>& vec_a,
    const std::vector<ValueType>& vec_b,
    const std::function<bool(ValueType, ValueType)>& comparison_function,
    int* argfail = nullptr) {
    TT_FATAL(
        vec_a.size() == vec_b.size(),
        "is_close_vectors -- vec_a.size()={} == vec_b.size()={}",
        vec_a.size(),
        vec_b.size());

    auto it = std::mismatch(vec_a.begin(), vec_a.end(), vec_b.begin(), comparison_function);
    if (it.first != vec_a.end()) {
        if (argfail) {
            *argfail = static_cast<int>(std::distance(vec_a.begin(), it.first));
        }
        return false;
    }
    return true;
}

template <typename ValueType, typename PackType>
bool is_close_packed_vectors(
    const std::vector<PackType>& vec_a,
    const std::vector<PackType>& vec_b,
    const std::function<bool(ValueType, ValueType)>& comparison_function,
    int* argfail = nullptr) {
    return is_close_vectors(
        unpack_vector<ValueType, PackType>(vec_a),
        unpack_vector<ValueType, PackType>(vec_b),
        comparison_function,
        argfail);
}

// Pearson correlation coefficient between two equally-sized float vectors.
// Empty inputs are treated as a failure (caller has nothing to validate against).
// Zero-variance handling: PCC is mathematically undefined when either input is
// constant. We pass only when *both* sides are constant AND element-wise equal
// (so constant-input identity checks still work); a constant device output
// against a varying golden — the failure mode for FP8 corruption that saturates
// to a single value — must not be silently accepted.
inline bool check_pcc(const std::vector<float>& a, const std::vector<float>& b, double min_pcc) {
    TT_FATAL(a.size() == b.size(), "check_pcc -- a.size()={} == b.size()={}", a.size(), b.size());

    const std::size_t n = a.size();
    if (n == 0) {
        log_error(tt::LogTest, "check_pcc: empty inputs — nothing to validate, returning false");
        return false;
    }
    double sum_a = 0.0, sum_b = 0.0, sum_a2 = 0.0, sum_b2 = 0.0, sum_ab = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        double ai = a[i], bi = b[i];
        sum_a += ai;
        sum_b += bi;
        sum_a2 += ai * ai;
        sum_b2 += bi * bi;
        sum_ab += ai * bi;
    }
    const double denom_a = (n * sum_a2) - (sum_a * sum_a);
    const double denom_b = (n * sum_b2) - (sum_b * sum_b);
    if (denom_a == 0.0 && denom_b == 0.0) {
        if (a == b) {
            return true;
        }
        log_error(tt::LogTest, "check_pcc: both inputs are constant but unequal — a[0]={}, b[0]={}", a[0], b[0]);
        return false;
    }
    if (denom_a == 0.0 || denom_b == 0.0) {
        log_error(
            tt::LogTest,
            "check_pcc: one input is constant while the other varies (a const={}, b const={}) — PCC is undefined",
            denom_a == 0.0,
            denom_b == 0.0);
        return false;
    }
    const double pcc = (n * sum_ab - sum_a * sum_b) / std::sqrt(denom_a * denom_b);

    if (pcc < min_pcc) {
        log_error(tt::LogTest, "check_pcc: PCC = {} < min_pcc = {}", pcc, min_pcc);
        return false;
    }
    return true;
}

}  // namespace tt::test_utils
