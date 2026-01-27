#pragma once

#include <cmath>
#include <vector>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::block_variants {

template <typename ValueType>
float compute_pcc(const std::vector<ValueType>& a, const std::vector<ValueType>& b) {
    TT_FATAL(a.size() == b.size(), "PCC input sizes mismatch: {} vs {}", a.size(), b.size());
    if (a.empty()) {
        return 1.0f;
    }

    double mean_a = 0.0;
    double mean_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        mean_a += static_cast<double>(a[i]);
        mean_b += static_cast<double>(b[i]);
    }
    mean_a /= static_cast<double>(a.size());
    mean_b /= static_cast<double>(b.size());

    double cov = 0.0;
    double var_a = 0.0;
    double var_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double da = static_cast<double>(a[i]) - mean_a;
        const double db = static_cast<double>(b[i]) - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    const double denom = std::sqrt(var_a * var_b);
    if (denom == 0.0) {
        for (size_t i = 0; i < a.size(); ++i) {
            if (static_cast<float>(a[i]) != static_cast<float>(b[i])) {
                return 0.0f;
            }
        }
        return 1.0f;
    }

    return static_cast<float>(cov / denom);
}

}  // namespace tt::tt_metal::block_variants
