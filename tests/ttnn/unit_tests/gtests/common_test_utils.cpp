// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common_test_utils.hpp"

#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::test_utils {

float pcc(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vectors must be of the same length.");
    }
    int n = x.size();
    float mean_x = 0, mean_y = 0;
    for (int i = 0; i < n; ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    float numerator = 0, sum_sq_x = 0, sum_sq_y = 0;
    for (int i = 0; i < n; ++i) {
        float diff_x = x[i] - mean_x;
        float diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }

    float denominator = std::sqrt(sum_sq_x * sum_sq_y);
    if (denominator == 0) {
        return 0;
    }

    return numerator / denominator;
}

float relative_frobenius(
    const std::vector<float>& actual, const std::vector<float>& expected, bool& expected_norm_is_zero) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Vectors must be of the same length.");
    }
    // Use double accumulators: squaring + summing O(1e6) bf16-range values in float
    // loses precision quickly and can make the relative norm wander by several percent.
    double err_sq = 0.0;
    double expected_sq = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double d = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        err_sq += d * d;
        expected_sq += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
    }
    const double err_norm = std::sqrt(err_sq);
    const double expected_norm = std::sqrt(expected_sq);
    expected_norm_is_zero = (expected_norm == 0.0);
    return static_cast<float>(expected_norm_is_zero ? err_norm : (err_norm / expected_norm));
}

AllcloseReport allclose_report(
    const std::vector<float>& actual, const std::vector<float>& expected, float rtol, float atol) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("Vectors must be of the same length.");
    }
    AllcloseReport report;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float b = expected[i];
        const float diff = std::abs(a - b);
        const float tol = atol + rtol * std::abs(b);
        const float margin = diff - tol;

        if (margin > 0.0f) {
            ++report.failures;
        }
        if (diff > report.worst_atol_diff) {
            report.worst_atol_diff = diff;
            report.worst_atol_index = i;
            report.worst_atol_actual = a;
            report.worst_atol_expected = b;
        }
        // Skip near-zero expected values when computing relative error to avoid
        // division noise from denormal-like tiny references.
        if (std::abs(b) > 1e-6f) {
            const float rel = diff / std::abs(b);
            if (rel > report.worst_rtol_rel) {
                report.worst_rtol_rel = rel;
                report.worst_rtol_index = i;
                report.worst_rtol_diff = diff;
                report.worst_rtol_actual = a;
                report.worst_rtol_expected = b;
            }
        }
        if (margin > report.worst_margin) {
            report.worst_margin = margin;
            report.worst_margin_index = i;
            report.worst_margin_diff = diff;
            report.worst_margin_tol = tol;
            report.worst_margin_actual = a;
            report.worst_margin_expected = b;
        }
    }
    return report;
}

Tensor dispatch_ops_to_device(const Tensor& input_tensor, QueueId cq_id) {
    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    auto guard = ttnn::with_command_queue_id(cq_id);

    Tensor output_tensor = ttnn::mul_sfpu(input_tensor, 2);
    for (int i = 0; i < 3; i++) {
        output_tensor = ttnn::neg(output_tensor);
        output_tensor = ttnn::neg(output_tensor);
        output_tensor = ttnn::mul_sfpu(output_tensor, 2);
    }
    output_tensor = ttnn::neg(output_tensor);
    output_tensor = ttnn::mul_sfpu(output_tensor, 2);
    output_tensor = ttnn::add_sfpu(output_tensor, 128);

    return output_tensor;
}

}  // namespace ttnn::test_utils
