// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common_test_utils.hpp"

#include <cstddef>
#include <cmath>
#include <limits>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::test_utils {

float pcc(const std::vector<float>& x, const std::vector<float>& y) {
    EXPECT_EQ(x.size(), y.size()) << "pcc: input vectors must be the same length";
    if (x.size() != y.size()) {
        return 0.0f;
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
    EXPECT_EQ(actual.size(), expected.size()) << "relative_frobenius: input vectors must be the same length";
    if (actual.size() != expected.size()) {
        expected_norm_is_zero = false;
        return std::numeric_limits<float>::quiet_NaN();
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
    EXPECT_EQ(actual.size(), expected.size()) << "allclose_report: input vectors must be the same length";
    if (actual.size() != expected.size()) {
        return AllcloseReport{};
    }
    AllcloseReport report;
    constexpr float inf = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float b = expected[i];

        // Match torch.allclose(..., equal_nan=False) semantics for non-finite elements:
        //   - matching same-sign infinities are close (+inf vs +inf, -inf vs -inf);
        //   - any NaN is not close (NaN == anything is always false in IEEE 754);
        //   - mismatched infinities and inf-vs-finite are not close.
        // The regular tolerance formula below cannot handle these cases on its own:
        // (inf - inf) is NaN, NaN propagates silently through every `>` comparison
        // (NaN > x is always false), so without this guard a non-finite element would
        // never be counted as a failure or surface in the worst-element diagnostics.
        // For the failure cases we use an infinite sentinel diff/margin so the bad
        // element wins every "worst" slot, while preserving the original a/b values
        // in the diagnostic fields so the reader can see whether it was a NaN or an Inf.
        if (!std::isfinite(a) || !std::isfinite(b)) {
            if (a == b) {
                // +inf vs +inf or -inf vs -inf: torch.allclose treats these as close.
                // (NaN == NaN is always false, so NaNs cannot reach this branch.)
                continue;
            }
            ++report.failures;
            if (inf > report.worst_atol_diff) {
                report.worst_atol_diff = inf;
                report.worst_atol_index = i;
                report.worst_atol_actual = a;
                report.worst_atol_expected = b;
            }
            if (inf > report.worst_rtol_rel) {
                report.worst_rtol_rel = inf;
                report.worst_rtol_index = i;
                report.worst_rtol_diff = inf;
                report.worst_rtol_actual = a;
                report.worst_rtol_expected = b;
            }
            if (inf > report.worst_margin) {
                report.worst_margin = inf;
                report.worst_margin_index = i;
                report.worst_margin_diff = inf;
                report.worst_margin_tol = 0.0f;
                report.worst_margin_actual = a;
                report.worst_margin_expected = b;
            }
            continue;
        }

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
        const float rel = diff / std::abs(b);
        if (rel > report.worst_rtol_rel) {
            report.worst_rtol_rel = rel;
            report.worst_rtol_index = i;
            report.worst_rtol_diff = diff;
            report.worst_rtol_actual = a;
            report.worst_rtol_expected = b;
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

NonfiniteReport check_nonfinite_positions(const std::vector<float>& actual, const std::vector<float>& expected) {
    EXPECT_EQ(actual.size(), expected.size()) << "check_nonfinite_positions: input vectors must be the same length";
    if (actual.size() != expected.size()) {
        return NonfiniteReport{};
    }
    NonfiniteReport report;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float b = expected[i];
        const bool a_is_nan = std::isnan(a);
        const bool b_is_nan = std::isnan(b);
        const bool a_is_inf = std::isinf(a);
        const bool b_is_inf = std::isinf(b);

        if (a_is_nan || b_is_nan || a_is_inf || b_is_inf) {
            report.any_nonfinite = true;
        }

        // NaN positions must match exactly.
        if (a_is_nan != b_is_nan) {
            report.positions_match = false;
            report.first_mismatch_index = i;
            report.first_mismatch_actual = a;
            report.first_mismatch_expected = b;
            return report;
        }
        // Inf positions and signs must match exactly.  std::signbit picks +/- correctly
        // for both +inf and -inf, so comparing it only when both are Inf is sufficient.
        if (a_is_inf != b_is_inf || (a_is_inf && b_is_inf && std::signbit(a) != std::signbit(b))) {
            report.positions_match = false;
            report.first_mismatch_index = i;
            report.first_mismatch_actual = a;
            report.first_mismatch_expected = b;
            return report;
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
