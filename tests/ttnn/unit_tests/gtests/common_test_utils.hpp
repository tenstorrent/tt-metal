// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::test_utils {

// Pearson Correlation Coefficient for two float vectors.
// Both vectors must be the same size; if not, records a non-fatal gtest failure
// (EXPECT_EQ) and returns 0.0f without computing a correlation.
float pcc(const std::vector<float>& x, const std::vector<float>& y);

// Relative Frobenius norm of the difference: ||actual - expected||_F / ||expected||_F.
// Matches tests/ttnn/utils_for_testing.py::comp_relative_frobenius semantics: when
// ||expected||_F is zero, falls back to the absolute Frobenius error and reports that
// via the `expected_norm_is_zero` out-parameter so callers can word failure messages
// appropriately ("Absolute" vs "Relative" Frobenius error).
// Both vectors must be the same size; if not, records a non-fatal gtest failure
// (EXPECT_EQ), sets `expected_norm_is_zero` to false, and returns NaN without computing
// a Frobenius norm.
float relative_frobenius(
    const std::vector<float>& actual, const std::vector<float>& expected, bool& expected_norm_is_zero);

// Summary of an element-wise comparison between `actual` and `expected`.
//
// In addition to the failure count it records the worst elements by absolute error,
// relative error and allclose margin so failures can be diagnosed without re-running
// the test.  All `*_index` fields are flat offsets into the input vectors; callers
// that want row/col coordinates should divide/modulo by the row stride themselves.
struct AllcloseReport {
    // Number of elements that violate |actual[i] - expected[i]| <= atol + rtol * |expected[i]|.
    std::size_t failures = 0;

    // Worst element by absolute error |actual[i] - expected[i]|.
    std::size_t worst_atol_index = 0;
    float worst_atol_diff = 0.0f;
    float worst_atol_actual = 0.0f;
    float worst_atol_expected = 0.0f;

    // Worst element by relative error |actual[i] - expected[i]| / |expected[i]|.
    std::size_t worst_rtol_index = 0;
    float worst_rtol_rel = 0.0f;
    float worst_rtol_diff = 0.0f;
    float worst_rtol_actual = 0.0f;
    float worst_rtol_expected = 0.0f;

    // Worst element by allclose margin: (|actual[i] - expected[i]|) - (atol + rtol * |expected[i]|).
    // Positive margin means the element failed allclose; reporting this identifies the element
    // most responsible for the failure.
    std::size_t worst_margin_index = 0;
    float worst_margin = -std::numeric_limits<float>::infinity();
    float worst_margin_diff = 0.0f;
    float worst_margin_tol = 0.0f;
    float worst_margin_actual = 0.0f;
    float worst_margin_expected = 0.0f;
};

// Compute an AllcloseReport comparing `actual` against `expected` using
// torch.allclose(..., equal_nan=False) semantics:
//   - finite elements are close iff |a - b| <= atol + rtol * |b|;
//   - matching same-sign infinities (+inf vs +inf, -inf vs -inf) are close;
//   - any NaNs, mismatched infinities (+inf vs -inf), and inf-vs-finite are not close.
// When a non-finite element is counted as a failure, it dominates every "worst"
// slot (absolute, relative and margin) using infinite sentinel diff/margin values,
// while the original `actual`/`expected` values are preserved in the diagnostic
// fields so the reader can tell whether it was a NaN or an Inf.
// Both vectors must be the same size; if not, records a non-fatal gtest failure
// (EXPECT_EQ) and returns a default-constructed AllcloseReport (failures = 0, all
// worst-element fields at their defaults) without inspecting any element.
AllcloseReport allclose_report(
    const std::vector<float>& actual, const std::vector<float>& expected, float rtol, float atol);

// Result of a non-finite position check between two vectors.  Records whether the
// positions and signs of NaN/Inf elements agree, and the first index where they
// don't when they don't.
//
// `positions_match` is true iff:
//   - NaN appears at the same set of indices in both vectors, AND
//   - Inf appears at the same set of indices in both vectors, AND
//   - Inf signs agree at those indices.
// When false, the `first_mismatch_*` fields describe the first index where the
// disagreement occurs.
//
// See `check_nonfinite_positions` for the producer and the recommended caller pattern.
struct NonfiniteReport {
    bool positions_match = true;
    // Whether either vector has any non-finite element (NaN or Inf).  When this is false,
    // both vectors are all-finite and `positions_match` is trivially true.
    bool any_nonfinite = false;
    // First index where the non-finite-position rule above is violated.  Unspecified when
    // `positions_match` is true.
    std::size_t first_mismatch_index = 0;
    float first_mismatch_actual = 0.0f;
    float first_mismatch_expected = 0.0f;
};

// Pre-check for unexpected NaN/Inf values between `actual` and `expected`.  Mirrors
// models/common/utility_functions.py::_comp_nonfinite.
// Both vectors must be the same size; if not, records a non-fatal gtest failure
// (EXPECT_EQ) and returns a default-constructed NonfiniteReport (positions_match = true,
// any_nonfinite = false) without inspecting any element.
//
// Run before tolerance/correlation metrics (allclose_report, pcc, relative_frobenius)
// so that an unexpected NaN/Inf produces a clear, dedicated failure. Examples:
//   - `positions_match == false` is a more direct diagnosis than what allclose_report
//     would produce: allclose_report will count a finite-vs-Inf element as a tolerance
//     failure and report it via the "worst" slots, but the message reads like an ordinary
//     out-of-tolerance miss rather than "the device produced a non-finite value here";
//   - `any_nonfinite == true` lets callers skip pcc / relative_frobenius entirely.  Those
//     metrics NaN-poison on any non-finite input (e.g. pcc evaluates inf - inf = NaN)
//     even when the non-finites match positions and signs, so once `any_nonfinite` is true
//     their numeric output is unreliable.
NonfiniteReport check_nonfinite_positions(const std::vector<float>& actual, const std::vector<float>& expected);

// Dispatches a series of elementwise arithmetic operations over a tensor to `cq_id`, according to the expression:
// `output_tensor = - 32 * (input_tensor) + 128`
Tensor dispatch_ops_to_device(const Tensor& input_tensor, QueueId cq_id);

}  // namespace ttnn::test_utils
