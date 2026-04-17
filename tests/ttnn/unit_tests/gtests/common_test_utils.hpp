// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::test_utils {

// Pearson Correlation Coefficient for two float vectors
float pcc(const std::vector<float>& x, const std::vector<float>& y);

// Relative Frobenius norm of the difference: ||actual - expected||_F / ||expected||_F.
// Matches tests/ttnn/utils_for_testing.py::comp_relative_frobenius semantics: when
// ||expected||_F is zero, falls back to the absolute Frobenius error and reports that
// via the `expected_norm_is_zero` out-parameter so callers can word failure messages
// appropriately ("Absolute" vs "Relative" Frobenius error).
float relative_frobenius(
    const std::vector<float>& actual, const std::vector<float>& expected, bool& expected_norm_is_zero);

// Summary of an element-wise comparison between `actual` and `expected`, using
// torch.allclose semantics: |actual[i] - expected[i]| <= atol + rtol * |expected[i]|.
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

    // Worst element by relative error |actual[i] - expected[i]| / |expected[i]| among
    // elements whose expected value is not near zero (|expected[i]| > 1e-6).
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

// Compute an AllcloseReport comparing `actual` against `expected` with torch.allclose
// tolerance semantics (|a - b| <= atol + rtol * |b|).  Both vectors must be the same size.
AllcloseReport allclose_report(
    const std::vector<float>& actual, const std::vector<float>& expected, float rtol, float atol);

// Dispatches a series of elementwise arithmetic operations over a tensor to `cq_id`, according to the expression:
// `output_tensor = - 32 * (input_tensor) + 128`
Tensor dispatch_ops_to_device(const Tensor& input_tensor, QueueId cq_id);

}  // namespace ttnn::test_utils
