// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared helpers for the per-op-family merge-gate smoke suites (test_matmul.cpp,
// test_normalization.cpp, test_reduction.cpp, ...).
//
// Kept separate from common_test_utils.hpp on purpose: that header is also included by the
// conv and CCL tests, and these helpers pull in gtest plus the device-tensor construction
// stack.

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "common_test_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::test_utils {

inline MemoryConfig dram_interleaved() {
    return MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
}

// Host vector -> device tensor with an explicit memory config (sharded or interleaved).
template <typename T>
Tensor make_device_tensor_mc(
    tt::tt_metal::distributed::MeshDevice& device,
    const ttnn::Shape& shape,
    const std::vector<T>& data,
    DataType dtype,
    Layout layout,
    const MemoryConfig& mem_cfg) {
    const tt::tt_metal::TensorLayout tensor_layout(dtype, tt::tt_metal::PageConfig(layout), mem_cfg);
    const tt::tt_metal::TensorSpec tensor_spec(shape, tensor_layout);
    return Tensor::from_vector(data, tensor_spec).to_device(&device, mem_cfg, ttnn::QueueId(0));
}

// Host vector -> DRAM-interleaved device tensor.
template <typename T>
Tensor make_device_tensor(
    tt::tt_metal::distributed::MeshDevice& device,
    const ttnn::Shape& shape,
    const std::vector<T>& data,
    DataType dtype,
    Layout layout) {
    return make_device_tensor_mc(device, shape, data, dtype, layout, dram_interleaved());
}

// Device tensor -> host float vector. bf16 needs an explicit element-wise
// conversion; block-float and fp32 tensors decode directly via to_vector<float>.
inline std::vector<float> to_float_vector(const Tensor& t) {
    if (t.dtype() == DataType::BFLOAT16) {
        const auto v = t.to_vector<bfloat16>();
        std::vector<float> out(v.size());
        std::transform(v.begin(), v.end(), out.begin(), [](bfloat16 x) { return static_cast<float>(x); });
        return out;
    }
    return t.to_vector<float>();
}

// Pass as pcc_min / frob_max to skip that aggregate check explicitly.
inline constexpr float kSkipCheck = -1.0f;

// Tolerance-based comparison. Always enforced: non-finite positions must match and every
// element must satisfy allclose(rtol, atol) -- an element-wise bound, strictly stronger at
// these tolerances than any aggregate metric. On top of that:
//
// - PCC >= pcc_min, on by default (0.99). Skipped automatically when either vector is
//   constant: correlation is undefined there (pcc() returns 0), and constant closed-form
//   goldens are this suite's main technique, so a correct result would otherwise fail.
//   Pass kSkipCheck to opt out explicitly, or a higher value to tighten.
// - Relative Frobenius error <= frob_max, off by default (kSkipCheck). No universal default
//   is sound: for the exact-zero goldens used throughout, relative_frobenius() falls back to
//   the ABSOLUTE norm, which grows with sqrt(N) -- any fixed threshold would fail correct
//   results at some tensor size. Opt in per case where rtol/atol are loose.
//
// Note the ASSERT_* macros below return from this helper, not from the calling test.
inline void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float rtol,
    float atol,
    float pcc_min = 0.99f,
    float frob_max = kSkipCheck) {
    ASSERT_EQ(actual.size(), expected.size());
    const NonfiniteReport nf = check_nonfinite_positions(actual, expected);
    ASSERT_TRUE(nf.positions_match) << "non-finite mismatch at flat index " << nf.first_mismatch_index
                                    << ": device=" << nf.first_mismatch_actual << " ref=" << nf.first_mismatch_expected;
    const AllcloseReport report = allclose_report(actual, expected, rtol, atol);
    EXPECT_EQ(report.failures, 0u) << report.failures << " element(s) failed allclose(rtol=" << rtol
                                   << ", atol=" << atol << "); worst: flat index " << report.worst_margin_index
                                   << " device=" << report.worst_margin_actual
                                   << " ref=" << report.worst_margin_expected << " diff=" << report.worst_margin_diff
                                   << " tol=" << report.worst_margin_tol;
    if (nf.any_nonfinite) {
        return;  // pcc / relative_frobenius NaN-poison on non-finite inputs
    }
    if (pcc_min >= 0.0f) {
        // PCC is undefined for constant vectors (pcc() returns 0); a deviation from a constant
        // golden is already caught by the mandatory element-wise allclose above.
        const auto [amin, amax] = std::minmax_element(actual.begin(), actual.end());
        const auto [emin, emax] = std::minmax_element(expected.begin(), expected.end());
        const bool degenerate = actual.empty() || *amin == *amax || *emin == *emax;
        if (!degenerate) {
            const float p = pcc(actual, expected);
            EXPECT_GE(p, pcc_min);
        }
    }
    if (frob_max >= 0.0f) {
        bool expected_norm_is_zero = false;
        const float f = relative_frobenius(actual, expected, expected_norm_is_zero);
        EXPECT_LE(f, frob_max) << (expected_norm_is_zero ? "absolute" : "relative") << " Frobenius error over limit";
    }
}

}  // namespace ttnn::test_utils
