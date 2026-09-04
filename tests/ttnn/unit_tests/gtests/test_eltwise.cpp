// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merge-gate smoke tests for the ttnn eltwise op family.
//
// Scope is deliberately narrow: this suite guards *infrastructure stability*, not numerical
// accuracy. Accuracy lives in the eltwise sanity and L2-nightly pytest suites, which are not
// triggered on a PR. What this catches is an incoming change that breaks the eltwise machinery
// outright -- a broadcast class that stops dispatching, a reader that computes the wrong stride,
// a work-split that drops cores -- before it reaches main and takes sanity down with it.
//
// The coverage axis is therefore the broadcast class, not the op:
//   - binary (add, multiply): no broadcast, outer-dim broadcast, row/col subtile broadcast,
//     scalar broadcast. These are the four shapes of SubtileBroadcastType plus the leading-dim
//     n_stride/c_stride path that sits underneath all of them.
//   - unary (abs, neg, exp): one shape. Unary has no broadcast axis; this is a liveness check
//     on the SFPU dispatch path.
//   - ternary (where, TTT): no broadcast, outer, mixed row+col, scalar -- TernaryBroadcastType
//     NONE / OUTER_BCAST / ROW_COL_BCAST / SCALAR_BCAST.
//
// Inputs are small deterministic integer ramps, chosen so every intermediate is exactly
// representable in bfloat16 and the golden is an exact equality rather than a tolerance. That
// matters here: a constant-valued input would pass even with a completely wrong stride, since
// every address holds the same number. The ramps differ along different axes per operand, so a
// misread lands on a different value and the test fails.
//
// exp is the one exception -- transcendental, so it compares with a tolerance.
//
// Runtime is a few seconds on an N300 including device bring-up.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tt_stl/small_vector.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "smoke_test_utils.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::eltwise::test {

class EltwiseSmoke : public TTNNFixtureWithSuiteDevice<EltwiseSmoke> {};

namespace detail {

using ttnn::test_utils::make_device_tensor;
using ttnn::test_utils::to_float_vector;

// An integer ramp: element i is (i % modulus) + offset. Values stay small so sums and products
// are exact in bfloat16, and the ramp varies along the fastest-moving axis so a stride error
// shows up as a wrong value rather than a coincidentally-equal one.
std::vector<float> ramp(size_t count, int modulus, int offset) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; i++) {
        out[i] = static_cast<float>(static_cast<int>(i % static_cast<size_t>(modulus)) + offset);
    }
    return out;
}

Tensor to_device_bf16(
    tt::tt_metal::distributed::MeshDevice& device, const ttnn::Shape& shape, const std::vector<float>& data) {
    std::vector<bfloat16> converted(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        converted[i] = bfloat16(data[i]);
    }
    return make_device_tensor(device, shape, converted, DataType::BFLOAT16, ttnn::TILE_LAYOUT);
}

// numpy/torch broadcasting: map a flat index in `out` to the flat index of the element of
// `in` that feeds it. `in` is right-aligned against `out`; a dim of 1 contributes nothing.
size_t broadcast_index(const ttnn::Shape& out, const ttnn::Shape& in, size_t flat_out) {
    const int out_rank = static_cast<int>(out.rank());
    const int in_rank = static_cast<int>(in.rank());
    size_t remainder = flat_out;
    size_t in_index = 0;
    size_t in_stride = 1;
    for (int d = out_rank - 1; d >= 0; d--) {
        const size_t coord = remainder % out[d];
        remainder /= out[d];
        const int in_dim = d - (out_rank - in_rank);
        if (in_dim >= 0) {
            const size_t extent = in[in_dim];
            in_index += (extent == 1 ? 0 : coord) * in_stride;
            in_stride *= extent;
        }
    }
    return in_index;
}

// Elementwise broadcast of the output shape: numpy rules, right-aligned.
ttnn::Shape broadcast_shape(const ttnn::Shape& a, const ttnn::Shape& b) {
    const int rank = std::max(static_cast<int>(a.rank()), static_cast<int>(b.rank()));
    ttsl::SmallVector<uint32_t> dims(rank, 1);
    for (int d = 0; d < rank; d++) {
        const int ad = d - (rank - static_cast<int>(a.rank()));
        const int bd = d - (rank - static_cast<int>(b.rank()));
        const uint32_t av = ad >= 0 ? a[ad] : 1;
        const uint32_t bv = bd >= 0 ? b[bd] : 1;
        dims[d] = std::max(av, bv);
    }
    return ttnn::Shape(dims);
}

size_t volume_of(const ttnn::Shape& s) {
    size_t v = 1;
    for (size_t d = 0; d < s.rank(); d++) {
        v *= s[d];
    }
    return v;
}

std::string describe(const ttnn::Shape& s) {
    std::string out = "[";
    for (size_t d = 0; d < s.rank(); d++) {
        out += (d ? "," : "") + std::to_string(s[d]);
    }
    return out + "]";
}

struct BinaryCase {
    const char* label;
    ttnn::Shape lhs;
    ttnn::Shape rhs;
};

// One case per broadcast class. Every combination here is exercised today by the eltwise pytest
// suites; the point of repeating them in C++ is that these run in the merge gate and those do not.
const std::vector<BinaryCase>& binary_cases() {
    static const std::vector<BinaryCase> cases = {
        // SubtileBroadcastType::NONE, no leading-dim broadcast: the plain path.
        {"no_bcast", ttnn::Shape({2, 3, 128, 128}), ttnn::Shape({2, 3, 128, 128})},
        // Leading-dim broadcast on both operands in opposite dims -- exercises the reader's
        // n_stride / c_stride arithmetic, which is where same-volume shapes diverge.
        {"outer_bcast", ttnn::Shape({3, 1, 128, 128}), ttnn::Shape({1, 3, 128, 128})},
        // SubtileBroadcastType::ROW_B_COL_A: lhs is a single tile column, rhs a single tile row.
        {"row_col_bcast", ttnn::Shape({2, 1, 128, 1}), ttnn::Shape({2, 1, 1, 128})},
        // SubtileBroadcastType::SCALAR_B, with a non-square output so H and W cannot be confused.
        {"scalar_bcast", ttnn::Shape({2, 1, 128, 256}), ttnn::Shape({2, 1, 1, 1})},
    };
    return cases;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Binary: ttnn::add and ttnn::multiply across the four broadcast classes.
// Covers binary_ng's reader stride computation and subtile-broadcast dispatch.
// ---------------------------------------------------------------------------

TEST_F(EltwiseSmoke, BinaryAddBroadcastClasses) {
    auto& device = *device_;
    for (const auto& c : detail::binary_cases()) {
        const auto out_shape = detail::broadcast_shape(c.lhs, c.rhs);
        const auto lhs_data = detail::ramp(detail::volume_of(c.lhs), 8, 1);
        const auto rhs_data = detail::ramp(detail::volume_of(c.rhs), 4, 1);

        const auto lhs = detail::to_device_bf16(device, c.lhs, lhs_data);
        const auto rhs = detail::to_device_bf16(device, c.rhs, rhs_data);
        const auto output = ttnn::add(lhs, rhs);

        ASSERT_EQ(output.logical_shape(), out_shape) << c.label;
        const auto result = detail::to_float_vector(output);
        ASSERT_EQ(result.size(), detail::volume_of(out_shape)) << c.label;
        for (size_t i = 0; i < result.size(); i++) {
            const float expected = lhs_data[detail::broadcast_index(out_shape, c.lhs, i)] +
                                   rhs_data[detail::broadcast_index(out_shape, c.rhs, i)];
            ASSERT_EQ(result[i], expected)
                << c.label << " " << detail::describe(c.lhs) << " + " << detail::describe(c.rhs) << " at flat index "
                << i;
        }
    }
}

TEST_F(EltwiseSmoke, BinaryMultiplyBroadcastClasses) {
    auto& device = *device_;
    for (const auto& c : detail::binary_cases()) {
        const auto out_shape = detail::broadcast_shape(c.lhs, c.rhs);
        const auto lhs_data = detail::ramp(detail::volume_of(c.lhs), 8, 1);
        const auto rhs_data = detail::ramp(detail::volume_of(c.rhs), 4, 1);

        const auto lhs = detail::to_device_bf16(device, c.lhs, lhs_data);
        const auto rhs = detail::to_device_bf16(device, c.rhs, rhs_data);
        const auto output = ttnn::multiply(lhs, rhs);

        ASSERT_EQ(output.logical_shape(), out_shape) << c.label;
        const auto result = detail::to_float_vector(output);
        ASSERT_EQ(result.size(), detail::volume_of(out_shape)) << c.label;
        for (size_t i = 0; i < result.size(); i++) {
            const float expected = lhs_data[detail::broadcast_index(out_shape, c.lhs, i)] *
                                   rhs_data[detail::broadcast_index(out_shape, c.rhs, i)];
            ASSERT_EQ(result[i], expected)
                << c.label << " " << detail::describe(c.lhs) << " * " << detail::describe(c.rhs) << " at flat index "
                << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Unary: liveness of the SFPU dispatch path. abs and neg are exact in bfloat16;
// exp is transcendental and compares with a tolerance.
// ---------------------------------------------------------------------------

TEST_F(EltwiseSmoke, UnaryAbsNegExp) {
    auto& device = *device_;
    const ttnn::Shape shape({2, 1, 128, 128});
    const size_t count = detail::volume_of(shape);

    // Straddles zero so abs and neg both have something to do.
    const auto signed_data = detail::ramp(count, 16, -8);
    const auto input = detail::to_device_bf16(device, shape, signed_data);

    {
        const auto result = detail::to_float_vector(ttnn::abs(input));
        ASSERT_EQ(result.size(), count);
        for (size_t i = 0; i < count; i++) {
            ASSERT_EQ(result[i], std::abs(signed_data[i])) << "abs at flat index " << i;
        }
    }
    {
        const auto result = detail::to_float_vector(ttnn::neg(input));
        ASSERT_EQ(result.size(), count);
        for (size_t i = 0; i < count; i++) {
            ASSERT_EQ(result[i], -signed_data[i]) << "neg at flat index " << i;
        }
    }
    {
        // Quarter steps over [-1, 1]: small enough that bfloat16 exp stays well inside tolerance.
        std::vector<float> exp_input = detail::ramp(count, 9, -4);
        for (auto& v : exp_input) {
            v /= 4.0f;
        }
        std::vector<float> expected(count);
        for (size_t i = 0; i < count; i++) {
            expected[i] = std::exp(exp_input[i]);
        }
        const auto result = detail::to_float_vector(ttnn::exp(detail::to_device_bf16(device, shape, exp_input)));
        ttnn::test_utils::expect_close(result, expected, /*rtol=*/0.02f, /*atol=*/0.02f);
    }
}

// ---------------------------------------------------------------------------
// Ternary: ttnn::where in its all-tensor (TTT) form across the broadcast classes.
// where is the op whose per-core strides are frozen into the program at cache-fill
// time, so a broken broadcast class here is the failure that reaches main quietest.
// ---------------------------------------------------------------------------

TEST_F(EltwiseSmoke, WhereTernaryBroadcastClasses) {
    auto& device = *device_;

    struct TernaryCase {
        const char* label;
        ttnn::Shape predicate;
        ttnn::Shape value_true;
        ttnn::Shape value_false;
    };

    const std::vector<TernaryCase> cases = {
        // TernaryBroadcastType::NONE
        {"no_bcast", ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 128, 128})},
        // OUTER_BCAST: value tensors broadcast over the leading dims only.
        {"outer_bcast", ttnn::Shape({3, 2, 128, 128}), ttnn::Shape({1, 1, 128, 128}), ttnn::Shape({3, 1, 128, 128})},
        // ROW_COL_BCAST: value_true broadcasts H, value_false broadcasts W -- the mixed case.
        {"row_col_bcast", ttnn::Shape({2, 2, 128, 128}), ttnn::Shape({1, 1, 1, 128}), ttnn::Shape({1, 1, 128, 1})},
        // SCALAR_BCAST: value_true is a single element per batch.
        {"scalar_bcast", ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 1, 1}), ttnn::Shape({2, 1, 128, 128})},
    };

    for (const auto& c : cases) {
        const auto out_shape =
            detail::broadcast_shape(detail::broadcast_shape(c.predicate, c.value_true), c.value_false);

        // Alternating 0/1, so both arms of the select are taken within every tile.
        const auto predicate_data = detail::ramp(detail::volume_of(c.predicate), 2, 0);
        // Disjoint value ranges: picking the wrong arm is unmistakable, not off by a rounding step.
        const auto true_data = detail::ramp(detail::volume_of(c.value_true), 16, 1);
        const auto false_data = detail::ramp(detail::volume_of(c.value_false), 16, 101);

        const auto predicate = detail::to_device_bf16(device, c.predicate, predicate_data);
        const auto value_true = detail::to_device_bf16(device, c.value_true, true_data);
        const auto value_false = detail::to_device_bf16(device, c.value_false, false_data);
        const auto output = ttnn::where(predicate, value_true, value_false);

        ASSERT_EQ(output.logical_shape(), out_shape) << c.label;
        const auto result = detail::to_float_vector(output);
        ASSERT_EQ(result.size(), detail::volume_of(out_shape)) << c.label;
        for (size_t i = 0; i < result.size(); i++) {
            const bool taken = predicate_data[detail::broadcast_index(out_shape, c.predicate, i)] != 0.0f;
            const float expected = taken ? true_data[detail::broadcast_index(out_shape, c.value_true, i)]
                                         : false_data[detail::broadcast_index(out_shape, c.value_false, i)];
            ASSERT_EQ(result[i], expected) << c.label << " at flat index " << i;
        }
    }
}

}  // namespace ttnn::operations::eltwise::test
