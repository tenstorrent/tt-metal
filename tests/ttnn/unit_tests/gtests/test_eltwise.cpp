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
//   - binary (add, multiply): one shape pair per SubtileBroadcastType -- NONE, SCALAR_A,
//     SCALAR_B, ROW_A_COL_B, ROW_B_COL_A, ROW_A, ROW_B, COL_A, COL_B -- plus the leading-dim
//     n_stride/c_stride path that sits underneath all of them. Each of those classes selects a
//     different reader kernel or compile-time define in binary_ng_program_factory.cpp, so one
//     dispatching says nothing about the next.
//   - unary (abs, neg, exp): one shape. Unary has no broadcast axis; this is a liveness check
//     on the SFPU dispatch path.
//   - ternary (where, TTT): one case per TernaryBroadcastType that has a TTT kernel triple in
//     ternary_op_utils.cpp -- NONE, OUTER_BCAST, ROW_BCAST, COL_BCAST, ROW_COL_BCAST and
//     SCALAR_BCAST. SCALAR_A_BCAST / SCALAR_B_BCAST are TTS/TST-only and are covered by the
//     tensor-scalar where tests in the pytest half of the gate.
//
// Every dim in the shape tables below is either 1 or a multiple of 32. Non-tile-aligned and
// padded shapes are deliberately out of scope here: padding behaviour is an accuracy property
// and it is covered by the nightly eltwise pytest suites. This suite only asks whether the
// class dispatches and reads the right addresses.
//
// Inputs are small deterministic coordinate-derived ramps, chosen so every intermediate is
// exactly representable in bfloat16 and the golden is an exact equality rather than a tolerance.
// That matters here: a constant-valued input would pass even with a completely wrong stride,
// since every address holds the same number. See coordinate_ramp below for why a flat i % m
// ramp is not enough either.
//
// exp is the one exception -- transcendental, so it compares with a tolerance.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// Odd, pairwise-distinct per-axis weights, indexed from the innermost axis outwards. Odd so they
// stay coprime with the power-of-two moduli used below; distinct so an H/W transposition on a
// square plane produces a different value instead of the same sum.
constexpr uint32_t axis_weight(size_t axis_from_innermost) {
    constexpr std::array<uint32_t, 6> kWeights = {1, 7, 5, 3, 11, 13};
    return axis_from_innermost < kWeights.size() ? kWeights[axis_from_innermost] : 15;
}

// A coordinate-derived integer ramp: the value at coordinates (c0 .. cN-1) is
//   ((sum over d of axis_weight(d) * coord(d)) % modulus) + offset
//
// The value depends on *every* coordinate, so a misread on any axis -- a wrong n_stride or
// c_stride landing on a different leading plane, a wrong row, a wrong column -- lands on a
// different value. A flat `i % modulus` ramp does not have that property: it repeats every
// `modulus` elements, so with 16384-element planes every N/C plane holds identical data and a
// wrong-plane read still returns the expected number, which is exactly the regression these
// cases exist to catch.
//
// Values stay in [offset, offset + modulus - 1]. Keeping the range small is what makes the
// goldens exact: bfloat16 has 8 significant bits, so the sums and products formed here are
// representable without rounding and the comparisons can be equality rather than tolerance.
std::vector<float> coordinate_ramp(const ttnn::Shape& shape, int modulus, int offset) {
    const auto count = static_cast<size_t>(shape.volume());
    const int rank = static_cast<int>(shape.rank());
    std::vector<float> out(count);
    for (size_t i = 0; i < count; i++) {
        size_t remainder = i;
        uint32_t weighted = 0;
        for (int d = rank - 1; d >= 0; d--) {
            const auto extent = static_cast<size_t>(shape[d]);
            weighted += axis_weight(static_cast<size_t>(rank - 1 - d)) * static_cast<uint32_t>(remainder % extent);
            remainder /= extent;
        }
        out[i] = static_cast<float>(static_cast<int>(weighted % static_cast<uint32_t>(modulus)) + offset);
    }
    return out;
}

Tensor to_device_bf16(
    tt::tt_metal::distributed::MeshDevice& device, const ttnn::Shape& shape, const std::vector<float>& data) {
    std::vector<bfloat16> converted(data.size());
    std::transform(data.begin(), data.end(), converted.begin(), [](float v) { return bfloat16(v); });
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

// One case per SubtileBroadcastType, in the order get_subtile_broadcast_type() tests them
// (binary_ng_device_operation.cpp), plus the leading-dim case. Every combination here is also
// exercised by the eltwise pytest suites; the point of repeating them in C++ is that these run
// in the merge gate and those do not.
const std::vector<BinaryCase>& binary_cases() {
    static const std::vector<BinaryCase> cases = {
        // NONE, no leading-dim broadcast: the plain path.
        {"no_bcast", ttnn::Shape({2, 3, 128, 128}), ttnn::Shape({2, 3, 128, 128})},
        // NONE subtile, but broadcasting on both operands in opposite leading dims -- exercises
        // the reader's n_stride / c_stride arithmetic, which is where same-volume shapes diverge.
        {"outer_bcast", ttnn::Shape({3, 1, 128, 128}), ttnn::Shape({1, 3, 128, 128})},
        // SCALAR_A / SCALAR_B: one operand is a single element per batch. Non-square output on
        // the SCALAR_B case so H and W cannot be confused.
        {"scalar_a", ttnn::Shape({2, 1, 1, 1}), ttnn::Shape({2, 1, 128, 128})},
        {"scalar_b", ttnn::Shape({2, 1, 128, 256}), ttnn::Shape({2, 1, 1, 1})},
        // ROW_A_COL_B / ROW_B_COL_A: one operand is a single tile row, the other a single column.
        {"row_a_col_b", ttnn::Shape({2, 1, 1, 128}), ttnn::Shape({2, 1, 128, 1})},
        {"row_b_col_a", ttnn::Shape({2, 1, 128, 1}), ttnn::Shape({2, 1, 1, 128})},
        // ROW_A / ROW_B: one operand broadcasts H against a full-shape operand.
        {"row_a", ttnn::Shape({2, 1, 1, 128}), ttnn::Shape({2, 1, 128, 128})},
        {"row_b", ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 1, 128})},
        // COL_A / COL_B: one operand broadcasts W against a full-shape operand.
        {"col_a", ttnn::Shape({2, 1, 128, 1}), ttnn::Shape({2, 1, 128, 128})},
        {"col_b", ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 128, 1})},
    };
    return cases;
}

// Shared body for the binary broadcast suites: same setup, same shape iteration, same
// assertions; only the op and its closed-form golden differ.
template <typename Op, typename ExpectedFn>
void run_binary_broadcast_cases(
    tt::tt_metal::distributed::MeshDevice& device, Op op, ExpectedFn expected_fn, const char* op_symbol) {
    for (const auto& c : binary_cases()) {
        const auto out_shape = broadcast_shape(c.lhs, c.rhs);
        // Different moduli per operand, so lhs and rhs are never accidentally interchangeable.
        const auto lhs_data = coordinate_ramp(c.lhs, 8, 1);
        const auto rhs_data = coordinate_ramp(c.rhs, 4, 1);

        const auto lhs = to_device_bf16(device, c.lhs, lhs_data);
        const auto rhs = to_device_bf16(device, c.rhs, rhs_data);
        const auto output = op(lhs, rhs);

        ASSERT_EQ(output.logical_shape(), out_shape) << c.label;
        const auto result = to_float_vector(output);
        ASSERT_EQ(result.size(), static_cast<size_t>(out_shape.volume())) << c.label;
        for (size_t i = 0; i < result.size(); i++) {
            const float expected = expected_fn(
                lhs_data[broadcast_index(out_shape, c.lhs, i)], rhs_data[broadcast_index(out_shape, c.rhs, i)]);
            ASSERT_EQ(result[i], expected) << c.label << " " << describe(c.lhs) << " " << op_symbol << " "
                                           << describe(c.rhs) << " at flat index " << i;
        }
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Binary: ttnn::add and ttnn::multiply across every subtile broadcast class.
// Covers binary_ng's reader stride computation and subtile-broadcast dispatch.
// ---------------------------------------------------------------------------

TEST_F(EltwiseSmoke, BinaryAddBroadcastClasses) {
    detail::run_binary_broadcast_cases(
        *device_, [](const Tensor& a, const Tensor& b) { return ttnn::add(a, b); }, std::plus<float>{}, "+");
}

TEST_F(EltwiseSmoke, BinaryMultiplyBroadcastClasses) {
    detail::run_binary_broadcast_cases(
        *device_, [](const Tensor& a, const Tensor& b) { return ttnn::multiply(a, b); }, std::multiplies<float>{}, "*");
}

// ---------------------------------------------------------------------------
// Unary: liveness of the SFPU dispatch path. abs and neg are exact in bfloat16;
// exp is transcendental and compares with a tolerance.
// ---------------------------------------------------------------------------

TEST_F(EltwiseSmoke, UnaryAbsNegExp) {
    auto& device = *device_;
    const ttnn::Shape shape({2, 1, 128, 128});
    const auto count = static_cast<size_t>(shape.volume());

    // Straddles zero so abs and neg both have something to do.
    const auto signed_data = detail::coordinate_ramp(shape, 16, -8);
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
        std::vector<float> exp_input = detail::coordinate_ramp(shape, 9, -4);
        for (auto& v : exp_input) {
            v /= 4.0f;
        }
        // The device sees the bfloat16 round of exp_input, not exp_input itself. Round the golden's
        // input the same way, so the tolerance below bounds the op's error and not the input
        // quantisation on top of it.
        std::vector<float> expected(count);
        for (size_t i = 0; i < count; i++) {
            expected[i] = std::exp(static_cast<float>(bfloat16(exp_input[i])));
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

    // One case per TernaryBroadcastType with a WHERE/TTT kernel triple. Classification follows
    // the three-tensor get_broadcast_type() in ternary_op_utils.cpp: ROW_BCAST needs all three
    // widths equal with a height broadcast, COL_BCAST all three heights equal with a width
    // broadcast, and the mixed case falls through to ROW_COL_BCAST.
    const std::vector<TernaryCase> cases = {
        // TernaryBroadcastType::NONE
        {"no_bcast", ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 128, 128})},
        // OUTER_BCAST: value tensors broadcast over the leading dims only.
        {"outer_bcast", ttnn::Shape({3, 2, 128, 128}), ttnn::Shape({1, 1, 128, 128}), ttnn::Shape({3, 1, 128, 128})},
        // ROW_BCAST: value_true broadcasts H; all three widths match -- ReaderRowBcastTTT.
        {"row_bcast", ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 1, 128}), ttnn::Shape({2, 1, 128, 128})},
        // COL_BCAST: value_true broadcasts W; all three heights match -- ReaderColBcastTTT.
        {"col_bcast", ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 128, 1}), ttnn::Shape({2, 1, 128, 128})},
        // ROW_COL_BCAST: value_true broadcasts H, value_false broadcasts W -- the mixed case.
        {"row_col_bcast", ttnn::Shape({2, 2, 128, 128}), ttnn::Shape({1, 1, 1, 128}), ttnn::Shape({1, 1, 128, 1})},
        // SCALAR_BCAST: value_true is a single element per batch.
        {"scalar_bcast", ttnn::Shape({2, 1, 128, 128}), ttnn::Shape({2, 1, 1, 1}), ttnn::Shape({2, 1, 128, 128})},
    };

    for (const auto& c : cases) {
        const auto out_shape =
            detail::broadcast_shape(detail::broadcast_shape(c.predicate, c.value_true), c.value_false);

        // Alternating 0/1, so both arms of the select are taken within every tile. The weighted
        // coordinate sum keeps it binary while still flipping across leading planes, so a
        // wrong-plane predicate read selects the wrong arm.
        const auto predicate_data = detail::coordinate_ramp(c.predicate, 2, 0);
        // Disjoint value ranges: picking the wrong arm is unmistakable, not off by a rounding step.
        const auto true_data = detail::coordinate_ramp(c.value_true, 16, 1);
        const auto false_data = detail::coordinate_ramp(c.value_false, 16, 101);

        const auto predicate = detail::to_device_bf16(device, c.predicate, predicate_data);
        const auto value_true = detail::to_device_bf16(device, c.value_true, true_data);
        const auto value_false = detail::to_device_bf16(device, c.value_false, false_data);
        const auto output = ttnn::where(predicate, value_true, value_false);

        ASSERT_EQ(output.logical_shape(), out_shape) << c.label;
        const auto result = detail::to_float_vector(output);
        ASSERT_EQ(result.size(), static_cast<size_t>(out_shape.volume())) << c.label;
        for (size_t i = 0; i < result.size(); i++) {
            const bool taken = predicate_data[detail::broadcast_index(out_shape, c.predicate, i)] != 0.0f;
            const float expected = taken ? true_data[detail::broadcast_index(out_shape, c.value_true, i)]
                                         : false_data[detail::broadcast_index(out_shape, c.value_false, i)];
            ASSERT_EQ(result[i], expected) << c.label << " at flat index " << i;
        }
    }
}

}  // namespace ttnn::operations::eltwise::test
