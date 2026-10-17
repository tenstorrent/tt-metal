// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <iostream>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {

template <typename T>
struct RelationalUnaryParamT {
    T scalar_input;
    T scalar_param;
    uint32_t h;
    uint32_t w;
};

using RelationalUnaryParam = RelationalUnaryParamT<int32_t>;

class RelationalUnaryFixture : public TTNNFixtureWithSuiteDevice<RelationalUnaryFixture>,
                               public testing::WithParamInterface<RelationalUnaryParam> {};

static ttnn::Tensor make_filled_tensor(
    ttnn::MeshDevice& device, uint32_t h, uint32_t w, DataType dtype, operations::unary::ScalarVariant fill_value) {
    std::array<uint32_t, 2> dimensions = {h, w};
    ttnn::Shape shape(dimensions);
    auto zero_tensor = ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);
    return ttnn::fill(zero_tensor, fill_value);
}

template <typename T, DataType DTYPE, typename UnaryOp>
void test_relational_unary_impl(
    ttnn::MeshDevice& device,
    const RelationalUnaryParamT<T>& p,
    const UnaryOp& unary_op,
    T expected_value,
    const char* op_name) {
    auto input_tensor = make_filled_tensor(device, p.h, p.w, DTYPE, p.scalar_input);
    auto output = unary_op(input_tensor, p.scalar_param);

    auto expected = make_filled_tensor(device, p.h, p.w, DTYPE, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);

    TT_FATAL(ttnn::allclose<T>(expected_host, output_host), "{} result mismatch", op_name);
}

TEST_P(RelationalUnaryFixture, EqUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl<int32_t, DataType::INT32>(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::eq_unary(t, s); },
        (p.scalar_input == p.scalar_param) ? 1 : 0,
        "eq_unary");
}

TEST_P(RelationalUnaryFixture, NeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl<int32_t, DataType::INT32>(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::ne_unary(t, s); },
        (p.scalar_input != p.scalar_param) ? 1 : 0,
        "ne_unary");
}

TEST_P(RelationalUnaryFixture, GtUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl<int32_t, DataType::INT32>(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::gt_unary(t, s); },
        (p.scalar_input > p.scalar_param) ? 1 : 0,
        "gt_unary");
}

TEST_P(RelationalUnaryFixture, LtUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl<int32_t, DataType::INT32>(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::lt_unary(t, s); },
        (p.scalar_input < p.scalar_param) ? 1 : 0,
        "lt_unary");
}

TEST_P(RelationalUnaryFixture, GeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl<int32_t, DataType::INT32>(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::ge_unary(t, s); },
        (p.scalar_input >= p.scalar_param) ? 1 : 0,
        "ge_unary");
}

TEST_P(RelationalUnaryFixture, LeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl<int32_t, DataType::INT32>(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::le_unary(t, s); },
        (p.scalar_input <= p.scalar_param) ? 1 : 0,
        "le_unary");
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUnaryTests,
    RelationalUnaryFixture,
    ::testing::Values(
        RelationalUnaryParam{2147483647, 2147483647, 128, 32},    // equality at max int
        RelationalUnaryParam{-2147483647, -2147483647, 128, 32},  // equality at min int for negative case
        RelationalUnaryParam{2147483647, 2147483646, 128, 32},    // gt, ne
        RelationalUnaryParam{2147483647, -1, 64, 32},             // gt, ge, ne
        RelationalUnaryParam{-45, -5, 32, 32},                    // lt, le, ne
        RelationalUnaryParam{5, -5, 32, 32},                      // gt, ge, ne
        RelationalUnaryParam{-5, 67, 64, 64},                     // lt, ne
        RelationalUnaryParam{0, 0, 4, 4},                         // eq, ge, le
        RelationalUnaryParam{17, -96, 64, 64},                    // gt, ge, ne
        RelationalUnaryParam{100, 458, 128, 128},                 // lt, ne
        RelationalUnaryParam{564, -118, 64, 128},                 // gt, ne
        RelationalUnaryParam{-2147483648, -2147483648, 128, 32},  // eq, ge, le at INT32_MIN
        RelationalUnaryParam{-2147483648, 0, 128, 32}));          // lt, le, ne at INT32_MIN

// UINT32 eq/ne against a scalar run through the same int32 kernel on the raw 32-bit word, so the
// word 0x80000000 (2147483648 unsigned) is the UINT32 value that INT32_MIN's load mode decides.
using RelationalUnaryUInt32Param = RelationalUnaryParamT<uint32_t>;

class RelationalUnaryUInt32Fixture : public TTNNFixtureWithSuiteDevice<RelationalUnaryUInt32Fixture>,
                                     public testing::WithParamInterface<RelationalUnaryUInt32Param> {};

TEST_P(RelationalUnaryUInt32Fixture, EqUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl<uint32_t, DataType::UINT32>(
        *device_,
        p,
        [](const auto& t, uint32_t s) { return ttnn::eq_unary(t, s); },
        (p.scalar_input == p.scalar_param) ? 1u : 0u,
        "eq_unary");
}

TEST_P(RelationalUnaryUInt32Fixture, NeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl<uint32_t, DataType::UINT32>(
        *device_,
        p,
        [](const auto& t, uint32_t s) { return ttnn::ne_unary(t, s); },
        (p.scalar_input != p.scalar_param) ? 1u : 0u,
        "ne_unary");
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUnaryUInt32Tests,
    RelationalUnaryUInt32Fixture,
    ::testing::Values(
        RelationalUnaryUInt32Param{2147483648u, 2147483648u, 128, 32},   // eq at 0x80000000
        RelationalUnaryUInt32Param{2147483648u, 0u, 128, 32},            // ne at 0x80000000
        RelationalUnaryUInt32Param{4294967295u, 4294967295u, 64, 32}));  // eq at UINT32_MAX

}  // namespace ttnn::operations::unary::test
