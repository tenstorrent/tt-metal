// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "ttnn/decorators.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {

struct RelationalUnaryParam {
    int32_t scalar_input;
    int32_t scalar_param;
    uint32_t h;
    uint32_t w;
};

class RelationalUnaryFixture : public TTNNFixtureWithSuiteDevice<RelationalUnaryFixture>,
                               public testing::WithParamInterface<RelationalUnaryParam> {};

static ttnn::Tensor make_int_tensor(ttnn::MeshDevice& device, uint32_t h, uint32_t w, int32_t fill_value) {
    std::array<uint32_t, 2> dimensions = {h, w};
    ttnn::Shape shape(dimensions);
    constexpr DataType dtype = DataType::INT32;
    auto zero_tensor = ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);
    return ttnn::fill(zero_tensor, fill_value);
}

template <typename UnaryOp>
void test_relational_unary_impl(
    ttnn::MeshDevice& device,
    const RelationalUnaryParam& p,
    const UnaryOp& unary_op,
    int32_t expected_value,
    const char* op_name) {
    auto input_tensor = make_int_tensor(device, p.h, p.w, p.scalar_input);
    auto output = unary_op(input_tensor, p.scalar_param);

    auto expected = make_int_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);

    TT_FATAL(ttnn::allclose<int32_t>(expected_host, output_host), "{} result mismatch", op_name);
}

TEST_P(RelationalUnaryFixture, EqUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::eq_unary(t, s); },
        (p.scalar_input == p.scalar_param) ? 1 : 0,
        "eq_unary");
}

TEST_P(RelationalUnaryFixture, NeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::ne_unary(t, s); },
        (p.scalar_input != p.scalar_param) ? 1 : 0,
        "ne_unary");
}

TEST_P(RelationalUnaryFixture, GtUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::gt_unary(t, s); },
        (p.scalar_input > p.scalar_param) ? 1 : 0,
        "gt_unary");
}

TEST_P(RelationalUnaryFixture, LtUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::lt_unary(t, s); },
        (p.scalar_input < p.scalar_param) ? 1 : 0,
        "lt_unary");
}

TEST_P(RelationalUnaryFixture, GeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
        *device_,
        p,
        [](const auto& t, auto s) { return ttnn::ge_unary(t, s); },
        (p.scalar_input >= p.scalar_param) ? 1 : 0,
        "ge_unary");
}

TEST_P(RelationalUnaryFixture, LeUnaryMatchesExpected) {
    auto p = GetParam();
    test_relational_unary_impl(
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
        RelationalUnaryParam{564, -118, 64, 128}));               // gt, ne

}  // namespace ttnn::operations::unary::test
