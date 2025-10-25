// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
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

class RelationalUnaryFixture : public TTNNFixtureWithDevice,
                               public testing::WithParamInterface<RelationalUnaryParam> {};

static ttnn::Tensor make_int_tensor(ttnn::MeshDevice& device, uint32_t h, uint32_t w, int32_t fill_value) {
    std::array<uint32_t, 2> dimensions = {h, w};
    ttnn::Shape shape(dimensions);
    constexpr DataType dtype = DataType::INT32;
    auto zero_tensor = ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);
    return ttnn::fill(zero_tensor, fill_value);
}

TEST_P(RelationalUnaryFixture, EqUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_int_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::eq_unary(input_tensor, p.scalar_param);

    int32_t expected_value = (p.scalar_input == p.scalar_param) ? 1 : 0;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_int_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    TT_FATAL(ttnn::allclose<int32_t>(expected_host, output_host), "eq_unary result mismatch");
}

TEST_P(RelationalUnaryFixture, NeUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_int_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::ne_unary(input_tensor, p.scalar_param);

    int32_t expected_value = (p.scalar_input != p.scalar_param) ? 1 : 0;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_int_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    TT_FATAL(ttnn::allclose<int32_t>(expected_host, output_host), "ne_unary result mismatch");
}

TEST_P(RelationalUnaryFixture, GtUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_int_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::gt_unary(input_tensor, p.scalar_param);

    int32_t expected_value = (p.scalar_input > p.scalar_param) ? 1 : 0;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_int_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    TT_FATAL(ttnn::allclose<int32_t>(expected_host, output_host), "gt_unary result mismatch");
}

TEST_P(RelationalUnaryFixture, LtUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_int_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::lt_unary(input_tensor, p.scalar_param);

    int32_t expected_value = (p.scalar_input < p.scalar_param) ? 1 : 0;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_int_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    TT_FATAL(ttnn::allclose<int32_t>(expected_host, output_host), "lt_unary result mismatch");
}

TEST_P(RelationalUnaryFixture, GeUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_int_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::ge_unary(input_tensor, p.scalar_param);

    int32_t expected_value = (p.scalar_input >= p.scalar_param) ? 1 : 0;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_int_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    TT_FATAL(ttnn::allclose<int32_t>(expected_host, output_host), "ge_unary result mismatch");
}

TEST_P(RelationalUnaryFixture, LeUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_int_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::le_unary(input_tensor, p.scalar_param);

    int32_t expected_value = (p.scalar_input <= p.scalar_param) ? 1 : 0;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_int_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    TT_FATAL(ttnn::allclose<int32_t>(expected_host, output_host), "le_unary result mismatch");
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
