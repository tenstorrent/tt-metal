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
#include "ttnn/tensor/to_string.hpp"

namespace ttnn::operations::unary::test {

struct RelationalUnaryU32Param {
    uint32_t scalar_input;
    uint32_t scalar_param;
    uint32_t h;
    uint32_t w;
};

class RelationalUnaryU32Fixture : public TTNNFixtureWithDevice,
                                  public testing::WithParamInterface<RelationalUnaryU32Param> {};

static ttnn::Tensor make_uint_tensor(ttnn::MeshDevice& device, uint32_t h, uint32_t w, uint32_t fill_value) {
    std::array<uint32_t, 2> dimensions = {h, w};
    ttnn::Shape shape(dimensions);
    constexpr DataType dtype = DataType::UINT32;
    auto zero_tensor = ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);
    return ttnn::fill(zero_tensor, fill_value);
}

TEST_P(RelationalUnaryU32Fixture, EqUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_uint_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::eq_unary(input_tensor, p.scalar_param);

    uint32_t expected_value = (p.scalar_input == p.scalar_param) ? 1u : 0u;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_uint_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    std::cout << "op_name\t: eq_unary (uint32)\n";
    std::cout << "scalar_input\t: " << p.scalar_input << "\n";
    std::cout << ttnn::to_string(input_tensor);
    std::cout << "scalar_param\t: " << p.scalar_param << "\n";
    std::cout << "actual output:" << std::endl;
    // output.print();
    std::cout << ttnn::to_string(output);
    std::cout << "\nexpected output:" << std::endl;
    // expected.print();
    std::cout << ttnn::to_string(expected);
    TT_FATAL(ttnn::allclose<uint32_t>(expected_host, output_host), "eq_unary uint32 result mismatch");
}

TEST_P(RelationalUnaryU32Fixture, NeUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_uint_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::ne_unary(input_tensor, p.scalar_param);

    uint32_t expected_value = (p.scalar_input != p.scalar_param) ? 1u : 0u;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_uint_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    std::cout << "op_name\t: ne_unary (uint32)\n";
    std::cout << "scalar_input\t: " << p.scalar_input << "\n";
    std::cout << ttnn::to_string(input_tensor);
    std::cout << "scalar_param\t: " << p.scalar_param << "\n";
    std::cout << "actual output:" << std::endl;
    // output.print();
    std::cout << ttnn::to_string(output);
    std::cout << "\nexpected output:" << std::endl;
    // expected.print();
    std::cout << ttnn::to_string(expected);
    TT_FATAL(ttnn::allclose<uint32_t>(expected_host, output_host), "ne_unary uint32 result mismatch");
}

TEST_P(RelationalUnaryU32Fixture, GtUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_uint_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::gt_unary(input_tensor, p.scalar_param);

    uint32_t expected_value = (p.scalar_input > p.scalar_param) ? 1u : 0u;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_uint_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    std::cout << "op_name\t: gt_unary (uint32)\n";
    std::cout << "scalar_input\t: " << p.scalar_input << "\n";
    std::cout << ttnn::to_string(input_tensor);
    std::cout << "scalar_param\t: " << p.scalar_param << "\n";
    std::cout << "actual output:" << std::endl;
    // output.print();
    std::cout << ttnn::to_string(output);
    std::cout << "\nexpected output:" << std::endl;
    // expected.print();
    std::cout << ttnn::to_string(expected);
    TT_FATAL(ttnn::allclose<uint32_t>(expected_host, output_host), "gt_unary uint32 result mismatch");
}

TEST_P(RelationalUnaryU32Fixture, LtUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_uint_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::lt_unary(input_tensor, p.scalar_param);

    uint32_t expected_value = (p.scalar_input < p.scalar_param) ? 1u : 0u;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_uint_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    std::cout << "op_name\t: lt_unary (uint32)\n";
    std::cout << "scalar_input\t: " << p.scalar_input << "\n";
    std::cout << ttnn::to_string(input_tensor);
    std::cout << "scalar_param\t: " << p.scalar_param << "\n";
    std::cout << "actual output:" << std::endl;
    //  output.print();
    std::cout << ttnn::to_string(output);
    std::cout << "\nexpected output:" << std::endl;
    // expected.print();
    std::cout << ttnn::to_string(expected);
    TT_FATAL(ttnn::allclose<uint32_t>(expected_host, output_host), "lt_unary uint32 result mismatch");
}

TEST_P(RelationalUnaryU32Fixture, GeUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_uint_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::ge_unary(input_tensor, p.scalar_param);

    uint32_t expected_value = (p.scalar_input >= p.scalar_param) ? 1u : 0u;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_uint_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    std::cout << "op_name\t: ge_unary (uint32)\n";
    std::cout << "scalar_input\t: " << p.scalar_input << "\n";
    std::cout << ttnn::to_string(input_tensor);
    std::cout << "scalar_param\t: " << p.scalar_param << "\n";
    std::cout << "actual output:" << std::endl;
    // output.print();
    std::cout << ttnn::to_string(output);
    std::cout << "\nexpected output:" << std::endl;
    // expected.print();
    std::cout << ttnn::to_string(expected);
    TT_FATAL(ttnn::allclose<uint32_t>(expected_host, output_host), "ge_unary uint32 result mismatch");
}

TEST_P(RelationalUnaryU32Fixture, LeUnaryMatchesExpected) {
    auto p = GetParam();
    auto& device = *device_;
    auto input_tensor = make_uint_tensor(device, p.h, p.w, p.scalar_input);
    auto output = ttnn::le_unary(input_tensor, p.scalar_param);

    uint32_t expected_value = (p.scalar_input <= p.scalar_param) ? 1u : 0u;
    std::array<uint32_t, 2> dims = {p.h, p.w};
    ttnn::Shape shape(dims);
    auto expected = make_uint_tensor(device, p.h, p.w, expected_value);

    auto expected_host = ttnn::from_device(expected);
    auto output_host = ttnn::from_device(output);
    std::cout << "op_name\t: le_unary (uint32)\n";
    std::cout << "scalar_input\t: " << p.scalar_input << "\n";
    std::cout << ttnn::to_string(input_tensor);
    std::cout << "scalar_param\t: " << p.scalar_param << "\n";
    std::cout << "actual output:" << std::endl;
    // output.print();
    std::cout << ttnn::to_string(output);
    std::cout << "\nexpected output:" << std::endl;
    // expected.print();
    std::cout << ttnn::to_string(expected);
    TT_FATAL(ttnn::allclose<uint32_t>(expected_host, output_host), "le_unary uint32 result mismatch");
}

INSTANTIATE_TEST_SUITE_P(
    RelationalUnaryU32Tests,
    RelationalUnaryU32Fixture,
    ::testing::Values(
        RelationalUnaryU32Param{0u, 0u, 128, 32},       // eq at zero
        RelationalUnaryU32Param{1u, 0u, 128, 32},       // gt, ne
        RelationalUnaryU32Param{0u, 1u, 64, 64},        // lt, ne, le
        RelationalUnaryU32Param{42u, 42u, 32, 32},      // eq
        RelationalUnaryU32Param{100u, 458u, 128, 128},  // lt, ne, le
        RelationalUnaryU32Param{564u, 118u, 64, 128},   // gt, ge, ne
        RelationalUnaryU32Param{2147483645u, 2147483647u, 32, 32},
        RelationalUnaryU32Param{2147483647u, 0u, 8, 8}));

}  // namespace ttnn::operations::unary::test
