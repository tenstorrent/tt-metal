// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_test_fixtures.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/expression/operations.hpp"
#include "ttnn/operations/eltwise/materialize/device/materialize_device_operation.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::materialize::test {

TEST_F(TTNNFixtureWithDevice, TestMaterializeReciprocal) {
    ttnn::Shape shape{1, 4, 320, 384};
    auto input_tensor = ttnn::random::uniform<bfloat16>(0.5, 1.5, shape, Layout::TILE);
    auto device_input_tensor = input_tensor.to_device(this->device_);
    auto golden = ttnn::reciprocal(device_input_tensor).cpu();

    auto expression = expression::reciprocal(device_input_tensor);
    auto device_output_tensor = ttnn::prim::materialize(expression);
    auto output_tensor = device_output_tensor.cpu();
    auto rtol = 0.01f;
    auto atol = 0.01f;
    auto allclose = ttnn::allclose<bfloat16>(golden, output_tensor, rtol, atol);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestMaterializeMul) {
    ttnn::Shape shape{1, 4, 320, 384};
    auto input_tensor_a = ttnn::random::uniform<bfloat16>(-16, 16, shape, Layout::TILE);
    auto input_tensor_b = ttnn::random::uniform<bfloat16>(-16, 16, shape, Layout::TILE);
    auto device_input_tensor_a = input_tensor_a.to_device(this->device_);
    auto device_input_tensor_b = input_tensor_b.to_device(this->device_);
    auto golden = ttnn::multiply(device_input_tensor_a, device_input_tensor_b).cpu();

    auto expression = expression::mul(device_input_tensor_a, device_input_tensor_b);
    auto device_output_tensor = ttnn::prim::materialize(expression);
    auto output_tensor = device_output_tensor.cpu();
    auto allclose = ttnn::allclose<bfloat16>(golden, output_tensor);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestMaterializeAtan2) {
    ttnn::Shape shape{1, 4, 320, 384};
    auto input_tensor_a = ttnn::random::uniform<bfloat16>(-16, 16, shape, Layout::TILE);
    auto input_tensor_b = ttnn::random::uniform<bfloat16>(-16, 16, shape, Layout::TILE);
    auto device_input_tensor_a = input_tensor_a.to_device(this->device_);
    auto device_input_tensor_b = input_tensor_b.to_device(this->device_);
    auto golden = ttnn::atan2(device_input_tensor_a, device_input_tensor_b).cpu();

    auto expression = expression::atan2(device_input_tensor_a, device_input_tensor_b);
    auto device_output_tensor = ttnn::prim::materialize(expression);
    auto output_tensor = device_output_tensor.cpu();
    auto rtol = 0.f;
    auto atol = 0.02f;
    auto allclose = ttnn::allclose<bfloat16>(golden, output_tensor, rtol, atol);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestMaterializeAddCMul) {
    ttnn::Shape shape{1, 4, 320, 384};
    auto input_tensor_a = ttnn::random::uniform<bfloat16>(-16, 16, shape, Layout::TILE);
    auto input_tensor_b = ttnn::random::uniform<bfloat16>(-16, 16, shape, Layout::TILE);
    auto input_tensor_c = ttnn::random::uniform<bfloat16>(-16, 16, shape, Layout::TILE);
    auto value = std::generate_canonical<float, 16>(ttnn::random::RANDOM_GENERATOR);
    auto device_input_tensor_a = input_tensor_a.to_device(this->device_);
    auto device_input_tensor_b = input_tensor_b.to_device(this->device_);
    auto device_input_tensor_c = input_tensor_c.to_device(this->device_);
    auto golden = ttnn::addcmul(device_input_tensor_a, device_input_tensor_b, device_input_tensor_c, value).cpu();

    constexpr auto addcmul_expression =
        [](const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c, float value) {
            auto mul = expression::mul(b, c);
            auto factor = expression::mul(value, mul);
            return expression::add(a, factor);
        };
    auto expression = addcmul_expression(device_input_tensor_a, device_input_tensor_b, device_input_tensor_c, value);
    auto device_output_tensor = ttnn::prim::materialize(expression);
    auto output_tensor = device_output_tensor.cpu();
    auto rtol = 0.04f;
    auto atol = 0.04f;
    auto allclose = ttnn::allclose<bfloat16>(golden, output_tensor, rtol, atol);
    ASSERT_TRUE(allclose);
}

TEST_F(TTNNFixtureWithDevice, TestMaterializeThreshold) {
    ttnn::Shape shape{1, 4, 320, 384};
    auto input_tensor = ttnn::random::uniform<bfloat16>(-16, 16, shape, Layout::TILE);
    auto threshold = 10;
    auto value = 100;
    auto device_input_tensor = input_tensor.to_device(this->device_);
    auto golden = ttnn::threshold(device_input_tensor, threshold, value).cpu();

    constexpr auto threshold_expression = [](const ttnn::Tensor& input, auto threshold, auto value) {
        auto cond = expression::gt(input, threshold);
        return expression::where(cond, input, value);
    };
    auto expression = threshold_expression(device_input_tensor, threshold, value);
    auto device_output_tensor = ttnn::prim::materialize(expression);
    auto output_tensor = device_output_tensor.cpu();
    auto allclose = ttnn::allclose<bfloat16>(golden, output_tensor);
    ASSERT_TRUE(allclose);
}

}  // namespace ttnn::operations::materialize::test
