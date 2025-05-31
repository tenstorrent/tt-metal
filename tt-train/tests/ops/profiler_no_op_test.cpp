// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class ProfilerNoOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ProfilerNoOpTest, ProfilerNoOpTest_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 91U, W = 187U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::profiler_no_op(input);
    std::cout << "Profiler_no_op_test:\nResult:\n";
    result.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == input_tensor.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, input_tensor, 1e-2F, 1e-2F));
}

TEST_F(ProfilerNoOpTest, ProfilerNoOpTest_Huge_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::profiler_no_op(input);
    std::cout << "Profiler_no_op_test:\nResult:\n";
    result.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == input_tensor.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, input_tensor, 1e-2F, 1e-2F));
}
