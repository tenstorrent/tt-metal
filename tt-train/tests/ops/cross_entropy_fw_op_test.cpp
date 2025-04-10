
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

class CrossEntropyForwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> calculate_cross_entropy_loss(const xt::xarray<float>& input, const xt::xarray<float>& target) {
    xt::xarray<float> max_input = xt::amax(input, -1);
    xt::xarray<float> max_input_expanded = xt::expand_dims(max_input, 3);
    xt::xarray<float> shift_input = input - max_input_expanded;
    xt::xarray<float> sum_exp_input = xt::sum(xt::exp(shift_input), -1);
    xt::xarray<float> log_sum_exp_input = xt::log(sum_exp_input);
    xt::xarray<float> log_sum_exp_input_expanded = xt::expand_dims(log_sum_exp_input, 3);
    xt::xarray<float> result = -shift_input * target + target * log_sum_exp_input_expanded;
    return result;
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Small_Forward) {
    using namespace ttml;

    const uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto input_logits = core::from_xtensor(example_xtensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input_logits.print();

    xt::xarray<float> target_xtensor = {{{{0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 0.F, 0.F}}}};
    assert((target_xtensor.shape() == example_xtensor.shape()));
    auto target = core::from_xtensor(target_xtensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input_logits, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(example_xtensor, target_xtensor);
    assert((expected_result.shape() == example_xtensor.shape()));
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result From :\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Negetive_Values) {
    using namespace ttml;

    const uint32_t N = 1, C = 1, H = 2, W = 4;

    xt::xarray<float> example_xtensor = {{{{-100.F, -101.F, -102.F, -103.F}, {-5.01F, -5.02F, -0.3F, -7.F}}}};
    auto input_logits = core::from_xtensor(example_xtensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input_logits.print();

    xt::xarray<float> target_xtensor = {{{{1.F, 0.F, 0.F, 0.F}, {0.F, 0.F, 1.F, 0.F}}}};
    assert((target_xtensor.shape() == example_xtensor.shape()));
    auto target = core::from_xtensor(target_xtensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input_logits, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(example_xtensor, target_xtensor);
    assert((expected_result.shape() == example_xtensor.shape()));
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result From :\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F, 1e-2F));
}
