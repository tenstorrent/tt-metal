
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
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
    xt::xarray<float> reduced_result = xt::sum(result, -1, xt::keep_dims);
    return reduced_result;
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
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result From :\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Batch) {
    using namespace ttml;

    const uint32_t N = 3, C = 1, H = 2, W = 47;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    std::vector<float> logits_data(N * C * H * W);
    std::vector<float> target_data(N * C * H * W, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_int_distribution<std::size_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t offset = (n * H + h) * W;  // flattened offset
            for (uint32_t w = 0; w < W; ++w) {
                logits_data[offset + w] = dist(gen);  // random logits
            }
            std::size_t true_class = class_dist(gen) % W;
            logits_data[offset + true_class] = 100 + n + h;  // one-hot
            target_data[offset + true_class] = 1.0f;         // one-hot
        }
    }

    xt::xarray<float> logits_tensor = xt::adapt(logits_data, shape);
    xt::xarray<float> target_tensor = xt::adapt(target_data, shape);

    auto input_logits = core::from_xtensor(logits_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input_logits.print();

    auto target = core::from_xtensor(target_tensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input_logits, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(logits_tensor, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result From :\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(expected_result, result_xtensor, 1e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64, C = 1, H = 1024, W = 1032;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    std::vector<float> logits_data(N * C * H * W);
    std::vector<float> target_data(N * C * H * W, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_int_distribution<std::size_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t offset = (n * H + h) * W;  // flattened offset
            for (uint32_t w = 0; w < W; ++w) {
                logits_data[offset + w] = dist(gen);  // random logits
            }
            std::size_t true_class = class_dist(gen) % W;
            target_data[offset + true_class] = 1.0f;  // one-hot
        }
    }

    xt::xarray<float> logits_tensor = xt::adapt(logits_data, shape);
    xt::xarray<float> target_tensor = xt::adapt(target_data, shape);

    auto input_logits = core::from_xtensor(logits_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input_logits.print();

    auto target = core::from_xtensor(target_tensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input_logits, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(logits_tensor, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result From :\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(expected_result, result_xtensor, 1e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Forward) {
    using namespace ttml;

    const uint32_t N = 1, C = 1, H = 1, W = 65536;
    const auto shape = ttnn::SmallVector<size_t>{N, C, H, W};

    // 1. Generate random logits in [-10, 10]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> logits_data(W);  // look at xt::rand() ?
    for (std::size_t i = 0; i < W; ++i) {
        logits_data[i] = dist(gen);
    }
    xt::xarray<float> input_logits = xt::adapt(logits_data, shape);  // do I need move here?
    auto input = core::from_xtensor(input_logits, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    // One-hot target
    std::vector<float> target_big_data(shape[3], 0.0f);
    std::uniform_int_distribution<std::size_t> class_dist(0, W - 1);
    std::size_t true_class = class_dist(gen);
    target_big_data[true_class] = 1.0f;
    xt::xarray<float> target_tensor = xt::adapt(target_big_data, shape);
    auto target = core::from_xtensor(target_tensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_logits, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result From :\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result));
}
