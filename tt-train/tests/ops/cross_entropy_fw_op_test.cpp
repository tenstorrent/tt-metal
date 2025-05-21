
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

xt::xarray<float> calculate_cross_entropy_loss(const xt::xarray<float>& input, const xt::xarray<uint32_t>& target) {
    const uint32_t N = target.shape(0);
    const uint32_t C = 1U;
    const uint32_t H = target.shape(1);
    const uint32_t W = 1U;
    xt::xarray<float> target_inputs = xt::zeros<float>({N, C, H, W});

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            size_t class_index = target(n, h);
            target_inputs(n, 0, h, 0) = input(n, 0, h, class_index);
        }
    }

    xt::xarray<float> max_input = xt::amax(input, -1, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_input;
    xt::xarray<float> log_exp_sum_test = xt::log(xt::sum(xt::exp(shifted_input), -1, xt::keep_dims));
    xt::xarray<float> result = -target_inputs + max_input + log_exp_sum_test;
    return result;
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Small_Forward) {
    using namespace ttml;

    const uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> input_tensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 1U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Negetive_Values) {
    using namespace ttml;

    const uint32_t N = 1, C = 1, H = 2, W = 4;

    xt::xarray<float> input_tensor = {{{{-100.F, -101.F, -102.F, -103.F}, {-5.01F, -5.02F, -0.3F, -7.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 0;
    target_tensor(0, 1) = 2U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 157U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1017U, W = 1018U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Forward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 65536U;
    const auto shape = ttnn::SmallVector<size_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Huge_Forward) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    const auto shape = ttnn::SmallVector<size_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}
