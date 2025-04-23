
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

xt::xarray<float> calculate_cross_entropy_loss(const xt::xarray<float>& input, const xt::xarray<float>& target) {
    xt::xarray<float> max_input = xt::amax(input, -1);
    xt::xarray<float> max_input_expanded = xt::expand_dims(max_input, 3);
    xt::xarray<float> shift_input = input - max_input_expanded;

    xt::xarray<float> target_inputs = xt::sum(input * target, -1, xt::keep_dims);
    xt::xarray<float> max_input_value = xt::amax(input, -1, xt::keep_dims);
    xt::xarray<float> log_exp_sum_test = xt::log(xt::sum(xt::exp(shift_input), -1, xt::keep_dims));

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

    xt::xarray<uint32_t> target_idx_xtensor = xt::zeros<uint32_t>({N, H});
    target_idx_xtensor(0, 0) = 1U;
    auto target_idx = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_idx_xtensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    std::cout << "Input Target Indexes:\n";
    target_idx.print();

    auto result = ttml::metal::cross_entropy_fw(input_logits, target, target_idx);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(example_xtensor, target_xtensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F, 1e-2F));
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

    xt::xarray<uint32_t> target_idx_xtensor = xt::zeros<uint32_t>({N, H});
    target_idx_xtensor(0, 0) = 0;
    target_idx_xtensor(0, 1) = 2U;
    auto target_idx = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_idx_xtensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target_idx.print();

    auto result = ttml::metal::cross_entropy_fw(input_logits, target, target_idx);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(example_xtensor, target_xtensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Batch) {
    using namespace ttml;

    // const uint32_t N = 2U, C = 1U, H = 91U, W = 157U;
    const uint32_t N = 1U, C = 1U, H = 91U, W = 29U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(rd());  // or fixed seed: std::mt19937 gen(42);
    xt::xarray<float> logits_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<float> target_tensor = xt::zeros<float>({N, C, H, W});

    xt::xarray<uint32_t> target_idx_xtensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, 0, h, true_class) = 1.0F;

            target_idx_xtensor(n, h) = true_class;
        }
    }

    auto target_idx = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_idx_xtensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target_idx.print();

    auto input_logits = core::from_xtensor(logits_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input_logits.print();

    auto target = core::from_xtensor(target_tensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input_logits, target, target_idx);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(logits_tensor, target_tensor);

    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            float error = std::abs(result_xtensor(n, 0, h, 0) - expected_result(n, 0, h, 0));
            float max_error = 1e-2F + 1e-2F * std::abs(expected_result(n, 0, h, 0));

            if (error > max_error) {
                std::cout << "result_xtensor(" << n << ", 0, " << h << ") = " << result_xtensor(n, 0, h, 0)
                          << " expected(" << n << ", 0, " << h << ") = " << expected_result(n, 0, h, 0) << "\n";

                std::cout << "Error: " << error << "\n";
                std::cout << "max_error: " << max_error << "\n";
            }
        }
    }

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
    // EXPECT_TRUE(false);
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1017U, W = 1018U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(rd());  // or fixed seed: std::mt19937 gen(42);
    xt::xarray<float> logits_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<float> target_tensor = xt::zeros<float>({N, C, H, W});

    xt::xarray<uint32_t> target_idx_xtensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, 0, h, true_class) = 1.0F;

            target_idx_xtensor(n, h) = true_class;
        }
    }

    auto target_idx = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_idx_xtensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target_idx.print();

    auto input_logits = core::from_xtensor(logits_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input_logits.print();

    auto target = core::from_xtensor(target_tensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input_logits, target, target_idx);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(logits_tensor, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(expected_result, result_xtensor, 3e-2F, 1e-2F));
    // How can I increase precision?
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Forward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 65536U;
    const auto shape = ttnn::SmallVector<size_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(rd());  // or fixed seed: std::mt19937 gen(42);
    xt::xarray<float> input_logits = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<float> target_tensor = xt::zeros<float>({N, C, H, W});

    xt::xarray<uint32_t> target_idx_xtensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, 0, h, true_class) = 1.0F;

            target_idx_xtensor(n, h) = true_class;
        }
    }

    auto target_idx = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_idx_xtensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target_idx.print();

    // xt::xarray<float> input_logits = xt::adapt(logits_data, shape);  // do I need move here?
    auto input = core::from_xtensor(input_logits, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    // One-hot target
    // xt::xarray<float> target_tensor = xt::adapt(target_big_data, shape);
    auto target = core::from_xtensor(target_tensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target, target_idx);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_logits, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Huge_Forward) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    const auto shape = ttnn::SmallVector<size_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(rd());  // or fixed seed: std::mt19937 gen(42);
    xt::xarray<float> input_logits = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<float> target_tensor = xt::zeros<float>({N, C, H, W});

    xt::xarray<uint32_t> target_idx_xtensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, 0, h, true_class) = 1.0F;

            target_idx_xtensor(n, h) = true_class;
        }
    }

    auto target_idx = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_idx_xtensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target_idx.print();

    auto input = core::from_xtensor(input_logits, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto target = core::from_xtensor(target_tensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target, target_idx);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_logits, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Target_Indexes) {
    using namespace ttml;

    const uint32_t N = 6U, C = 1U, H = 32U, W = 32U;
    const auto shape = ttnn::SmallVector<size_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(rd());  // or fixed seed: std::mt19937 gen(42);
    xt::xarray<float> input_logits = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<float> target_tensor = xt::zeros<float>({N, C, H, W});

    xt::xarray<uint32_t> target_idx_xtensor = xt::zeros<uint32_t>({N, C, H});

    std::uniform_int_distribution<std::size_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = h;                                       // class_dist(gen);
            target_tensor(n, 0, h, true_class) = 1.0F;                     // One-hot target
            input_logits(n, 0, h, true_class) = 100.0F + h * 10 + n * 10;  // spike to make sure this is highest

            // test target index
            target_idx_xtensor(n, 0, h) = h;
        }
    }

    // for (uint32_t h = 0; h < H; ++h) {
    //     std::cout << "target_idx_xtensor(0, 0, " << h << ", 0) = " << input_logits(1, 0, h, h) << "\n";
    // }

    auto target_idx = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_idx_xtensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target_idx.print();

    auto input = core::from_xtensor(input_logits, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto target = core::from_xtensor(target_tensor, &autograd::ctx().get_device());
    std::cout << "Input Targets:\n";
    target.print();

    auto result = ttml::metal::cross_entropy_fw(input, target, target_idx);
    std::cout << "CrossEntropyForward_Test:\nResult:\n";
    result.print();

    auto expected_result = calculate_cross_entropy_loss(input_logits, target_tensor);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F, 1e-2F));
}
