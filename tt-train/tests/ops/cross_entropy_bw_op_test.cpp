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

class CrossEntropyBackwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> calculate_cross_entropy_backward(
    const xt::xarray<float>& input, const xt::xarray<uint32_t>& target, const float scaler = 1.0F) {
    const uint32_t N = target.shape(0);
    const uint32_t C = 1U;
    const uint32_t H = target.shape(1);
    const uint32_t W = 1U;

    const auto input_shape = input.shape();
    xt::xarray<float> target_inputs = xt::zeros<float>(input_shape);

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            size_t class_index = target(n, h);
            target_inputs(n, 0, h, class_index) = 1.0F;
        }
    }

    xt::xarray<float> scaler_tensor(input_shape);
    scaler_tensor.fill(scaler);

    xt::xarray<float> max_input = xt::amax(input, -1, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_input;
    xt::xarray<float> exp_shifted_input = xt::exp(shifted_input);
    xt::xarray<float> exp_sum = xt::sum(exp_shifted_input, -1, xt::keep_dims);
    xt::xarray<float> result = exp_shifted_input / exp_sum - target_inputs;
    return result * scaler_tensor;
}

void printTensor(const xt::xarray<float>& tensor) {
    const uint32_t N = tensor.shape()[0];
    const uint32_t C = tensor.shape()[1];
    const uint32_t H = tensor.shape()[2];
    const uint32_t W = tensor.shape()[3];

    fmt::print("Tensor shape: {} x {} x {} x {}\n", N, C, H, W);
    fmt::print("[\n");
    for (uint32_t n = 0; n < N; ++n) {
        fmt::print("[\n");
        for (uint32_t h = 0; h < H; ++h) {
            fmt::print("[ ");
            for (uint32_t w = 0; w < W; ++w) {
                fmt::print("{:.4f}, ", tensor(n, 0, h, w));
            }
            fmt::print("]\n");
        }
        fmt::print("]\n");
    }
    fmt::print("]\n");
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Small_Backward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 8U;

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

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, scaler);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result[0].print();

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result[0]);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 91U, W = 187U;
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

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, scaler);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result[0].print();

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result[0]);
    assert((result_xtensor.shape() == expected_result.shape()));

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t w = 0; w < W; ++w) {
                auto error = std::abs(result_xtensor(n, 0, h, w) - expected_result(n, 0, h, w));
                auto max_error = 0.01F + std::abs(expected_result(n, 0, h, w)) * 0.03F;

                if (error > max_error) {
                    std::cout << "Error at (" << n << ", " << h << ", " << w << "): " << result_xtensor(n, 0, h, w)
                              << " vs " << expected_result(n, 0, h, w) << " (error: " << error
                              << ", max_error: " << max_error << ")" << " target: " << target_tensor(n, h) << std::endl;
                }
            }
        }
    }

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1024, W = 1024U;
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

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, scaler);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result[0].print();

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result[0]);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Large_Backward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 32U, W = 128007U;
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

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, scaler);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result[0].print();

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result[0]);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Huge_Backward) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 64, W = 128000U;
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

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, scaler);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result[0].print();

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result[0]);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Softmax_Test) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 10U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};

    std::random_device rd;
    std::mt19937 gen(42);
    // xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);
    xt::xarray<float> input_tensor = {
        {{{-5.1250F, -4.5938F, -1.8906F, 0.1196F, -3.2188F, -0.1826F, -5.2188F, -2.2656F, 3.5312F, 0.7969F}}}};

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 8U;

    xt::xarray<float> target_one_hot_tensor = xt::zeros<float>({N, 1U, 1U, W});
    target_one_hot_tensor(0, 0, 0, target_tensor(0, 0)) = 1.0F;
    auto target_one_hot = core::from_xtensor(target_one_hot_tensor, &autograd::ctx().get_device());

    // std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    // for (uint32_t n = 0; n < N; ++n) {
    //     for (uint32_t h = 0; h < H; ++h) {
    //         uint32_t true_class = class_dist(gen);
    //         target_tensor(n, h) = true_class;
    //     }
    // }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);
    std::cout << "Input Target Indexes:\n";
    target.print();

    float scaler = 1.0F;  //  / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, scaler);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result[0].print();

    fmt::print("Softmax Result:\n");
    printTensor(core::to_xtensor(result[1]));

    auto custom_softmax = ttnn_fixed::softmax(input, 3);
    std::cout << "Custom Softmax Result:\n";
    printTensor(core::to_xtensor(custom_softmax));

    auto custom_grad = ttnn::subtract(custom_softmax, target_one_hot);
    fmt::print("Custom Grad Result:\n");
    printTensor(core::to_xtensor(custom_grad));

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    fmt::print("Expected bw Result:\n");
    printTensor(expected_result);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result[0]);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
    EXPECT_TRUE(false);
}
