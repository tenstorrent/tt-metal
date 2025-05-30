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

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> xt_softmax(const xt::xarray<float>& input, uint32_t dim = 3U) {
    xt::xarray<float> max_value = xt::amax(input, dim, xt::keep_dims);
    xt::xarray<float> exp_shifted_input = xt::exp(input - max_value);
    xt::xarray<float> exp_sum = xt::sum(exp_shifted_input, -1, xt::keep_dims);
    xt::xarray<float> result = exp_shifted_input / exp_sum;
    return result;
}

TEST_F(SoftmaxTest, SoftmaxTest_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 91U, W = 187U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    int32_t dim = 3U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::softmax(input, dim);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result.print();

    auto ttnn_softmax = ttnn_fixed::softmax(input, dim);
    auto ttnn_softmax_xtensor = core::to_xtensor(ttnn_softmax);

    auto expected_result = xt_softmax(input_tensor, dim);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t w = 0; w < W; ++w) {
                float error = std::abs(result_xtensor(n, 1U, h, w) - expected_result(n, 1U, h, w));
                float max_error = 1e-2F + 3e-2F * std::abs(expected_result(n, 1U, h, w));

                if (error > max_error) {
                    std::cout << "Mismatch at (" << n << ", " << 0 << ", " << h << ", " << w
                              << "): result = " << result_xtensor(n, 0, h, w)
                              << ", expected = " << expected_result(n, 0, h, w) << ", error = " << error
                              << ", max_error = " << max_error << std::endl;
                }
            }
        }
    }

    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, SoftmaxTest_Big_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 32U, W = 128007U;
    const auto shape = ttnn::SmallVector<uint32_t>{N, C, H, W};
    int32_t dim = 3U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::random::rand<float>({N, C, H, W}, -10.0F, 10.0F, gen);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::softmax(input, dim);
    std::cout << "CrossEntropyBackward_Test:\nResult:\n";
    result.print();

    // auto ttnn_softmax = ttnn_fixed::softmax(input, dim); // crash with big batch ???
    // auto ttnn_softmax_xtensor = core::to_xtensor(ttnn_softmax);

    auto expected_result = xt_softmax(input_tensor, dim);
    auto expected_result_print = core::from_xtensor(expected_result, &autograd::ctx().get_device());
    std::cout << "Expected Result:\n";
    expected_result_print.print();

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}
