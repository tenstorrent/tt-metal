// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_fixed/trivial_ttnn_ops.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class TrivialTnnFixedTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(TrivialTnnFixedTest, TestMaxNegativeOne) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data(24, -1.F);
    auto shape = ttnn::Shape({1, 2, 3, 4});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto res = ttnn::max(tensor, /* dim */ 3, /* keepdim */ true);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_EQ(res_vector.size(), 6);
    bool all_equal = true;
    for (const auto& value : res_vector) {
        if (std::fabs(value + 1.F) > 1e-2) {
            all_equal = false;
        }
    }
    EXPECT_TRUE(all_equal);
}

TEST_F(TrivialTnnFixedTest, TestMaxNegativeBatch) {
    auto* device = &ttml::autograd::ctx().get_device();

    auto shape = ttnn::Shape({4, 1, 1, 4});
    std::vector<float> data(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            data[i * 4 + j] = -static_cast<float>(i + 1);
        }
    }
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto res = ttnn::max(tensor, /* dim */ 3, /* keepdim */ true);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_EQ(res_vector.size(), 4);
    bool all_equal = true;
    for (int i = 0; i < 4 && all_equal; ++i) {
        if (std::fabs(res_vector[i] - (-static_cast<float>(i + 1))) > 1e-2) {
            all_equal = false;
        }
    }
    EXPECT_TRUE(all_equal);
}

TEST_F(TrivialTnnFixedTest, TestStableSoftmax_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 1U;
    const size_t features = 2U;
    std::vector<float> data(batch_size * features);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = 100.F + static_cast<float>(i);
    }
    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_data = ttml::core::to_vector(tensor);
    EXPECT_NEAR(tensor_data[0], 100.F, 1e-2);
    EXPECT_NEAR(tensor_data[1], 101.F, 1e-2);

    auto res = ttml::ttnn_fixed::softmax(tensor, /* dim */ 3);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_NEAR(res_vector[0], 0.2689F, 2e-2);
    EXPECT_NEAR(res_vector[1], 0.7311F, 2e-2);
}

TEST_F(TrivialTnnFixedTest, TestOriginalStableSoftmax_AllNegative) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 1U;
    const size_t features = 2U;
    std::vector<float> data(batch_size * features);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = -100.F + static_cast<float>(i);
    }
    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_data = ttml::core::to_vector(tensor);
    EXPECT_NEAR(tensor_data[0], -100.F, 1e-2);
    EXPECT_NEAR(tensor_data[1], -99.F, 1e-2);
    auto compute_kernel_config = ttml::core::ComputeKernelConfig::precise();
    auto res = ttnn::softmax(
        tensor,
        /* dim */ 3,
        /*memory_config */ std::nullopt,
        compute_kernel_config,
        /*stable*/ true);
    auto res_vector = ttml::core::to_vector(res);
    EXPECT_NEAR(res_vector[0], 0.2689F, 2e-2);
    EXPECT_NEAR(res_vector[1], 0.7311F, 2e-2);
}

TEST_F(TrivialTnnFixedTest, TestStableSoftmax_2) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 1U;
    const size_t features = 10U;
    std::vector<float> data(batch_size * features, 0.F);
    data[0] = 1.0F;
    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_data = ttml::core::to_vector(tensor);
    EXPECT_NEAR(tensor_data[0], 1.F, 1e-2);
    EXPECT_NEAR(tensor_data[1], 0.F, 1e-2);

    auto res = ttml::ttnn_fixed::softmax(tensor, /* dim */ 3);
    auto res_vector = ttml::core::to_vector(res);

    auto exp_sum = 0.0F;
    for (auto& elem : data) {
        exp_sum += std::exp(elem);
    }

    for (int i = 0; i < res_vector.size(); ++i) {
        EXPECT_NEAR(res_vector[i], std::exp(data[i]) / exp_sum, 1e-2);
    }
}

TEST_F(TrivialTnnFixedTest, TestSumOverBatch_0) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 10U;
    const size_t features = 4U;
    std::vector<float> data(batch_size * features);
    std::iota(data.begin(), data.end(), 0);

    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_shape = tensor.logical_shape();
    EXPECT_EQ(tensor_shape[0], batch_size);
    EXPECT_EQ(tensor_shape[1], 1U);
    EXPECT_EQ(tensor_shape[2], 1U);
    EXPECT_EQ(tensor_shape[3], features);

    auto result = ttml::ttnn_fixed::sum_over_batch(tensor);
    const auto& result_shape = result.logical_shape();
    ASSERT_EQ(result_shape.rank(), 4U);
    EXPECT_EQ(result_shape[0], 1U);
    EXPECT_EQ(result_shape[1], 1U);
    EXPECT_EQ(result_shape[2], 1U);
    EXPECT_EQ(result_shape[3], features);
}

TEST_F(TrivialTnnFixedTest, TestDivide) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 2U;
    const size_t features = 64U;
    std::vector<float> lhs(batch_size * features);
    std::vector<float> rhs(batch_size * features);

    for (int i = 0; i < lhs.size(); ++i) {
        lhs[i] = static_cast<float>(i);
        rhs[i] = static_cast<float>(i + 1);
    }

    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto lhs_tensor = ttml::core::from_vector(lhs, shape, device);
    auto rhs_tensor = ttml::core::from_vector(rhs, shape, device);

    auto result = ttml::ttnn_fixed::divide(lhs_tensor, rhs_tensor);
    const auto& result_shape = result.logical_shape();
    ASSERT_EQ(result_shape.rank(), 4U);
    EXPECT_EQ(result_shape[0], batch_size);
    EXPECT_EQ(result_shape[1], 1U);
    EXPECT_EQ(result_shape[2], 1U);
    EXPECT_EQ(result_shape[3], features);

    std::vector<float> resulting_vector = ttml::core::to_vector(result);
    EXPECT_EQ(resulting_vector.size(), batch_size * features);
    for (int i = 0; i < resulting_vector.size(); ++i) {
        EXPECT_NEAR(resulting_vector[i], static_cast<float>(i) / static_cast<float>(i + 1), 1e-2);
    }
}

TEST_F(TrivialTnnFixedTest, TestSumOverBatch_1) {
    auto* device = &ttml::autograd::ctx().get_device();

    const size_t batch_size = 2U;
    const size_t features = 64U;
    std::vector<float> data(batch_size * features);
    float step = 0.1F;
    float value = 0.0F;
    for (int i = 0; i < data.size(); ++i) {
        data[i] = value;
        value += step;
    }

    auto shape = ttnn::Shape({batch_size, 1, 1, features});
    auto tensor = ttml::core::from_vector(data, shape, device);
    auto tensor_shape = tensor.logical_shape();
    EXPECT_EQ(tensor_shape[0], batch_size);
    EXPECT_EQ(tensor_shape[1], 1U);
    EXPECT_EQ(tensor_shape[2], 1U);
    EXPECT_EQ(tensor_shape[3], features);

    auto result = ttml::ttnn_fixed::sum_over_batch(tensor);
    const auto& result_shape = result.logical_shape();
    ASSERT_EQ(result_shape.rank(), 4U);
    EXPECT_EQ(result_shape[0], 1U);
    EXPECT_EQ(result_shape[1], 1U);
    EXPECT_EQ(result_shape[2], 1U);
    EXPECT_EQ(result_shape[3], features);

    std::vector<float> resulting_vector = ttml::core::to_vector(result);
    EXPECT_EQ(resulting_vector.size(), features);
    const float eps = 1.0F;
    for (int i = 0; i < resulting_vector.size(); ++i) {
        float expected_value = 0.F;
        for (int j = 0; j < batch_size; ++j) {
            expected_value += static_cast<float>(i + j * features) * step;
        }

        EXPECT_NEAR(expected_value, resulting_vector[i], eps);
    }
}
