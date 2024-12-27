#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class DropoutTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(DropoutTest, TestSeed) {
    uint32_t dropout_seed1 = 42;
    uint32_t dropout_seed2 = 32;
    float scale = 2.0F;
    float prob = 0.5F;
    xt::random::seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    xt::xarray<float> xtensor_a = xt::random::rand({128 * 64}, -0.5, 0.5).reshape({2, 1, 64, 64});

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);
    auto num_cache_before = device->num_program_cache_entries();
    auto result01 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed1);
    auto result02 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed2);
    auto result11 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed1);
    auto result12 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed2);
    auto num_cache_after = device->num_program_cache_entries();

    auto result01_vec = ttml::core::to_xtensor(result01);
    auto result02_vec = ttml::core::to_xtensor(result02);
    auto result11_vec = ttml::core::to_xtensor(result11);
    auto result12_vec = ttml::core::to_xtensor(result12);

    EXPECT_TRUE(xt::allclose(result01_vec, result11_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(result02_vec, result12_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_FALSE(xt::allclose(result01_vec, result02_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_EQ(num_cache_before, num_cache_after - 1);
}
