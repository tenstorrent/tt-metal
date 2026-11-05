// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <numeric>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/device.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
#include "utils/memory_utils.hpp"

class ReduceOpTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void SetUp() override {
        ttml::autograd::ctx().set_seed(42);
    }
};

TEST_F(ReduceOpTest, TestMeanDim0) {
    ttml::autograd::ctx().set_seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor_a =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{2, 1, 64, 64}, -0.5F, 0.5F, seed);

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_mean_dim0 = ttml::ttnn_fixed::mean_ttnn(xtensor_a_tensor, 0, true);
    auto moreh_mean_dim0 = ttml::ttnn_fixed::mean_moreh(xtensor_a_tensor, 0, true);

    xt::xarray<float> mean_xtensor = xt::mean(xtensor_a, {0}, xt::evaluation_strategy::immediate);
    mean_xtensor.reshape({1, 1, 64, 64});

    auto mean_ttnn = ttml::core::to_xtensor(ttnn_mean_dim0);
    auto mean_moreh = ttml::core::to_xtensor(moreh_mean_dim0);

    EXPECT_TRUE(xt::allclose(mean_ttnn, mean_moreh, /*rtol=*/7e-2, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_ttnn, /*rtol=*/1e-3, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_moreh, /*rtol=*/1e-3, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestSumDim0) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor_a =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{2, 1, 64, 64}, -0.1F, 0.1F, seed);

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_sum_dim0 = ttml::ttnn_fixed::sum_ttnn(xtensor_a_tensor, 0, true);
    auto moreh_sum_dim0 = ttml::ttnn_fixed::sum_moreh(xtensor_a_tensor, 0, true);

    xt::xarray<float> sum_xtensor = xt::sum(xtensor_a, {0}, xt::evaluation_strategy::immediate);
    sum_xtensor.reshape({1, 1, 64, 64});

    auto sum_ttnn = ttml::core::to_xtensor(ttnn_sum_dim0);
    auto sum_moreh = ttml::core::to_xtensor(moreh_sum_dim0);

    EXPECT_TRUE(xt::allclose(sum_ttnn, sum_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(sum_xtensor, sum_ttnn, /*rtol=*/1e-2, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(sum_xtensor, sum_moreh, /*rtol=*/1e-2, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestMeanDim3) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor_a =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{2, 1, 64, 64}, -0.5F, 0.5F, seed);

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_mean_dim3 = ttml::ttnn_fixed::mean_ttnn(xtensor_a_tensor, 3, true);
    auto moreh_mean_dim3 = ttml::ttnn_fixed::mean_moreh(xtensor_a_tensor, 3, true);

    xt::xarray<float> mean_xtensor = xt::mean(xtensor_a, {3}, xt::evaluation_strategy::immediate);
    mean_xtensor.reshape({2, 1, 64, 1});

    auto mean_ttnn = ttml::core::to_xtensor(ttnn_mean_dim3);
    auto mean_moreh = ttml::core::to_xtensor(moreh_mean_dim3);
    EXPECT_TRUE(xt::allclose(mean_ttnn, mean_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_ttnn, /*rtol=*/1e-3, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_moreh, /*rtol=*/1e-3, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestSumDim3) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor_a =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{2, 1, 64, 64}, -0.1F, 0.1F, seed);

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_sum_dim3 = ttml::ttnn_fixed::sum_ttnn(xtensor_a_tensor, 3, true);
    auto moreh_sum_dim3 = ttml::ttnn_fixed::sum_moreh(xtensor_a_tensor, 3, true);

    xt::xarray<float> sum_xtensor = xt::sum(xtensor_a, {3}, xt::evaluation_strategy::immediate);
    sum_xtensor.reshape({2, 1, 64, 1});

    auto sum_ttnn = ttml::core::to_xtensor(ttnn_sum_dim3);
    auto sum_moreh = ttml::core::to_xtensor(moreh_sum_dim3);

    EXPECT_TRUE(xt::allclose(sum_ttnn, sum_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(sum_xtensor, sum_ttnn, /*rtol=*/1e-2, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(sum_xtensor, sum_moreh, /*rtol=*/1e-2, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestMeanLargeDim3) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> xtensor_a =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{2, 1, 512, 1024}, -0.5F, 0.5F, seed);

    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);

    auto ttnn_mean_dim3 = ttml::ttnn_fixed::mean_ttnn(xtensor_a_tensor, 3, true);
    auto moreh_mean_dim3 = ttml::ttnn_fixed::mean_moreh(xtensor_a_tensor, 3, true);

    xt::xarray<float> mean_xtensor = xt::mean(xtensor_a, {3}, xt::evaluation_strategy::immediate);
    mean_xtensor.reshape({2, 1, 512, 1});

    auto mean_ttnn = ttml::core::to_xtensor(ttnn_mean_dim3);
    auto mean_moreh = ttml::core::to_xtensor(moreh_mean_dim3);

    EXPECT_TRUE(xt::allclose(mean_ttnn, mean_moreh, /*rtol=*/1e-4, /*atol=*/1e-3));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_ttnn, /*rtol=*/1e-3, /*atol=*/1e-2));
    EXPECT_TRUE(xt::allclose(mean_xtensor, mean_moreh, /*rtol=*/1e-3, /*atol=*/1e-2));
}

TEST_F(ReduceOpTest, TestSumOverLeadingDims) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    const auto check = [&](const std::vector<std::size_t>& shape, const bool fp32) {
        const xt::xarray<float> data = ttml::test_utils::make_uniform_xarray<float>(shape, -0.5F, 0.5F, rng());
        const auto tensor = fp32 ? ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, device)
                                 : ttml::core::from_xtensor(data, device);
        // Reference over the values as stored on device.
        const xt::xarray<double> stored = xt::cast<double>(ttml::core::to_xtensor(tensor));
        std::vector<std::size_t> leading_dims(shape.size() - 1U);
        std::iota(leading_dims.begin(), leading_dims.end(), 0U);
        const xt::xarray<double> expected = xt::sum(stored, leading_dims, xt::keep_dims);

        const xt::xarray<double> result =
            xt::cast<double>(ttml::core::to_xtensor(ttml::ttnn_fixed::sum_over_leading_dims(tensor)));
        ASSERT_EQ(result.shape(), expected.shape());
        const double rtol = fp32 ? 1e-3 : 1e-2;
        const double atol = fp32 ? 1e-3 : 2e-2;
        EXPECT_TRUE(xt::allclose(result, expected, rtol, atol))
            << (fp32 ? "fp32" : "bf16") << " shape " << ::testing::PrintToString(shape);
    };
    // S that is a whole number of tiles (the dims before S are folded into the rows) and S that is not (each dim larger
    // than 1 is reduced on its own), with size-1 dims in different positions, heads with and without a batch, one row
    // per batch, widths that need padding, and a single row (nothing to reduce).
    const std::vector<std::vector<std::size_t>> shapes = {
        {1, 1, 64, 96},
        {3, 1, 64, 80},
        {2, 4, 32, 64},
        {1, 4, 32, 64},
        {2, 1, 50, 80},
        {1, 1, 650, 80},
        {4, 1, 1, 96},
        {2, 3, 1, 64},
        {1, 1, 1, 96},
        {3, 64, 80},
        {2, 50, 80},
        {64, 96}};
    for (const auto& shape : shapes) {
        check(shape, /* fp32 */ false);
    }
    // fp32 input (polynorm_bw's partials) takes different ttnn::sum paths: the folded rows here also split S across
    // cores.
    for (const auto& shape :
         std::vector<std::vector<std::size_t>>{{4, 1, 256, 128}, {2, 1, 50, 128}, {1, 1, 64, 128}}) {
        check(shape, /* fp32 */ true);
    }
}

// Neither a size-1 leading dim nor a batch may cost a full-size copy. For these bf16 inputs, ttnn::sum over {0, 1, 2}
// writes the whole tensor again in fp32 through fast_reduce_nc.
TEST_F(ReduceOpTest, TestSumOverLeadingDimsMakesNoFullSizeCopy) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto& rng = ttml::autograd::ctx().get_generator();
    const std::vector<std::array<std::size_t, 4>> shapes = {{1, 1, 2048, 1024}, {4, 1, 512, 1024}};
    for (const auto& shape : shapes) {
        const auto tensor =
            ttml::core::from_xtensor(ttml::test_utils::make_uniform_xarray<float>(shape, -0.5F, 0.5F, rng()), device);
        const auto input_bytes = static_cast<long long>(tensor.padded_shape().volume()) * tensor.element_size();

        // Compile the program first, so the capture below only sees what the reduction allocates.
        (void)ttml::ttnn_fixed::sum_over_leading_dims(tensor);

        ttml::utils::MemoryUsageTracker::clear();
        long long peak = 0;
        {
            const auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
            (void)ttml::ttnn_fixed::sum_over_leading_dims(tensor);
            ttml::utils::MemoryUsageTracker::end_capture("sum_over_leading_dims");
            // The guard clears the stored traces when it is destroyed, so read the peak before the block ends.
            peak = ttml::utils::MemoryUsageTracker::get_dram_usage("sum_over_leading_dims").peak;
        }
        EXPECT_LT(peak, input_bytes / 4) << "peak DRAM " << peak << " B while reducing a " << input_bytes << " B "
                                         << shape[0] << "x" << shape[1] << "x" << shape[2] << "x" << shape[3]
                                         << " input";
    }
}
