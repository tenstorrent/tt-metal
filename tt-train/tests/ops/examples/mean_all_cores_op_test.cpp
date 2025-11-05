// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include <umd/device/cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class MeanAllCoresTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> compute_mean_all_dims(const xt::xarray<float>& input_tensor) {
    // Compute mean across all dimensions except batch dimension (dim 0)
    // xt::xarray<float> result =  xt::empty<float>({1U, 1U, 1U, 1U});
    xt::xarray<float> result = xt::sum(input_tensor, {2, 3}, xt::keep_dims);
    return result;
}

TEST_F(MeanAllCoresTest, MeanAllCoresTest_Small_Batch) {
    using namespace ttml;

    const uint32_t B = 1U, H = 1U, S = 64U, W = 128U;

    std::random_device rd;
    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::empty<float>({B, H, S, W});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{input_tensor.data(), input_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); },
        seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());
    std::cout << "Input Logits:\n";
    input.print();

    auto result = ttml::metal::mean_all_cores(input);
    xt::xarray<float> result_xtensor = core::to_xtensor(result);

    xt::xarray<float> expected_result = compute_mean_all_dims(input_tensor);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F, 1e-2F));
}
