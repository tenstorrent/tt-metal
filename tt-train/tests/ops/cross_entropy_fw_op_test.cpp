
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/cross_entropy_fw/device/cross_entropy_fw_device_operation.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"

class CrossEntropyForwardTest : public ::testing::Test {
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

TEST_F(CrossEntropyForwardTest, ValidatesPreallocatedOutputContractOnCacheMissAndHit) {
    using Operation = ttml::metal::ops::cross_entropy_fw::device::CrossEntropyForwardDeviceOperation;
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;

    constexpr uint32_t N = 2U;
    constexpr uint32_t H = 33U;
    constexpr uint32_t W = 65U;
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = ttml::core::from_xtensor(xt::zeros<float>({N, 1U, H, W}), device);
    auto target = ttml::core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        xt::zeros<uint32_t>({N, H}), device, ttnn::Layout::ROW_MAJOR);

    const Operation::operation_attributes_t attributes{};
    const auto expected_spec = Operation::compute_output_specs(
        attributes, Operation::tensor_args_t{.input = input, .target = target, .preallocated_output = std::nullopt});
    auto expected_output = ttnn::create_device_tensor(expected_spec, input.device());

    const auto undersized_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1U, 1U, 32U, 1U}),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    const auto wrong_shape_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1U, 1U, 128U, 1U}),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    const auto overpadded_spec = tt::tt_metal::TensorSpec(
        expected_spec.logical_shape(),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::Layout::TILE,
            ttnn::DRAM_MEMORY_CONFIG,
            tt::tt_metal::Alignment({64U, 64U})));
    const auto l1_output_spec = tt::tt_metal::TensorSpec(
        expected_spec.logical_shape(),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, ttnn::L1_MEMORY_CONFIG));

    auto undersized = ttnn::create_device_tensor(undersized_spec, input.device());
    auto wrong_shape = ttnn::create_device_tensor(wrong_shape_spec, input.device());
    auto overpadded = ttnn::create_device_tensor(overpadded_spec, input.device());
    auto l1_output = ttnn::create_device_tensor(l1_output_spec, input.device());

    const auto expect_rejected_on_miss_and_hit = [&](const Operation::tensor_args_t& args) {
        EXPECT_ANY_THROW(Adapter::validate_on_program_cache_miss(attributes, args));
        EXPECT_ANY_THROW(Adapter::validate_on_program_cache_hit(attributes, args));
    };
    for (const auto& output : {undersized, wrong_shape, overpadded, l1_output}) {
        expect_rejected_on_miss_and_hit(
            Operation::tensor_args_t{.input = input, .target = target, .preallocated_output = output});
    }

    const auto l1_input_spec = tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, ttnn::L1_MEMORY_CONFIG));
    const auto l1_target_spec = tt::tt_metal::TensorSpec(
        target.logical_shape(),
        tt::tt_metal::TensorLayout(ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR, ttnn::L1_MEMORY_CONFIG));
    const auto overpadded_input_spec = tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::Layout::TILE,
            ttnn::DRAM_MEMORY_CONFIG,
            tt::tt_metal::Alignment({64U, 64U})));
    auto l1_input = ttnn::create_device_tensor(l1_input_spec, input.device());
    auto l1_target = ttnn::create_device_tensor(l1_target_spec, input.device());
    auto overpadded_input = ttnn::create_device_tensor(overpadded_input_spec, input.device());
    expect_rejected_on_miss_and_hit(
        Operation::tensor_args_t{.input = l1_input, .target = target, .preallocated_output = std::nullopt});
    expect_rejected_on_miss_and_hit(
        Operation::tensor_args_t{.input = input, .target = l1_target, .preallocated_output = std::nullopt});
    expect_rejected_on_miss_and_hit(
        Operation::tensor_args_t{.input = overpadded_input, .target = target, .preallocated_output = std::nullopt});

    const Operation::tensor_args_t valid_args{.input = input, .target = target, .preallocated_output = expected_output};
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_miss(attributes, valid_args));
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_hit(attributes, valid_args));

    auto result = ttnn::prim::ttml_cross_entropy_fw(input, target, expected_output);
    EXPECT_EQ(result.buffer()->address(), expected_output.buffer()->address());
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Small_Forward) {
    using namespace ttml;

    const uint32_t N = 1, H = 1;

    xt::xarray<float> input_tensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 1U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Negetive_Values) {
    using namespace ttml;

    const uint32_t N = 1, H = 2;

    xt::xarray<float> input_tensor = {{{{-100.F, -101.F, -102.F, -103.F}, {-5.01F, -5.02F, -0.3F, -7.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 0;
    target_tensor(0, 1) = 2U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 157U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

// Disabled: non-deterministic accuracy failures — https://github.com/tenstorrent/tt-metal/issues/46121
TEST_F(CrossEntropyForwardTest, DISABLED_CrossEntropyForward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1017U, W = 1018U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Forward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 65536U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, NIGHTLY_CrossEntropyForward_Huge_Forward) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}
