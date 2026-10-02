// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/cross_entropy_bw/device/cross_entropy_bw_device_operation.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"

class CrossEntropyBackwardTest : public ::testing::Test {
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

xt::xarray<float> calculate_cross_entropy_backward(
    const xt::xarray<float>& input, const xt::xarray<uint32_t>& target, const float scaler = 1.0F) {
    const uint32_t N = target.shape(0);
    const uint32_t H = target.shape(1);

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

TEST_F(CrossEntropyBackwardTest, ValidatesPreallocatedOutputContractOnCacheMissAndHit) {
    using Operation = ttml::metal::ops::cross_entropy_bw::device::CrossEntropyBackwardDeviceOperation;
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;

    constexpr uint32_t N = 2U;
    constexpr uint32_t H = 33U;
    constexpr uint32_t W = 65U;
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = ttml::core::from_xtensor(xt::zeros<float>({N, 1U, H, W}), device);
    auto target = ttml::core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        xt::zeros<uint32_t>({N, H}), device, ttnn::Layout::ROW_MAJOR);

    const Operation::operation_attributes_t attributes{.scaler = 1.0F};
    const auto expected_spec = Operation::compute_output_specs(
        attributes, Operation::tensor_args_t{.input = input, .target = target, .preallocated_output = std::nullopt});
    auto expected_output = ttnn::create_device_tensor(expected_spec, input.device());

    const auto undersized_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1U, 1U, 32U, 32U}),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    const auto wrong_shape_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1U, 1U, 66U, 65U}),
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

    auto result = ttnn::prim::ttml_cross_entropy_bw(input, target, attributes.scaler, expected_output);
    EXPECT_EQ(result.buffer()->address(), expected_output.buffer()->address());
}

TEST_F(CrossEntropyBackwardTest, ProgramCacheSeparatesTargetPagePitchAndRebindsAddresses) {
    using Operation = ttml::metal::ops::cross_entropy_bw::device::CrossEntropyBackwardDeviceOperation;

    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t num_dram_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const uint32_t N = 2U * num_dram_banks + 1U;
    constexpr uint32_t H = 32U;
    constexpr uint32_t W = 64U;
    constexpr float SCALER = 0.25F;

    const auto make_input_host = [&](uint32_t seed) {
        return ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, 1U, H, W}, -2.0F, 2.0F, seed);
    };
    const auto make_target_host = [&](uint32_t label) {
        auto target = xt::zeros<uint32_t>({N, H});
        target.fill(label);
        return target;
    };
    const auto make_target = [&](const xt::xarray<uint32_t>& host, uint32_t row_alignment) {
        const auto spec = tt::tt_metal::TensorSpec(
            ttnn::Shape({N, H}),
            tt::tt_metal::TensorLayout(
                ttnn::DataType::UINT32,
                ttnn::Layout::ROW_MAJOR,
                ttnn::DRAM_MEMORY_CONFIG,
                tt::tt_metal::Alignment({1U, row_alignment})));
        return ttnn::Tensor::from_vector(std::vector<uint32_t>(host.begin(), host.end()), spec, device);
    };

    const auto input_a_host = make_input_host(201U);
    const auto input_a_prime_host = make_input_host(202U);
    const auto input_b_host = make_input_host(203U);
    const auto input_b_prime_host = make_input_host(204U);
    const auto target_a_host = make_target_host(2U);
    const auto target_a_prime_host = make_target_host(3U);
    const auto target_b_host = make_target_host(1U);
    const auto target_b_prime_host = make_target_host(5U);

    auto input_a = ttml::core::from_xtensor(input_a_host, device);
    auto input_a_prime = ttml::core::from_xtensor(input_a_prime_host, device);
    auto input_b = ttml::core::from_xtensor(input_b_host, device);
    auto input_b_prime = ttml::core::from_xtensor(input_b_prime_host, device);
    auto target_a = make_target(target_a_host, 1U);
    auto target_a_prime = make_target(target_a_prime_host, 1U);
    auto target_b = make_target(target_b_host, 64U);
    auto target_b_prime = make_target(target_b_prime_host, 64U);

    ASSERT_EQ(target_a.buffer()->aligned_page_size(), H * sizeof(uint32_t));
    ASSERT_EQ(target_b.buffer()->aligned_page_size(), 64U * sizeof(uint32_t));
    EXPECT_NE(target_a.buffer()->address(), target_a_prime.buffer()->address());
    EXPECT_NE(target_b.buffer()->address(), target_b_prime.buffer()->address());

    const Operation::operation_attributes_t attributes{.scaler = SCALER};
    EXPECT_NE(
        Operation::compute_program_hash(
            attributes,
            Operation::tensor_args_t{.input = input_a, .target = target_a, .preallocated_output = std::nullopt}),
        Operation::compute_program_hash(
            attributes,
            Operation::tensor_args_t{.input = input_a, .target = target_b, .preallocated_output = std::nullopt}));

    device->disable_and_clear_program_cache();
    device->enable_program_cache();

    auto output_a = ttnn::prim::ttml_cross_entropy_bw(input_a, target_a, SCALER);
    const auto entries_after_a = device->num_program_cache_entries();
    EXPECT_GT(entries_after_a, 0U);

    auto output_a_prime = ttnn::prim::ttml_cross_entropy_bw(input_a_prime, target_a_prime, SCALER);
    const auto entries_after_a_prime = device->num_program_cache_entries();
    EXPECT_EQ(entries_after_a_prime, entries_after_a);

    // A has the smaller physical target pitch. Reusing A's program for B stays in bounds but
    // reads B's row padding as labels once a DRAM bank receives its second page.
    auto output_b = ttnn::prim::ttml_cross_entropy_bw(input_b, target_b, SCALER);
    const auto entries_after_b = device->num_program_cache_entries();
    EXPECT_GT(entries_after_b, entries_after_a_prime);

    auto output_b_prime = ttnn::prim::ttml_cross_entropy_bw(input_b_prime, target_b_prime, SCALER);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_b);

    EXPECT_TRUE(xt::allclose(
        ttml::core::to_xtensor(output_a),
        calculate_cross_entropy_backward(input_a_host, target_a_host, SCALER),
        3e-2F,
        1e-2F));
    EXPECT_TRUE(xt::allclose(
        ttml::core::to_xtensor(output_a_prime),
        calculate_cross_entropy_backward(input_a_prime_host, target_a_prime_host, SCALER),
        3e-2F,
        1e-2F));
    EXPECT_TRUE(xt::allclose(
        ttml::core::to_xtensor(output_b),
        calculate_cross_entropy_backward(input_b_host, target_b_host, SCALER),
        3e-2F,
        1e-2F));
    EXPECT_TRUE(xt::allclose(
        ttml::core::to_xtensor(output_b_prime),
        calculate_cross_entropy_backward(input_b_prime_host, target_b_prime_host, SCALER),
        3e-2F,
        1e-2F));

    device->disable_and_clear_program_cache();
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Small_Backward) {
    using namespace ttml;

    const uint32_t N = 1U, H = 1U;

    xt::xarray<float> input_tensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 1U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});
    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 91U, W = 187U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1024, W = 1024U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyBackward_Large_Backward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 32U, W = 128007U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, NIGHTLY_CrossEntropyBackward_Huge_Backward) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 64, W = 128000U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);
    xt::xarray<float> grad_tensor = xt::ones<float>({1U, 1U, 1U, 1U});

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto grad = core::from_xtensor(grad_tensor, &autograd::ctx().get_device());

    float scaler = 1.0F / (static_cast<float>(N) * static_cast<float>(H));

    auto result = ttml::metal::cross_entropy_bw(input, target, grad, scaler);

    auto expected_result = calculate_cross_entropy_backward(input_tensor, target_tensor, scaler);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyBackwardTest, CrossEntropyForwardBackward_ReduceMeanVsNone) {
    using namespace ttml;

    const uint32_t N = 5U, C = 1U, H = 91U, W = 187U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    auto& rng = ttml::autograd::ctx().get_generator();
    const uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);
    xt::xarray<uint32_t> target_tensor =
        ttml::test_utils::make_uniform_xarray<uint32_t>(std::array<std::size_t, 2>{N, H}, 0U, W - 1U, seed + 1U);

    auto input = ttml::autograd::create_tensor(
        core::from_xtensor(input_tensor, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto target = ttml::autograd::create_tensor(core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR));

    auto result_none = ttml::ops::cross_entropy_loss(input, target, ttml::ops::ReduceType::NONE);
    auto result_none_with_mean_after = ttml::ops::mean(result_none);
    auto result_mean = ttml::ops::cross_entropy_loss(input, target, ttml::ops::ReduceType::MEAN);

    result_mean->backward();
    result_none_with_mean_after->backward();

    auto result_mean_grad = core::to_xtensor(result_mean->get_grad());
    auto result_none_with_mean_after_grad = core::to_xtensor(result_none_with_mean_after->get_grad());

    auto result_none_after_mean_xtensor = core::to_xtensor(result_none_with_mean_after->get_value());
    auto result_mean_xtensor = core::to_xtensor(result_mean->get_value());

    assert((result_none_after_mean_xtensor.shape() == result_mean_xtensor.shape()));
    EXPECT_TRUE(xt::allclose(result_none_after_mean_xtensor, result_mean_xtensor, 3e-2F, 1e-2F));
    assert((result_none_with_mean_after_grad.shape() == result_mean_grad.shape()));
    EXPECT_TRUE(xt::allclose(result_none_with_mean_after_grad, result_mean_grad, 3e-2F, 1e-2F));
}
