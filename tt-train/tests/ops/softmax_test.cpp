// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/types.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <ttnn/distributed/types.hpp>
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/softmax/device/softmax_device_operation.hpp"
#include "metal/ops/softmax_backward/device/softmax_backward_device_operation.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

using shape_type = std::array<std::size_t, 4>;

namespace {

class ProgramCacheGuard {
public:
    explicit ProgramCacheGuard(ttnn::distributed::MeshDevice* device) : device_(device) {
        device_->set_program_cache_misses_allowed(true);
        device_->disable_and_clear_program_cache();
        device_->enable_program_cache();
    }

    ~ProgramCacheGuard() {
        device_->set_program_cache_misses_allowed(true);
        device_->disable_and_clear_program_cache();
    }

private:
    ttnn::distributed::MeshDevice* device_;
};

template <typename Adapter, typename Attributes, typename TensorArgs>
void expect_validation_failure_on_miss_and_hit(
    const Attributes& attributes, const TensorArgs& tensor_args, std::string_view diagnostic) {
    EXPECT_THAT(
        ([&] { Adapter::validate_on_program_cache_miss(attributes, tensor_args); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(std::string(diagnostic))));
    EXPECT_THAT(
        ([&] { Adapter::validate_on_program_cache_hit(attributes, tensor_args); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(std::string(diagnostic))));
}

using SoftmaxOp = ttml::metal::ops::softmax::device::SoftmaxDeviceOperation;
using SoftmaxAdapter = ttnn::device_operation::MeshDeviceOperationAdapter<SoftmaxOp>;
using SoftmaxBackwardOp = ttml::metal::ops::softmax_backward::device::SoftmaxBackwardDeviceOperation;
using SoftmaxBackwardAdapter = ttnn::device_operation::MeshDeviceOperationAdapter<SoftmaxBackwardOp>;

}  // namespace

class SoftmaxTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> xt_softmax(const xt::xarray<float>& input, uint32_t dim = 3U) {
    xt::xarray<float> max_value = xt::amax(input, dim, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_value;  // for numerical stability
    xt::xarray<float> exp_shifted_input = xt::exp(shifted_input);
    xt::xarray<float> exp_sum = xt::sum(exp_shifted_input, dim, xt::keep_dims);
    xt::xarray<float> result = exp_shifted_input / exp_sum;
    return result;
}

ttnn::Tensor make_overpadded_tensor(
    const xt::xarray<float>& host_tensor, ttnn::distributed::MeshDevice* device, const ttnn::Shape& padded_shape) {
    auto row_major = ttml::core::from_xtensor(host_tensor, device, ttnn::Layout::ROW_MAJOR);
    return ttnn::tilize_with_val_padding(row_major, padded_shape, 0.0F);
}

TEST_F(SoftmaxTest, ProgramCacheKeysPhysicalPaddingAndPreservesOutputSpec) {
    auto* device = &ttml::autograd::ctx().get_device();

    constexpr shape_type shape{1U, 2U, 59U, 64U};
    const auto host_first = ttml::test_utils::make_uniform_xarray<float>(shape, -3.0F, 3.0F, 70U);
    const auto host_second = ttml::test_utils::make_uniform_xarray<float>(shape, -3.0F, 3.0F, 76U);
    const auto expected_first = xt_softmax(host_first);
    const auto expected_second = xt_softmax(host_second);

    auto default_input = ttml::core::from_xtensor(host_first, device);
    auto default_preallocated = ttnn::create_device_tensor(default_input.tensor_spec(), device);
    auto overpadded_input = make_overpadded_tensor(host_first, device, ttnn::Shape({1, 2, 96, 64}));
    auto overpadded_preallocated = ttnn::create_device_tensor(overpadded_input.tensor_spec(), device);
    auto fresh_overpadded_input = make_overpadded_tensor(host_second, device, ttnn::Shape({1, 2, 96, 64}));
    auto fresh_overpadded_preallocated = ttnn::create_device_tensor(fresh_overpadded_input.tensor_spec(), device);

    ASSERT_EQ(overpadded_input.tensor_spec(), fresh_overpadded_input.tensor_spec());
    ASSERT_NE(overpadded_input.buffer()->address(), fresh_overpadded_input.buffer()->address());
    ASSERT_NE(overpadded_preallocated.buffer()->address(), fresh_overpadded_preallocated.buffer()->address());

    ProgramCacheGuard cache_guard(device);
    ASSERT_EQ(device->num_program_cache_entries(), 0U);

    const auto entries_before_default = device->num_program_cache_entries();
    auto default_output = ttnn::prim::ttml_softmax(default_input, 3, default_preallocated);
    const auto entries_after_default = device->num_program_cache_entries();
    ASSERT_GT(entries_after_default, entries_before_default);
    EXPECT_EQ(default_output.tensor_spec(), default_input.tensor_spec());
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(default_output), expected_first, 3e-2F, 1e-2F));

    const auto entries_before_overpadded = device->num_program_cache_entries();
    auto overpadded_output = ttnn::prim::ttml_softmax(overpadded_input, 3, overpadded_preallocated);
    const auto entries_after_overpadded = device->num_program_cache_entries();
    EXPECT_GT(entries_after_overpadded, entries_before_overpadded)
        << "different physical row geometry must compile a distinct program";
    EXPECT_EQ(overpadded_output.tensor_spec(), overpadded_input.tensor_spec());
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(overpadded_output), expected_first, 3e-2F, 1e-2F));

    const auto entries_before_hit = device->num_program_cache_entries();
    device->set_program_cache_misses_allowed(false);
    auto fresh_overpadded_output = ttnn::prim::ttml_softmax(fresh_overpadded_input, 3, fresh_overpadded_preallocated);
    device->set_program_cache_misses_allowed(true);
    EXPECT_EQ(device->num_program_cache_entries(), entries_before_hit)
        << "fresh addresses with the same physical spec must reuse the cached program";
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(fresh_overpadded_output), expected_second, 3e-2F, 1e-2F));

    const SoftmaxOp::operation_attributes_t attributes{.dim = 3};
    const SoftmaxOp::tensor_args_t tensor_args{.input = overpadded_input, .preallocated_output = std::nullopt};
    ASSERT_EQ(SoftmaxOp::compute_output_specs(attributes, tensor_args), overpadded_input.tensor_spec());
    auto automatic_output = ttml::metal::softmax(overpadded_input, 3);
    EXPECT_EQ(automatic_output.tensor_spec(), overpadded_input.tensor_spec());
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(automatic_output), expected_first, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, RejectsUnsupportedBuffersAndMismatchedPreallocatedOutput) {
    auto* device = &ttml::autograd::ctx().get_device();

    constexpr shape_type shape{1U, 1U, 64U, 64U};
    const auto host = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, 71U);
    auto input = ttml::core::from_xtensor(host, device);
    const SoftmaxOp::operation_attributes_t attributes{.dim = 3};

    auto l1_input = ttml::ttnn_fixed::to_l1_interleaved(input);
    expect_validation_failure_on_miss_and_hit<SoftmaxAdapter>(
        attributes,
        SoftmaxOp::tensor_args_t{.input = l1_input, .preallocated_output = std::nullopt},
        "Tensor 'Input' must be stored in DRAM");

    const auto wrong_output_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, 32, 32}),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
    auto wrong_output = ttnn::create_device_tensor(wrong_output_spec, device);
    expect_validation_failure_on_miss_and_hit<SoftmaxAdapter>(
        attributes,
        SoftmaxOp::tensor_args_t{.input = input, .preallocated_output = wrong_output},
        "Preallocated softmax output must have the same tensor spec as the input");

    constexpr shape_type width_padded_shape{1U, 1U, 32U, 33U};
    const auto width_padded_host = ttml::test_utils::make_uniform_xarray<float>(width_padded_shape, -1.0F, 1.0F, 72U);
    auto width_overpadded = make_overpadded_tensor(width_padded_host, device, ttnn::Shape({1, 1, 32, 96}));
    expect_validation_failure_on_miss_and_hit<SoftmaxAdapter>(
        attributes,
        SoftmaxOp::tensor_args_t{.input = width_overpadded, .preallocated_output = std::nullopt},
        "Softmax only supports padding in the height dimension");

    const auto custom_tile_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, 32, 64}),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, tt::tt_metal::Tile({16, 32})),
            ttnn::DRAM_MEMORY_CONFIG));
    auto custom_tile_input = ttnn::create_device_tensor(custom_tile_spec, device);
    expect_validation_failure_on_miss_and_hit<SoftmaxAdapter>(
        attributes,
        SoftmaxOp::tensor_args_t{.input = custom_tile_input, .preallocated_output = std::nullopt},
        "Tensor 'Input' must use the canonical 32x32 tile");

    const auto transposed_tile_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, 32, 64}),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, tt::tt_metal::Tile({32, 32}, /*transpose_tile=*/true)),
            ttnn::DRAM_MEMORY_CONFIG));
    auto transposed_tile_input = ttnn::create_device_tensor(transposed_tile_spec, device);
    expect_validation_failure_on_miss_and_hit<SoftmaxAdapter>(
        attributes,
        SoftmaxOp::tensor_args_t{.input = transposed_tile_input, .preallocated_output = std::nullopt},
        "Tensor 'Input' must use the canonical 32x32 tile");
}

class SoftmaxMultiDeviceContractTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(SoftmaxMultiDeviceContractTest, RejectsForeignOutputAndGradient) {
    auto& parent_mesh = ttml::autograd::ctx().get_device();
    const auto input_mesh = parent_mesh.create_submesh(
        tt::tt_metal::distributed::MeshShape(1, 1), tt::tt_metal::distributed::MeshCoordinate(0, 0));
    const auto other_mesh = parent_mesh.create_submesh(
        tt::tt_metal::distributed::MeshShape(1, 1), tt::tt_metal::distributed::MeshCoordinate(0, 1));

    const xt::xarray<float> data = xt::ones<float>({1, 1, 32, 64});
    const auto input = ttml::core::from_xtensor(data, input_mesh.get());
    const auto foreign_tensor = ttml::core::from_xtensor(data, other_mesh.get());

    const SoftmaxOp::operation_attributes_t softmax_attributes{.dim = 3};
    expect_validation_failure_on_miss_and_hit<SoftmaxAdapter>(
        softmax_attributes,
        SoftmaxOp::tensor_args_t{.input = input, .preallocated_output = foreign_tensor},
        "Preallocated softmax output must be on the same device as the input");

    const SoftmaxBackwardOp::operation_attributes_t backward_attributes{.dim = 3, .sub_core_grids = std::nullopt};
    expect_validation_failure_on_miss_and_hit<SoftmaxBackwardAdapter>(
        backward_attributes,
        SoftmaxBackwardOp::tensor_args_t{.softmax_output = input, .upstream_grad = foreign_tensor},
        "Softmax output and upstream gradient must be on the same device");
}

// Disabled: flaky — https://github.com/tenstorrent/tt-metal/issues/46422
TEST_F(SoftmaxTest, DISABLED_SoftmaxTest_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 59U, W = 197U;
    const auto shape = shape_type{N, C, H, W};
    int32_t dim = 3U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float, shape_type>(shape, -10.0F, 10.0F, seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    ttnn::Tensor ttml_softmax = ttml::metal::softmax(input, dim);
    auto ttml_softmax_xtensor = core::to_xtensor(ttml_softmax);

    ttnn::Tensor ttnn_softmax = ttnn_fixed::softmax(input, dim);
    auto ttnn_softmax_xtensor = core::to_xtensor(ttnn_softmax);

    // Host side reference using FP32 and xtensor
    auto expected_result = xt_softmax(input_tensor, dim);

    ASSERT_EQ(ttml_softmax_xtensor.shape(), expected_result.shape());

    // ttml vs host
    EXPECT_TRUE(xt::allclose(ttml_softmax_xtensor, expected_result, 3e-2F, 1e-2F));

    // ttnn vs host
    EXPECT_TRUE(xt::allclose(ttnn_softmax_xtensor, expected_result, 3e-2F, 1e-2F));

    // ttml vs ttnn
    EXPECT_TRUE(xt::allclose(ttml_softmax_xtensor, ttnn_softmax_xtensor, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, SoftmaxTest_Big_Batch) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 32U, W = 128007U;
    const auto shape = shape_type{N, C, H, W};
    int32_t dim = 3U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor = ttml::test_utils::make_uniform_xarray<float>(shape, -10.0F, 10.0F, seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto result = ttml::metal::softmax(input, dim);

    auto expected_result = xt_softmax(input_tensor, dim);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    ASSERT_EQ(result_xtensor.shape(), expected_result.shape());
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, NIGHTLY_SoftmaxTest_Huge_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    const auto shape = shape_type{N, C, H, W};
    int32_t dim = 3U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float, shape_type>(shape, -10.0F, 10.0F, seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto result = ttml::metal::softmax(input, dim);

    auto expected_result = xt_softmax(input_tensor, dim);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    ASSERT_EQ(result_xtensor.shape(), expected_result.shape());
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(SoftmaxTest, SoftmaxTest_Large_Values) {
    using namespace ttml;

    int32_t dim = 3U;

    xt::xarray<float> input_tensor = {
        {{{5.36871e+08,  -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08,
           -9.98244e+08, -9.98244e+08, -9.98244e+08, -9.98244e+08}}}};

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    ttnn::Tensor ttml_softmax = ttml::metal::softmax(input, dim);
    auto ttml_softmax_xtensor = core::to_xtensor(ttml_softmax);

    ttnn::Tensor ttnn_softmax = ttnn_fixed::softmax(input, dim);
    auto ttnn_softmax_xtensor = core::to_xtensor(ttnn_softmax);

    // Host side reference using FP32 and xtensor
    auto expected_result = xt_softmax(input_tensor, dim);

    ASSERT_EQ(ttml_softmax_xtensor.shape(), expected_result.shape());

    // ttml vs host
    EXPECT_TRUE(xt::allclose(ttml_softmax_xtensor, expected_result, 3e-2F, 1e-2F));

    // ttnn vs host
    EXPECT_TRUE(xt::allclose(ttnn_softmax_xtensor, expected_result, 3e-2F, 1e-2F));

    // ttml vs ttnn
    EXPECT_TRUE(xt::allclose(ttml_softmax_xtensor, ttnn_softmax_xtensor, 3e-2F, 1e-2F));
}
