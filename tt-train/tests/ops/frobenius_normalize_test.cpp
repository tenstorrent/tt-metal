// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/frobenius_normalize/device/frobenius_normalize_device_operation.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"

namespace {

struct FrobeniusCase {
    std::array<uint32_t, 4> shape;
    std::string name;
};

xt::xarray<float> frobenius_normalize_ref(const xt::xarray<float>& X, float eps) {
    const auto squares = X * X;
    const float sum_sq = xt::sum(squares)();
    const float norm = std::sqrt(sum_sq) + eps;
    return X / norm;
}

tt::tt_metal::TensorSpec make_frobenius_spec(
    const tt::tt_metal::MemoryConfig& memory_config,
    const tt::tt_metal::Alignment& alignment,
    const tt::tt_metal::Tile& tile = tt::tt_metal::Tile{}) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, 64, 32},
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16, tt::tt_metal::PageConfig(ttnn::Layout::TILE, tile), memory_config, alignment));
}

ttnn::Tensor make_frobenius_tensor(
    float offset, const tt::tt_metal::MemoryConfig& memory_config, const tt::tt_metal::Alignment& alignment) {
    const auto spec = make_frobenius_spec(memory_config, alignment);
    std::vector<float> data(spec.logical_shape().volume());
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = offset + static_cast<float>((i % 17U) + 1U) / 32.0F;
    }
    return ttnn::Tensor::from_vector<float>(
        data,
        spec,
        &ttml::autograd::ctx().get_device(),
        std::nullopt,
        /* pad_value=*/0.0F);
}

void expect_frobenius_matches_reference(const ttnn::Tensor& input, const ttnn::Tensor& output, float epsilon) {
    const auto expected = frobenius_normalize_ref(ttml::core::to_xtensor(input), epsilon);
    const auto actual = ttml::core::to_xtensor(output);
    EXPECT_TRUE(xt::allclose(actual, expected, /*rtol=*/1e-2F, /*atol=*/1e-2F));
}

constexpr auto kOutputTileValidationError =
    "requires preallocated output to use an untransposed 32x32 tile with four 16x16 faces";

using FrobeniusOperation = ttml::metal::ops::frobenius_normalize::device::FrobeniusNormalizeDeviceOperation;
using FrobeniusAdapter = ttnn::device_operation::MeshDeviceOperationAdapter<FrobeniusOperation>;

void expect_frobenius_validation_failure_on_miss_and_hit(
    const FrobeniusOperation::operation_attributes_t& attributes,
    const FrobeniusOperation::tensor_args_t& tensor_args,
    std::string_view diagnostic) {
    EXPECT_THAT(
        [&] { FrobeniusOperation::validate_on_program_cache_miss(attributes, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(std::string(diagnostic))));
    EXPECT_THAT(
        [&] { FrobeniusAdapter::validate_on_program_cache_hit(attributes, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(std::string(diagnostic))));
}

}  // namespace

class FrobeniusNormalizeTest : public ::testing::TestWithParam<FrobeniusCase> {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(FrobeniusNormalizeTest, MatchesCpuReference) {
    using namespace ttml;

    const auto& c = GetParam();
    constexpr float kEps = 1e-7f;

    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(c.name));
    const auto data = ttml::test_utils::make_uniform_xarray<float>(c.shape, -1.0f, 1.0f, seed);
    const auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());

    const auto bf16_data = core::to_xtensor(input_tensor);
    const auto expected = frobenius_normalize_ref(bf16_data, kEps);

    const auto result_tensor = metal::frobenius_normalize(input_tensor, kEps);
    const auto result = core::to_xtensor(result_tensor);

    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f));
}

static std::string CaseName(const ::testing::TestParamInfo<FrobeniusCase>& info) {
    return info.param.name;
}

static const FrobeniusCase kCases[] = {
    {{1, 1, 32, 32}, "SingleTile"},
    {{1, 1, 32, 64}, "TwoTiles"},
    {{1, 1, 32, 96}, "ThreeTiles"},
    {{1, 1, 64, 64}, "SmallMatrix"},
    {{1, 1, 256, 320}, "MediumMatrix"},
    {{1, 1, 2048, 5632}, "ProductionSize"},
};

INSTANTIATE_TEST_SUITE_P(All, FrobeniusNormalizeTest, ::testing::ValuesIn(kCases), CaseName);

class FrobeniusNormalizeContractTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        ttml::autograd::ctx().get_device().disable_and_clear_program_cache();
    }
};

TEST_F(FrobeniusNormalizeContractTest, PreservesPaddedOutputSpecOnColdAndWarmRuns) {
    constexpr float kEps = 1e-7F;
    const tt::tt_metal::Alignment padded_alignment{32, 64};
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    auto cold_input = make_frobenius_tensor(0.0F, ttnn::DRAM_MEMORY_CONFIG, padded_alignment);
    auto warm_input = make_frobenius_tensor(0.5F, ttnn::DRAM_MEMORY_CONFIG, padded_alignment);
    ASSERT_EQ(cold_input.padded_shape(), ttnn::Shape({1, 1, 64, 64}));
    ASSERT_EQ(cold_input.buffer()->num_pages(), 4U);
    device->clear_program_cache();

    const auto entries_before = device->num_program_cache_entries();
    const auto cold_output = ttml::metal::frobenius_normalize(cold_input, kEps);
    const auto entries_after_cold = device->num_program_cache_entries();
    ASSERT_GT(entries_after_cold, entries_before) << "cold Frobenius run did not populate the program cache";
    EXPECT_EQ(cold_output.tensor_spec(), cold_input.tensor_spec());
    EXPECT_EQ(cold_output.buffer()->num_pages(), cold_input.buffer()->num_pages());
    expect_frobenius_matches_reference(cold_input, cold_output, kEps);

    const auto warm_output = ttml::metal::frobenius_normalize(warm_input, kEps);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_cold)
        << "same-spec warm Frobenius run should reuse the cached program";
    EXPECT_EQ(warm_output.tensor_spec(), warm_input.tensor_spec());
    EXPECT_EQ(warm_output.buffer()->num_pages(), warm_input.buffer()->num_pages());
    expect_frobenius_matches_reference(warm_input, warm_output, kEps);
}

TEST_F(FrobeniusNormalizeContractTest, RejectsL1InputAndOutputOnMissAndHitValidationPaths) {
    const tt::tt_metal::Alignment alignment{32, 32};
    auto* device = &ttml::autograd::ctx().get_device();

    auto dram_input = make_frobenius_tensor(0.0F, ttnn::DRAM_MEMORY_CONFIG, alignment);
    auto l1_input = make_frobenius_tensor(0.25F, ttnn::L1_MEMORY_CONFIG, alignment);
    auto l1_output = ttnn::create_device_tensor(make_frobenius_spec(ttnn::L1_MEMORY_CONFIG, alignment), device);
    const FrobeniusOperation::operation_attributes_t attributes{};

    expect_frobenius_validation_failure_on_miss_and_hit(
        attributes,
        FrobeniusOperation::tensor_args_t{.input = l1_input, .preallocated_output = std::nullopt},
        "requires input in DRAM");
    expect_frobenius_validation_failure_on_miss_and_hit(
        attributes,
        FrobeniusOperation::tensor_args_t{.input = dram_input, .preallocated_output = l1_output},
        "requires preallocated output in DRAM");
}

TEST_F(FrobeniusNormalizeContractTest, RejectsPreallocatedOutputWithDifferentPhysicalMapping) {
    constexpr float kEps = 1e-7F;
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = make_frobenius_tensor(0.0F, ttnn::DRAM_MEMORY_CONFIG, tt::tt_metal::Alignment{32, 64});
    auto mismatched_output = ttnn::create_device_tensor(
        make_frobenius_spec(ttnn::DRAM_MEMORY_CONFIG, tt::tt_metal::Alignment{128, 32}), device);

    ASSERT_EQ(input.buffer()->num_pages(), mismatched_output.buffer()->num_pages());
    ASSERT_NE(input.tensor_spec(), mismatched_output.tensor_spec());
    expect_frobenius_validation_failure_on_miss_and_hit(
        FrobeniusOperation::operation_attributes_t{.epsilon = kEps},
        FrobeniusOperation::tensor_args_t{.input = input, .preallocated_output = mismatched_output},
        "Preallocated output TensorSpec must match input TensorSpec exactly");
}

TEST_F(FrobeniusNormalizeContractTest, RejectsTransposedPreallocatedOutputOnColdAndWarmValidationPaths) {
    auto* device = &ttml::autograd::ctx().get_device();
    const tt::tt_metal::Alignment alignment{32, 32};
    const auto input = ttnn::create_device_tensor(make_frobenius_spec(ttnn::DRAM_MEMORY_CONFIG, alignment), device);
    const auto transposed_output = ttnn::create_device_tensor(
        make_frobenius_spec(
            ttnn::DRAM_MEMORY_CONFIG, alignment, tt::tt_metal::Tile({32, 32}, /* transpose_tile */ true)),
        device);
    const auto attributes = FrobeniusOperation::operation_attributes_t{};
    const auto tensor_args = FrobeniusOperation::tensor_args_t{
        .input = input,
        .preallocated_output = transposed_output,
    };

    expect_frobenius_validation_failure_on_miss_and_hit(attributes, tensor_args, kOutputTileValidationError);
}
