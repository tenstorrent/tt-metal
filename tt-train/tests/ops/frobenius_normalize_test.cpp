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

ttnn::Tensor make_tiled_device_tensor(const tt::tt_metal::Tile& tile) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, 32, 32},
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16, ttnn::PageConfig(ttnn::Layout::TILE, tile), ttnn::DRAM_MEMORY_CONFIG));
    return ttnn::create_device_tensor(spec, &ttml::autograd::ctx().get_device());
}

constexpr auto kTileValidationError = "requires an untransposed 32x32 tile with four 16x16 faces";

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

class FrobeniusNormalizeValidationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(FrobeniusNormalizeValidationTest, RejectsNoncanonicalTilesOnColdAndWarmValidationPaths) {
    using Operation = ttml::metal::ops::frobenius_normalize::device::FrobeniusNormalizeDeviceOperation;
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;
    namespace fn_device = ttml::metal::ops::frobenius_normalize::device;

    const std::array noncanonical_tiles = {
        tt::tt_metal::Tile({16, 32}),
        tt::tt_metal::Tile({32, 16}),
        tt::tt_metal::Tile({16, 16}),
        tt::tt_metal::Tile({16, 32}, {8, 16}),
        tt::tt_metal::Tile({32, 32}, /* transpose_tile */ true),
    };

    for (const auto& tile : noncanonical_tiles) {
        SCOPED_TRACE(::testing::Message() << "tile=" << tile);
        const auto input = make_tiled_device_tensor(tile);
        const auto attributes = fn_device::FrobeniusNormalizeAttributes{};
        const auto tensor_args = fn_device::FrobeniusNormalizeTensorArgs{
            .input = input,
            .preallocated_output = std::nullopt,
        };

        EXPECT_THAT(
            [&] { Operation::validate_on_program_cache_miss(attributes, tensor_args); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(kTileValidationError)));
        EXPECT_THAT(
            [&] { Adapter::validate_on_program_cache_hit(attributes, tensor_args); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(kTileValidationError)));
    }
}

TEST_F(FrobeniusNormalizeValidationTest, AcceptsCanonicalTileOnColdAndWarmValidationPaths) {
    using Operation = ttml::metal::ops::frobenius_normalize::device::FrobeniusNormalizeDeviceOperation;
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;
    namespace fn_device = ttml::metal::ops::frobenius_normalize::device;

    const auto input = make_tiled_device_tensor(tt::tt_metal::Tile{});
    const auto attributes = fn_device::FrobeniusNormalizeAttributes{};
    const auto tensor_args = fn_device::FrobeniusNormalizeTensorArgs{
        .input = input,
        .preallocated_output = std::nullopt,
    };

    EXPECT_NO_THROW(Operation::validate_on_program_cache_miss(attributes, tensor_args));
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_hit(attributes, tensor_args));
}
