// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/frobenius_normalize/device/frobenius_normalize_device_operation.hpp"
#include "test_utils/random_data.hpp"

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

ttnn::Tensor make_aligned_tensor(
    const std::vector<float>& data,
    const tt::tt_metal::Alignment& alignment,
    ttnn::distributed::MeshDevice* device,
    float pad_value = 0.0F) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape{1, 1, 64, 32},
        tt::tt_metal::TensorLayout(
            ttnn::DataType::BFLOAT16, ttnn::PageConfig(ttnn::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG, alignment));
    return ttnn::Tensor::from_vector<float>(data, spec, device, std::nullopt, pad_value);
}

void expect_matches_reference(const ttnn::Tensor& input, const ttnn::Tensor& output) {
    constexpr float kEps = 1e-7F;
    const auto expected = frobenius_normalize_ref(ttml::core::to_xtensor(input), kEps);
    const auto actual = ttml::core::to_xtensor(output);
    EXPECT_TRUE(xt::allclose(actual, expected, /*rtol=*/1e-2F, /*atol=*/1e-2F));
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

class FrobeniusNormalizeProgramCacheTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().get_device().disable_and_clear_program_cache();
    }
};

TEST_F(FrobeniusNormalizeProgramCacheTest, DistinguishesPaddedSpecsAndReusesChangedAddresses) {
    constexpr float kEps = 1e-7F;
    constexpr size_t kLogicalVolume = 1U * 1U * 64U * 32U;
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data_a(kLogicalVolume);
    std::vector<float> data_b(kLogicalVolume);
    std::vector<float> data_a2(kLogicalVolume);
    for (size_t i = 0; i < kLogicalVolume; ++i) {
        data_a[i] = static_cast<float>((i % 17U) + 1U) / 17.0F;
        data_b[i] = -static_cast<float>((i % 23U) + 1U) / 23.0F;
        data_a2[i] = static_cast<float>((i % 29U) + 1U) / 29.0F;
    }

    const tt::tt_metal::Alignment compact_alignment{32, 32};
    const tt::tt_metal::Alignment padded_alignment{32, 64};
    const auto input_a = make_aligned_tensor(data_a, compact_alignment, device);
    const auto input_b = make_aligned_tensor(data_b, padded_alignment, device);
    const auto input_a2 = make_aligned_tensor(data_a2, compact_alignment, device);
    auto output_a = make_aligned_tensor(std::vector<float>(kLogicalVolume, 7.0F), compact_alignment, device, 7.0F);
    auto output_b = make_aligned_tensor(std::vector<float>(kLogicalVolume, 11.0F), padded_alignment, device, 11.0F);
    auto output_a2 = make_aligned_tensor(std::vector<float>(kLogicalVolume, 13.0F), compact_alignment, device, 13.0F);

    device->enable_program_cache();
    device->clear_program_cache();

    const auto entries_before = device->num_program_cache_entries();
    const auto result_a = ttnn::prim::ttml_frobenius_normalize(input_a, kEps, output_a)[0];
    const auto entries_after_a = device->num_program_cache_entries();
    ASSERT_EQ(entries_after_a, entries_before + 1U);
    expect_matches_reference(input_a, result_a);

    const auto result_b = ttnn::prim::ttml_frobenius_normalize(input_b, kEps, output_b)[0];
    const auto entries_after_b = device->num_program_cache_entries();
    EXPECT_EQ(entries_after_b, entries_after_a + 1U)
        << "different physical shapes and preallocated output specs must compile distinct programs";
    expect_matches_reference(input_b, result_b);

    const auto result_a2 = ttnn::prim::ttml_frobenius_normalize(input_a2, kEps, output_a2)[0];
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_b)
        << "equal specs at new addresses should reuse the program after runtime address override";
    expect_matches_reference(input_a2, result_a2);
}
