// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"

class ProfilerNoOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

ttnn::Tensor make_row_major_input(
    const ttnn::Shape& shape,
    const tt::tt_metal::Alignment& alignment,
    tt::tt_metal::distributed::MeshDevice* device,
    float offset) {
    std::vector<::bfloat16> values;
    values.reserve(shape.volume());
    for (std::size_t i = 0; i < shape.volume(); ++i) {
        values.emplace_back(offset + static_cast<float>(i % 251U) / 16.0F);
    }

    const auto spec = tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            ttnn::DRAM_MEMORY_CONFIG,
            alignment));
    return ttnn::Tensor::from_vector(std::move(values), spec).to_device(device);
}

void expect_logical_identity(const ttnn::Tensor& input, const ttnn::Tensor& result) {
    const auto input_on_host = ttml::core::to_xtensor(input);
    const auto result_on_host = ttml::core::to_xtensor(result);
    EXPECT_EQ(result.layout(), tt::tt_metal::Layout::TILE);
    EXPECT_EQ(result.dtype(), tt::tt_metal::DataType::BFLOAT16);
    EXPECT_EQ(result_on_host.shape(), input_on_host.shape());
    EXPECT_TRUE(xt::allclose(result_on_host, input_on_host, /*rtol=*/0.0F, /*atol=*/0.0F));
}

}  // namespace

TEST_F(ProfilerNoOpTest, ProfilerNoOpTest_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 187U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, -10.0F, 10.0F, seed);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    const auto input_before = core::to_xtensor(input);
    auto result = ttml::metal::profiler_no_op(input, "identifier");

    const auto input_on_host = core::to_xtensor(input);
    const auto result_on_host = core::to_xtensor(result);
    EXPECT_EQ(result.layout(), tt::tt_metal::Layout::TILE);
    EXPECT_EQ(result.dtype(), tt::tt_metal::DataType::BFLOAT16);
    EXPECT_TRUE(xt::allclose(input_on_host, input_before, /*rtol=*/0.0F, /*atol=*/0.0F));
    EXPECT_EQ(result_on_host.shape(), input_on_host.shape());
    EXPECT_TRUE(xt::allclose(result_on_host, input_on_host, /*rtol=*/0.0F, /*atol=*/0.0F));
}

TEST_F(ProfilerNoOpTest, PreservesAlignmentBoundaryShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const std::array<ttnn::Shape, 2> shapes = {ttnn::Shape{1, 1, 1, 17}, ttnn::Shape{1, 1, 2, 1}};

    for (std::size_t i = 0; i < shapes.size(); ++i) {
        auto input = make_row_major_input(shapes[i], {}, device, 10.0F * static_cast<float>(i + 1U));
        auto result = ttml::metal::profiler_no_op(input, "alignment_boundary_" + std::to_string(i));
        expect_logical_identity(input, result);
    }
}

TEST_F(ProfilerNoOpTest, PreservesExplicitHeightPadding) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto input = make_row_major_input(ttnn::Shape{2, 1, 17, 33}, tt::tt_metal::Alignment({32, 64}), device, 3.0F);

    auto result = ttml::metal::profiler_no_op(input, "height_padding");

    expect_logical_identity(input, result);
}

TEST_F(ProfilerNoOpTest, ProgramCacheSeparatesDifferentPagePitches) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const ttnn::Shape shape{1, 1, 64, 32};
    auto compact = make_row_major_input(shape, {}, device, 1.0F);
    auto wide_pages = make_row_major_input(shape, tt::tt_metal::Alignment({64}), device, 100.0F);
    const auto entries_before = device->num_program_cache_entries();

    auto compact_result = ttml::metal::profiler_no_op(compact, "cache_pitch");
    const auto entries_after_compact = device->num_program_cache_entries();
    auto wide_result = ttml::metal::profiler_no_op(wide_pages, "cache_pitch");
    const auto entries_after_wide = device->num_program_cache_entries();

    EXPECT_EQ(entries_after_compact, entries_before + 1U);
    EXPECT_EQ(entries_after_wide, entries_after_compact + 1U);
    expect_logical_identity(compact, compact_result);
    expect_logical_identity(wide_pages, wide_result);
}
