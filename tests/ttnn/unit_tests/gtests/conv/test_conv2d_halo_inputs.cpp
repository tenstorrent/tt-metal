// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// conv2d's optional halo inputs, each against the default conv on the same shapes, on the L1 path and on both DRAM
// slicing paths:
// - prologue_scale / prologue_shift: conv2d(x) with the prologue against conv2d of ttnn::silu(x * scale + shift), the
//   separate passes the prologue replaces.
// - halo_boundary: conv2d of an image whose border comes from the boundary tensor against conv2d, without padding, of
//   the image with that border written in.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv_types.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp"

namespace ttnn::operations::conv::conv2d::test {

using tt::tt_metal::distributed::MeshDevice;
using SliceType = Conv2dSliceConfig::SliceType;

class Conv2dHaloInputsTest : public TTNNFixtureWithSuiteDevice<Conv2dHaloInputsTest> {
public:
    // halo keeps its kernel config tensors in L1_SMALL, which the default fixture does not reserve.
    static void SetUpTestSuite() {
        device_holder_ = ttnn::open_mesh_device(/*device_id=*/0, /*l1_small_size=*/16384, DEFAULT_TRACE_REGION_SIZE);
        device_ = device_holder_.get();
    }
};

namespace {

constexpr uint32_t H = 16, W = 64, C = 64, CO = 64;

std::vector<bfloat16> random_values(size_t n, uint32_t seed, float range) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-range, range);
    std::vector<bfloat16> out(n);
    for (auto& v : out) {
        v = bfloat16(dist(rng));
    }
    return out;
}

Tensor dram(MeshDevice& device, const std::vector<bfloat16>& data, const ttnn::Shape& shape, Layout layout) {
    const MemoryConfig mem(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    return Tensor::from_vector(data, TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(layout), mem)))
        .to_device(&device, mem);
}

struct Conv {
    std::array<uint32_t, 2> kernel{3, 3};
    std::array<uint32_t, 4> padding{1, 1, 1, 1};
    std::optional<Conv2dSliceConfig> slice;
    std::optional<Tensor> scale, shift, boundary;
};

std::vector<float> run(MeshDevice& device, const Tensor& input, uint32_t h, uint32_t w, const Conv& conv) {
    const Tensor weights = Tensor::from_vector(
        random_values(CO * C * conv.kernel[0] * conv.kernel[1], 7, 0.25f),
        TensorSpec(
            ttnn::Shape({CO, C, conv.kernel[0], conv.kernel[1]}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})));
    Conv2dConfig config;
    config.shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    const Tensor out = std::get<static_cast<int>(ConvResultType::OUTPUT)>(ttnn::conv2d_fused(
        input,
        weights,
        &device,
        C,
        CO,
        1,
        h,
        w,
        conv.kernel,
        {1, 1},
        conv.padding,
        {1, 1},
        1,
        std::nullopt,
        std::nullopt,
        config,
        std::nullopt,
        std::nullopt,
        conv.slice,
        false,
        false,
        {.prologue_scale = conv.scale, .prologue_shift = conv.shift, .halo_boundary = conv.boundary}));
    return out.to_vector<float>();
}

size_t mismatches(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 0.06f) {
    EXPECT_EQ(actual.size(), expected.size());
    size_t failures = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        failures += std::abs(actual[i] - expected[i]) > atol + 0.01f * std::abs(expected[i]);
    }
    return failures;
}

const std::array<std::optional<Conv2dSliceConfig>, 3> kPaths{
    Conv2dSliceConfig{.slice_type = SliceType::L1_FULL},
    Conv2dSliceConfig{.slice_type = SliceType::DRAM_WIDTH, .num_slices = 2},
    Conv2dSliceConfig{.slice_type = SliceType::DRAM_HEIGHT, .num_slices = 2}};

}  // namespace

// The L1 path applies the prologue in the halo; DRAM slicing applies it in each slice's padded_slice. The prologue and
// the separate ops round x * scale and + shift to bf16 at different points: ~0.02 rms on these outputs, where a
// misplaced or missing prologue moves them by ~1.
TEST_F(Conv2dHaloInputsTest, PrologueEqualsConvOfTheAffineSilu) {
    auto& device = *device_;
    const Tensor x = dram(device, random_values(H * W * C, 1, 2.0f), ttnn::Shape({1, H, W, C}), Layout::TILE);
    const Tensor scale = dram(device, random_values(C, 2, 1.5f), ttnn::Shape({1, 1, 1, C}), Layout::TILE);
    const Tensor shift = dram(device, random_values(C, 3, 1.0f), ttnn::Shape({1, 1, 1, C}), Layout::TILE);
    const Tensor y = ttnn::silu(ttnn::add(ttnn::multiply(x, scale), shift));
    for (const auto& slice : kPaths) {
        const auto expected = run(device, y, H, W, {.slice = slice});
        const auto actual = run(device, x, H, W, {.slice = slice, .scale = scale, .shift = shift});
        EXPECT_EQ(mismatches(actual, expected, 0.15f), 0u) << "slice type " << static_cast<int>(slice->slice_type);
    }
}

// Unsliced (L1), the boundary holds the whole image's border. In two width slices, the outer slices take their left /
// right border from it and every slice its top and bottom.
TEST_F(Conv2dHaloInputsTest, HaloBoundaryEqualsConvOfTheBorderedImage) {
    auto& device = *device_;
    const auto framed = random_values((H + 2) * (W + 2) * C, 4, 2.0f);
    const auto stick = [&](uint32_t r, uint32_t c) { return framed.begin() + ((r * (W + 2)) + c) * C; };
    std::vector<bfloat16> interior, border;
    for (uint32_t r = 1; r <= H; ++r) {
        interior.insert(interior.end(), stick(r, 1), stick(r, W + 1));
    }
    border.insert(border.end(), stick(0, 0), stick(1, 0));
    border.insert(border.end(), stick(H + 1, 0), stick(H + 2, 0));
    for (uint32_t c : {0u, W + 1}) {
        for (uint32_t r = 1; r <= H; ++r) {
            border.insert(border.end(), stick(r, c), stick(r, c + 1));
        }
    }
    for (const auto& slice : {kPaths[0], kPaths[1]}) {
        const auto expected =
            run(device,
                dram(device, framed, ttnn::Shape({1, H + 2, W + 2, C}), Layout::TILE),
                H + 2,
                W + 2,
                {.padding = {0, 0, 0, 0}, .slice = slice});
        const auto actual = run(
            device,
            dram(device, interior, ttnn::Shape({1, H, W, C}), Layout::TILE),
            H,
            W,
            {.slice = slice,
             .boundary = dram(
                 device, border, ttnn::Shape({1, 1, static_cast<uint32_t>(border.size() / C), C}), Layout::ROW_MAJOR)});
        EXPECT_EQ(mismatches(actual, expected), 0u) << "slice type " << static_cast<int>(slice->slice_type);
    }
}

// The prologue and the boundary run in the halo: refused where there is none (1x1 conv as a matmul), where the slices
// cannot place the boundary (height or automatic slicing), for a scale without its shift, and for a ROW_MAJOR input
// (an unsliced one would reach run_L1_op without a padded_slice to apply the prologue).
TEST_F(Conv2dHaloInputsTest, HaloInputsRejectedWhereTheyCannotRun) {
    auto& device = *device_;
    const Tensor x = dram(device, random_values(H * W * C, 5, 2.0f), ttnn::Shape({1, H, W, C}), Layout::TILE);
    const Tensor affine = dram(device, random_values(C, 6, 1.0f), ttnn::Shape({1, 1, 1, C}), Layout::TILE);
    const Tensor boundary = dram(
        device,
        random_values(2 * (W + 2 + H) * C, 8, 1.0f),
        ttnn::Shape({1, 1, 2 * (W + 2 + H), C}),
        Layout::ROW_MAJOR);
    EXPECT_ANY_THROW(run(device, x, H, W, {.slice = kPaths[0], .scale = affine}));
    EXPECT_ANY_THROW(
        run(device,
            x,
            H,
            W,
            {.kernel = {1, 1}, .padding = {0, 0, 0, 0}, .slice = kPaths[0], .scale = affine, .shift = affine}));
    EXPECT_ANY_THROW(run(device, x, H, W, {.slice = kPaths[2], .boundary = boundary}));
    EXPECT_ANY_THROW(run(device, x, H, W, {.boundary = boundary}));
    const Tensor x_rm = dram(device, random_values(H * W * C, 9, 2.0f), ttnn::Shape({1, H, W, C}), Layout::ROW_MAJOR);
    for (const auto& slice : kPaths) {
        EXPECT_ANY_THROW(run(device, x_rm, H, W, {.slice = slice, .scale = affine, .shift = affine}));
    }
}

}  // namespace ttnn::operations::conv::conv2d::test
