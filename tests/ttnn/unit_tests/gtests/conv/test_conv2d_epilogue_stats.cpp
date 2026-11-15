// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// conv2d's optional epilogue_stats: the per output channel sum and sum of squares of the conv output, written while
// the output is packed. Against the same conv without it, on the L1 path and on both DRAM slicing paths:
// - the output is unchanged, bit for bit;
// - the stats equal fp32 sums over that output on the host;
// - a conv with stats compiles as many programs as without (a slice's page base is not in the program hash).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp"

namespace ttnn::operations::conv::conv2d::epilogue_stats_test {

using tt::tt_metal::distributed::MeshDevice;
using SliceType = Conv2dSliceConfig::SliceType;

class Conv2dEpilogueStatsTest : public TTNNFixtureWithSuiteDevice<Conv2dEpilogueStatsTest> {
public:
    // halo keeps its kernel config tensors in L1_SMALL, which the default fixture does not reserve.
    static void SetUpTestSuite() {
        device_holder_ = ttnn::open_mesh_device(/*device_id=*/0, /*l1_small_size=*/16384, DEFAULT_TRACE_REGION_SIZE);
        device_ = device_holder_.get();
    }
};

namespace {

constexpr uint32_t H = 16, W = 64, C = 64, CO = 64, NTILES = CO / 32;

std::vector<bfloat16> random_values(size_t n, uint32_t seed, float range) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-range, range);
    std::vector<bfloat16> out(n);
    for (auto& v : out) {
        v = bfloat16(dist(rng));
    }
    return out;
}

Tensor dram(MeshDevice& device, const TensorSpec& spec, const auto& data) {
    return Tensor::from_vector(data, spec).to_device(&device, spec.memory_config());
}

const MemoryConfig kDram(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);

// [1, 1, 32, 32 * pages] FLOAT32 TILE: page p is tile column p.
Tensor zero_stats(MeshDevice& device, uint32_t pages) {
    return dram(
        device,
        TensorSpec(
            ttnn::Shape({1, 1, 32, 32 * pages}), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), kDram)),
        std::vector<float>(32 * 32 * pages, 0.0f));
}

struct Run {
    std::optional<Conv2dSliceConfig> slice;
    bool bias = false;
    std::optional<Tensor> stats;
    uint32_t width = W;
};

std::vector<float> conv(MeshDevice& device, const Tensor& input, const Run& run) {
    const auto spec = [](const ttnn::Shape& shape) {
        return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));
    };
    const Tensor weights = Tensor::from_vector(random_values(CO * C * 9, 7, 0.25f), spec(ttnn::Shape({CO, C, 3, 3})));
    const std::optional<const Tensor> bias = run.bias
                                                 ? std::optional<const Tensor>(Tensor::from_vector(
                                                       random_values(CO, 8, 1.0f), spec(ttnn::Shape({1, 1, 1, CO}))))
                                                 : std::nullopt;
    Conv2dConfig config;
    config.shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    const Tensor out = std::get<static_cast<int>(ConvResultType::OUTPUT)>(ttnn::conv2d_fused(
        input,
        weights,
        &device,
        C,
        CO,
        1,
        H,
        run.width,
        {3, 3},
        {1, 1},
        std::array<uint32_t, 4>{1, 1, 1, 1},
        {1, 1},
        1,
        std::nullopt,
        bias,
        config,
        std::nullopt,
        std::nullopt,
        run.slice,
        false,
        false,
        {.epilogue_stats = run.stats}));
    return out.to_vector<float>();
}

// Channel sums (first CO) and sums of squares (last CO) from the stats pages: slice s from page s * stride, core k of
// a slice at 2 * k * NTILES, the sum tiles then the square tiles, row 0 of each.
std::vector<double> device_sums(const Tensor& stats, uint32_t stride) {
    const auto v = stats.to_vector<float>();
    const uint32_t pages = v.size() / (32 * 32);
    std::vector<double> sums(2 * CO, 0.0);
    for (uint32_t p = 0; p < pages; ++p) {
        const uint32_t q = (p % stride) % (2 * NTILES);
        for (uint32_t i = 0; i < 32; ++i) {
            sums[(q / NTILES) * CO + (q % NTILES) * 32 + i] += v[p * 32 + i];
        }
    }
    return sums;
}

// The host sums, and the sums of the magnitudes they add up (the scale of the device's bf16 rounding).
std::pair<std::vector<double>, std::vector<double>> host_sums(const std::vector<float>& out) {
    std::vector<double> sums(2 * CO, 0.0), scale(2 * CO, 0.0);
    for (size_t i = 0; i < out.size(); ++i) {
        const double v = out[i];
        sums[i % CO] += v;
        scale[i % CO] += std::abs(v);
        sums[CO + i % CO] += v * v;
        scale[CO + i % CO] += v * v;
    }
    return {sums, scale};
}

const std::array<Conv2dSliceConfig, 3> kPaths{
    Conv2dSliceConfig{.slice_type = SliceType::L1_FULL},
    Conv2dSliceConfig{.slice_type = SliceType::DRAM_WIDTH, .num_slices = 2},
    Conv2dSliceConfig{.slice_type = SliceType::DRAM_HEIGHT, .num_slices = 2}};

}  // namespace

TEST_F(Conv2dEpilogueStatsTest, StatsAreTheChannelSumsOfTheUnchangedOutput) {
    auto& device = *device_;
    const Tensor x = dram(
        device,
        TensorSpec(ttnn::Shape({1, H, W, C}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), kDram)),
        random_values(H * W * C, 1, 2.0f));
    for (const auto& slice : kPaths) {
        for (bool bias : {false, true}) {
            const auto plain = conv(device, x, {.slice = slice, .bias = bias});
            const Tensor stats = zero_stats(device, ttnn::conv2d_epilogue_stats_num_pages(&device, CO, 2));
            const auto with_stats = conv(device, x, {.slice = slice, .bias = bias, .stats = stats});
            const std::string where = "slice type " + std::to_string(static_cast<int>(slice.slice_type)) +
                                      (bias ? " with bias" : " without bias");
            EXPECT_TRUE(with_stats == plain) << where << ": the output changed";
            const auto [expected, scale] = host_sums(plain);
            const auto actual = device_sums(stats, ttnn::conv2d_epilogue_stats_num_pages(&device, CO));
            for (uint32_t i = 0; i < 2 * CO; ++i) {
                EXPECT_NEAR(actual[i], expected[i], 0.05 + 3e-3 * scale[i]) << where << " stat " << i;
            }
        }
    }
}

// Each slice's page base is a runtime argument: a 4-slice conv (32 columns each) with stats compiles as many programs
// as without, so the two middle slices still share one.
TEST_F(Conv2dEpilogueStatsTest, StatsAddNoPrograms) {
    auto& device = *device_;
    constexpr uint32_t width = 4 * 32;
    const Tensor x = dram(
        device,
        TensorSpec(ttnn::Shape({1, H, width, C}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), kDram)),
        random_values(H * width * C, 2, 2.0f));
    const Conv2dSliceConfig slices{.slice_type = SliceType::DRAM_WIDTH, .num_slices = 4};
    const auto programs = [&](const std::optional<Tensor>& stats) {
        device.clear_program_cache();
        conv(device, x, {.slice = slices, .stats = stats, .width = width});
        return device.num_program_cache_entries();
    };
    const size_t plain = programs(std::nullopt);
    EXPECT_EQ(programs(zero_stats(device, ttnn::conv2d_epilogue_stats_num_pages(&device, CO, 4))), plain);
}

// The stats pages of every slice must fit the tensor: one slice's pages for two slices, one page unsliced.
TEST_F(Conv2dEpilogueStatsTest, StatsPagesOutsideTheTensorAreRejected) {
    auto& device = *device_;
    const Tensor x = dram(
        device,
        TensorSpec(ttnn::Shape({1, H, W, C}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), kDram)),
        random_values(H * W * C, 3, 2.0f));
    EXPECT_ANY_THROW(conv(
        device,
        x,
        {.slice = kPaths[1], .stats = zero_stats(device, ttnn::conv2d_epilogue_stats_num_pages(&device, CO))}));
    EXPECT_ANY_THROW(conv(device, x, {.slice = kPaths[0], .stats = zero_stats(device, 1)}));
}

}  // namespace ttnn::operations::conv::conv2d::epilogue_stats_test
