// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// padded_slice's optional prologue_scale / prologue_shift: silu(x * scale + shift) in the untilize pass, against a host
// fp32 golden of the default output on the same slice.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/experimental/padded_slice/padded_slice.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp"

namespace ttnn::operations::experimental::test {

using tt::tt_metal::distributed::MeshDevice;

class PaddedSlicePrologueTest : public TTNNFixtureWithSuiteDevice<PaddedSlicePrologueTest> {};

namespace {

constexpr uint32_t H = 8, W = 32, NUM_CORES = 4;

std::vector<bfloat16> random_values(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<bfloat16> out(n);
    for (auto& v : out) {
        v = bfloat16(dist(rng));
    }
    return out;
}

Tensor dram(
    MeshDevice& device,
    const std::vector<bfloat16>& data,
    const ttnn::Shape& shape,
    DataType dtype = DataType::BFLOAT16) {
    const MemoryConfig mem(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem));
    const Tensor host = dtype == DataType::BFLOAT16
                            ? Tensor::from_vector(data, spec)
                            : Tensor::from_vector(std::vector<float>(data.begin(), data.end()), spec);
    return host.to_device(&device, mem);
}

MemoryConfig height_sharded(uint32_t sticks, uint32_t c) {
    return MemoryConfig(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(CoreRangeSet(CoreRange({0, 0}, {NUM_CORES - 1, 0})), {sticks / NUM_CORES, c}));
}

struct Slice {
    std::array<uint32_t, 4> begins, ends;
    uint32_t sticks() const { return (ends[1] - begins[1]) * (ends[2] - begins[2]); }
};

Tensor slice(
    const Tensor& input,
    const Slice& s,
    const MemoryConfig& mem,
    const std::optional<Tensor>& scale = std::nullopt,
    const std::optional<Tensor>& shift = std::nullopt) {
    return ttnn::experimental::padded_slice(
        input,
        ttsl::SmallVector<uint32_t>(s.begins.begin(), s.begins.end()),
        ttsl::SmallVector<uint32_t>(s.ends.begin(), s.ends.end()),
        ttsl::SmallVector<uint32_t>{1, 1, 1, 1},
        mem,
        std::nullopt,
        std::nullopt,
        scale,
        shift);
}

std::vector<float> values_of(const Tensor& t) {
    const auto v = t.to_vector<bfloat16>();
    return {v.begin(), v.end()};
}

size_t mismatches(
    const std::vector<float>& plain,
    const std::vector<float>& actual,
    const std::vector<bfloat16>& scale,
    const std::vector<bfloat16>& shift) {
    const size_t c = scale.size();
    size_t failures = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        const float z = plain[i] * static_cast<float>(scale[i % c]) + static_cast<float>(shift[i % c]);
        const float golden = z / (1.0f + std::exp(-z));
        failures += std::abs(actual[i] - golden) > 0.02f + 0.02f * std::abs(golden);
    }
    return failures;
}

}  // namespace

// A height slice and a width slice, the two ways conv2d's DRAM slicing cuts its input; 512 channels = 16 tiles per
// row, wider than one pack_untilize block. The second scale / shift pair runs from the program cache.
TEST_F(PaddedSlicePrologueTest, PrologueAppliesAffineSiluToTheSlice) {
    auto& device = *device_;
    for (uint32_t c : {64u, 512u}) {
        const Tensor input = dram(device, random_values(H * W * c, c), ttnn::Shape({1, H, W, c}));
        for (const Slice& s : {Slice{{0, 2, 0, 0}, {1, 6, W, c}}, Slice{{0, 0, 8, 0}, {1, H, 24, c}}}) {
            const MemoryConfig mem = height_sharded(s.sticks(), c);
            const auto plain = values_of(slice(input, s, mem));
            for (uint32_t seed : {1u, 2u}) {
                const auto scale = random_values(c, 10 * seed + c);
                const auto shift = random_values(c, 10 * seed + c + 1);
                const auto actual = values_of(slice(
                    input,
                    s,
                    mem,
                    dram(device, scale, ttnn::Shape({1, 1, 1, c})),
                    dram(device, shift, ttnn::Shape({1, 1, 1, c}))));
                ASSERT_EQ(actual.size(), plain.size());
                EXPECT_EQ(mismatches(plain, actual, scale, shift), 0u)
                    << "c " << c << " rows [" << s.begins[1] << ", " << s.ends[1] << ") cols [" << s.begins[2] << ", "
                    << s.ends[2] << ") seed " << seed;
            }
        }
    }
}

// The scale / shift buffers use the input format and every core reads the same [1, C] tiles, so the prologue needs a
// bf16 input, both tensors, and an output shard holding all channels.
TEST_F(PaddedSlicePrologueTest, PrologueRejectsUnsupportedInputs) {
    auto& device = *device_;
    constexpr uint32_t c = 64;
    const auto values = random_values(H * W * c, 3);
    const auto affine = random_values(c, 4);
    const Slice s{{0, 2, 0, 0}, {1, 6, W, c}};
    const MemoryConfig mem = height_sharded(s.sticks(), c);
    const Tensor input = dram(device, values, ttnn::Shape({1, H, W, c}));
    const Tensor scale = dram(device, affine, ttnn::Shape({1, 1, 1, c}));

    const Tensor bf8_input = dram(device, values, ttnn::Shape({1, H, W, c}), DataType::BFLOAT8_B);
    const Tensor bf8_scale = dram(device, affine, ttnn::Shape({1, 1, 1, c}), DataType::BFLOAT8_B);
    EXPECT_ANY_THROW(slice(bf8_input, s, mem, bf8_scale, bf8_scale));
    EXPECT_ANY_THROW(slice(input, s, mem, std::nullopt, scale));

    const Slice half_channels{{0, 2, 0, 0}, {1, 6, W, c / 2}};
    EXPECT_ANY_THROW(slice(input, half_channels, height_sharded(s.sticks(), c / 2), scale, scale));
}

}  // namespace ttnn::operations::experimental::test
