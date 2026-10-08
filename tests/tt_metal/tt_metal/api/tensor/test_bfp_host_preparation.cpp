// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <future>
#include <limits>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/host_bfp_conversion.hpp>

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T>
void check_factory(const Shape& shape, DataType dtype, const Tile& tile = Tile{}, MemoryConfig memory = {}) {
    std::vector<T> data(shape.volume());
    std::mt19937 rng(59764);
    for (size_t i = 0; i < data.size(); ++i) {
        if constexpr (std::is_same_v<T, bfloat16>) {
            // Include every BF16 encoding, including signed zeros and NaN payloads.
            data[i] = std::bit_cast<bfloat16>(static_cast<uint16_t>(i));
        } else {
            data[i] = std::bit_cast<float>(static_cast<uint32_t>(rng()));
        }
    }
    const auto page = PageConfig(Layout::TILE, tile);
    const auto spec = TensorSpec(shape, TensorLayout(dtype, page, memory));
    const auto source_dtype = std::is_same_v<T, float> ? DataType::FLOAT32 : DataType::BFLOAT16;
    const auto source_spec = TensorSpec(shape, TensorLayout(source_dtype, page, memory));
    // Independent staged path: first materialize source-type tiles, then pack.
    const auto tiled = HostTensor::from_span<T>(data, source_spec);
    const auto expected = to_dtype(tiled, dtype);
    const auto expected_words = host_buffer::get_as<uint32_t>(expected);
    auto check = [&](const HostTensor& actual) {
        EXPECT_EQ(actual.tensor_spec(), spec);
        EXPECT_EQ(
            experimental::per_core_allocation::is_per_core_allocation(actual.tensor_spec().memory_config()),
            experimental::per_core_allocation::is_per_core_allocation(memory));
        const auto words = host_buffer::get_as<uint32_t>(actual);
        EXPECT_EQ(words.size(), expected_words.size());
        EXPECT_TRUE(std::equal(words.begin(), words.end(), expected_words.begin(), expected_words.end()));
    };
    check(HostTensor::from_span<T>(data, spec));
    check(HostTensor::from_vector<T>(data, spec));
    check(HostTensor::from_vector<T>(std::move(data), spec));
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

class BfpHostPreparation : public ::testing::TestWithParam<DataType> {};

TEST_P(BfpHostPreparation, Float32MatrixAndBatch) {
    for (const auto& shape : {Shape{32, 32}, Shape{64, 96}, Shape{2, 3, 64, 128}, Shape{1024, 512}}) {
        CMAKE_UNIQUE_NAMESPACE::check_factory<float>(shape, GetParam());
    }
}

TEST_P(BfpHostPreparation, AllBfloat16EncodingsAndParallelPath) {
    CMAKE_UNIQUE_NAMESPACE::check_factory<bfloat16>(Shape{1024, 512}, GetParam());
}

TEST_P(BfpHostPreparation, PaddingAndTileFallbacks) {
    for (const auto& shape : {Shape{35, 67}, Shape{2, 3, 19, 65}, Shape{64, 128}}) {
        for (const auto& tile : {Tile({32, 32}), Tile({16, 32}), Tile({32, 32}, true)}) {
            CMAKE_UNIQUE_NAMESPACE::check_factory<float>(shape, GetParam(), tile);
            CMAKE_UNIQUE_NAMESPACE::check_factory<bfloat16>(shape, GetParam(), tile);
        }
    }
}

TEST_P(BfpHostPreparation, PreservePerCoreAllocation) {
    auto memory = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 0})}), {64, 128}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory, true);
    CMAKE_UNIQUE_NAMESPACE::check_factory<bfloat16>(Shape{64, 128}, GetParam(), Tile{}, memory);
}

TEST_P(BfpHostPreparation, ConcurrentCalls) {
    std::vector<std::future<void>> pending;
    for (size_t i = 0; i < 4; ++i) {
        pending.emplace_back(std::async(std::launch::async, [dtype = GetParam()] {
            CMAKE_UNIQUE_NAMESPACE::check_factory<float>(Shape{1024, 512}, dtype);
        }));
    }
    for (auto& result : pending) {
        result.get();
    }
}

TEST_P(BfpHostPreparation, StridedRowsAndSourceLifetime) {
    const Shape shape{512, 256};
    const size_t stride = 389;
    const auto spec = TensorSpec(shape, TensorLayout(GetParam(), PageConfig(Layout::TILE), MemoryConfig{}));
    std::vector<float> source(511 * stride + 256, -999.0f);
    std::vector<float> contiguous(shape.volume());
    std::mt19937 rng(59764);
    std::uniform_real_distribution<float> values(-1.0f, 1.0f);
    for (size_t row = 0; row < 512; ++row) {
        for (size_t col = 0; col < 256; ++col) {
            source[row * stride + col] = contiguous[row * 256 + col] = values(rng);
        }
    }
    const auto expected = HostTensor::from_span<float>(contiguous, spec);
    const auto actual = experimental::try_create_bfp_host_tensor<float>(source, spec, stride);
    ASSERT_TRUE(actual.has_value());
    std::fill(source.begin(), source.end(), 99.0f);
    EXPECT_EQ(actual->tensor_spec(), spec);
    const auto words = host_buffer::get_as<uint32_t>(*actual);
    const auto expected_words = host_buffer::get_as<uint32_t>(expected);
    EXPECT_TRUE(std::equal(words.begin(), words.end(), expected_words.begin(), expected_words.end()));
}

TEST_P(BfpHostPreparation, StridedFactoryRejectsUnsupportedViews) {
    const Shape shape{64, 64};
    const auto spec = TensorSpec(shape, TensorLayout(GetParam(), PageConfig(Layout::TILE), MemoryConfig{}));
    std::vector<float> source(shape.volume(), 1.0f);
    EXPECT_FALSE(experimental::try_create_bfp_host_tensor<float>(source, spec, 63).has_value());
    EXPECT_FALSE(experimental::try_create_bfp_host_tensor<float>(source, spec, 128).has_value());
    EXPECT_FALSE(
        experimental::try_create_bfp_host_tensor<float>(source, spec, std::numeric_limits<size_t>::max()).has_value());
    const auto padded = TensorSpec(Shape{63, 63}, spec.tensor_layout());
    EXPECT_FALSE(experimental::try_create_bfp_host_tensor<float>(source, padded, 64).has_value());
    const auto ordinary = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), MemoryConfig{}));
    EXPECT_FALSE(experimental::try_create_bfp_host_tensor<float>(source, ordinary, 64).has_value());
}

INSTANTIATE_TEST_SUITE_P(Formats, BfpHostPreparation, ::testing::Values(DataType::BFLOAT4_B, DataType::BFLOAT8_B));

}  // namespace
}  // namespace tt::tt_metal
