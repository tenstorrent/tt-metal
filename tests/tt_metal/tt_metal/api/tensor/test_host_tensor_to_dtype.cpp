// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Deterministic ramp for test data
template <typename T>
std::vector<T> make_ramp(size_t count) {
    std::vector<T> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(static_cast<float>(i % 251));
    }
    return data;
}

std::vector<float> make_bfp_data(const Shape& shape, const Tile& tile) {
    // Boundary-sensitive to faces: cycles face indices and steps magnitudes for BFP checks.
    const size_t count = shape.volume();
    const size_t face_width = tile.get_face_shape()[1];
    std::vector<float> data(count);
    size_t i = 0;
    for (size_t face_row = 0; i < count; ++face_row) {
        for (size_t face_index = 0; face_index < face_width && i < count; ++face_index) {
            data[i++] = static_cast<float>(face_index) + static_cast<float>(face_row) * 0.5f;
        }
    }
    return data;
}

using PackedBfp = std::vector<uint32_t>;
using UnpackedBfp = std::vector<float>;

// BFP -> float golden: packed bytes plus their reference unpack (not the pre-quant floats).
std::pair<PackedBfp, UnpackedBfp> generate_bfp8_dataset(const Shape& shape, const Tile& tile) {
    const auto src = make_bfp_data(shape, tile);
    auto packed = pack_as_bfp8_tiles(ttsl::make_const_span(src), /*row_major_input=*/true, /*is_exp_a=*/false, tile);
    auto unpacked = unpack_bfp8_tiles_into_float_vec(packed, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    return {std::move(packed), std::move(unpacked)};
}

std::pair<PackedBfp, UnpackedBfp> generate_bfp4_dataset(const Shape& shape, const Tile& tile) {
    const auto src = make_bfp_data(shape, tile);
    auto packed = pack_as_bfp4_tiles(ttsl::make_const_span(src), /*row_major_input=*/true, /*is_exp_a=*/false, tile);
    auto unpacked = unpack_bfp4_tiles_into_float_vec(packed, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    return {std::move(packed), std::move(unpacked)};
}

// Float(TILE) -> BFP golden: tile-layout floats plus their reference pack.
std::pair<UnpackedBfp, PackedBfp> generate_float_to_bfp8_dataset(const Shape& shape, const Tile& tile) {
    auto floats = make_bfp_data(shape, tile);
    auto packed =
        pack_as_bfp8_tiles(ttsl::make_const_span(floats), /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    return {std::move(floats), std::move(packed)};
}

std::pair<UnpackedBfp, PackedBfp> generate_float_to_bfp4_dataset(const Shape& shape, const Tile& tile) {
    auto floats = make_bfp_data(shape, tile);
    auto packed =
        pack_as_bfp4_tiles(ttsl::make_const_span(floats), /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    return {std::move(floats), std::move(packed)};
}

template <typename T>
HostTensor make_host_tensor(std::vector<T> data, const TensorSpec& spec) {
    return HostTensor::from_buffer(HostBuffer(std::move(data)), spec);
}

// Repeats a short pattern until it fills a tensor, so a boundary table can be checked through the
// same shapes the rest of this file uses.
template <typename T>
std::vector<T> repeat_to_volume(const std::vector<T>& pattern, size_t volume) {
    std::vector<T> data(volume);
    for (size_t i = 0; i < volume; ++i) {
        data[i] = pattern[i % pattern.size()];
    }
    return data;
}

// Boundary floats for the conversions to integral dtypes. A plain static_cast of any of the
// out-of-range rows is undefined behavior and the host architectures disagree on it: 2^31 to INT32
// is INT32_MIN on x86_64 and INT32_MAX on aarch64, and -1.0f to UINT32 is 0xFFFFFFFF on x86_64 and
// 0 on aarch64. Every conversion below has to saturate, on both.
constexpr float k_largest_below_int32_max = 2147483520.0f;   // largest float below 2^31
constexpr float k_two_pow_31 = 2147483648.0f;                // first float past INT32_MAX
constexpr float k_largest_below_uint32_max = 4294967040.0f;  // largest float below 2^32
constexpr float k_two_pow_32 = 4294967296.0f;                // first float past UINT32_MAX

std::vector<float> boundary_floats() {
    return {
        0.0f,
        -1.0f,
        k_largest_below_int32_max,
        k_two_pow_31,
        k_largest_below_uint32_max,
        k_two_pow_32,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
}

// The same table in bfloat16, whose largest finite value is far past every integral destination.
std::vector<bfloat16> boundary_bfloat16s() {
    return {
        bfloat16(0.0f),
        bfloat16(-1.0f),
        bfloat16::truncate(std::numeric_limits<float>::max()),  // largest finite bfloat16, ~3.39e38
        bfloat16(k_two_pow_31),
        bfloat16(std::numeric_limits<float>::infinity()),
        bfloat16(-std::numeric_limits<float>::infinity()),
        bfloat16(std::numeric_limits<float>::quiet_NaN()),
    };
}

TensorSpec row_major_spec(const Shape& shape, DataType dtype) {
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), memory_config));
}

bool exact_spec_match(const TensorSpec& a, const TensorSpec& b) {
    return a == b && experimental::per_core_allocation::is_per_core_allocation(a.memory_config()) ==
                         experimental::per_core_allocation::is_per_core_allocation(b.memory_config());
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

using ::testing::Eq;
using ::testing::Pointwise;

TEST(HostTensorToDtype, NonBfpPreservesMetadata) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto alignment = tt::tt_metal::Alignment({32, 32});
    auto tile = Tile({16, 16});

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT16);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);
}

TEST(HostTensorToDtype, TileBfp8ToFloat32ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [packed, unpacked_golden] = CMAKE_UNIQUE_NAMESPACE::generate_bfp8_dataset(shape, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(packed, source_spec);

    auto result = to_dtype(source, DataType::FLOAT32);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    EXPECT_THAT(host_buffer::get_as<float>(result), Pointwise(Eq(), unpacked_golden));
}

TEST(HostTensorToDtype, Float32ToTileBfp8ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [floats, packed_golden] = CMAKE_UNIQUE_NAMESPACE::generate_float_to_bfp8_dataset(shape, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(floats, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT8_B);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT8_B);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    EXPECT_THAT(host_buffer::get_as<uint32_t>(result), Pointwise(Eq(), packed_golden));
}

TEST(HostTensorToDtype, TileBfp4ToFloat32ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [packed, unpacked_golden] = CMAKE_UNIQUE_NAMESPACE::generate_bfp4_dataset(shape, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT4_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(packed, source_spec);

    auto result = to_dtype(source, DataType::FLOAT32);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    EXPECT_THAT(host_buffer::get_as<float>(result), Pointwise(Eq(), unpacked_golden));
}

TEST(HostTensorToDtype, Float32ToTileBfp4ValueCheck) {
    const Shape shape{32, 32};
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto alignment = Alignment({32, 32});
    const auto tile = Tile({16, 16});
    const auto& [floats, packed_golden] = CMAKE_UNIQUE_NAMESPACE::generate_float_to_bfp4_dataset(shape, tile);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = CMAKE_UNIQUE_NAMESPACE::make_host_tensor(floats, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT4_B);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT4_B, PageConfig(Layout::TILE, tile), memory_config, alignment));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT4_B);
    EXPECT_EQ(result.layout(), Layout::TILE);
    EXPECT_EQ(result.tensor_spec().tile(), tile);

    EXPECT_THAT(host_buffer::get_as<uint32_t>(result), Pointwise(Eq(), packed_golden));
}

TEST(HostTensorToDtype, Float32ToBfloat16RowMajorValueCheck) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 1})}), {16, 64}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory_config, true);

    auto source_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT16);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), memory_config));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);

    auto result_data = result.to_vector<bfloat16>();
    EXPECT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(static_cast<float>(result_data[i]), static_cast<float>(bfloat16(data[i])));
    }
}

TEST(HostTensorToDtype, Bfloat16ToFloat32RowMajorValueCheck) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<bfloat16>(shape.volume());
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 1})}), {16, 64}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory_config, true);

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<bfloat16>(data, source_spec);

    auto result = to_dtype(source, DataType::FLOAT32);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::FLOAT32);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);

    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(result_data[i], static_cast<float>(data[i]));
    }
}

TEST(HostTensorToDtype, PerCoreAllocationPreserved) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());

    // Sharded MemoryConfig
    auto memory_config = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{CoreRangeSet({CoreRange({0, 0}, {0, 0})}), {32, 64}, ShardOrientation::ROW_MAJOR}};
    experimental::per_core_allocation::set_per_core_allocation(memory_config, true);

    auto alignment = tt::tt_metal::Alignment({32, 64});
    auto tile = Tile({32, 32});

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE, tile), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    auto result = to_dtype(source, DataType::BFLOAT16);

    auto expected_spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, tile), memory_config, alignment));

    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_TRUE(experimental::per_core_allocation::is_per_core_allocation(result.tensor_spec().memory_config()));
}

TEST(HostTensorToDtype, OversizedBufferToDtype) {
    Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));

    // Create an oversized buffer
    auto oversized_data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume() + 100);

    // Create tensor from buffer (from_buffer accepts oversized buffers if they are large enough)
    auto host_tensor = HostTensor::from_buffer(HostBuffer(std::vector<float>(oversized_data)), spec);

    // Currently to_dtype asserts on exact packed size, so it will fail if it's oversized.
    EXPECT_ANY_THROW(to_dtype(host_tensor, DataType::BFLOAT16));
}

TEST(HostTensorToDtype, UndersizedBufferToDtypeThrows) {
    Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    // Use BFLOAT8_B so we test the pre-unpack size check
    auto spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, Tile({16, 16})), memory_config));

    // Create an undersized buffer
    auto undersized_data = std::vector<uint32_t>(10, 0);  // Much smaller than required

    // from_buffer doesn't check size, so this succeeds
    auto host_tensor = HostTensor::from_buffer(HostBuffer(std::vector<uint32_t>(undersized_data)), spec);

    EXPECT_ANY_THROW(to_dtype(host_tensor, DataType::FLOAT32));
}

TEST(HostTensorToDtype, MalformedBfpBufferToDtypeThrows) {
    Shape shape{32, 32};
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto spec =
        TensorSpec(shape, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE, Tile({16, 16})), memory_config));

    // Create a buffer that is slightly off in size (e.g. missing one word)
    size_t expected_size_bytes = spec.compute_packed_buffer_size_bytes();
    auto malformed_data = std::vector<uint32_t>((expected_size_bytes / sizeof(uint32_t)) - 1, 0);

    auto host_tensor = HostTensor::from_buffer(HostBuffer(std::vector<uint32_t>(malformed_data)), spec);

    EXPECT_ANY_THROW(to_dtype(host_tensor, DataType::FLOAT32));
}
TEST(HostTensorToDtype, RowMajorToBfpPhysicalMismatchThrows) {
    const Shape shape{32, 24};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    // Alignment that is NOT a multiple of the default tile width
    auto alignment = tt::tt_metal::Alignment({32, 24});

    auto source_spec =
        TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config, alignment));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    // This should throw because BFLOAT8_B forces TILE layout, which forces alignment to be
    // a multiple of the tile size. So output physical shape width will be 32 instead of 24.
    EXPECT_ANY_THROW(to_dtype(source, DataType::BFLOAT8_B));
}

TEST(HostTensorToDtype, Float32ToInt8RowMajorValueCheck) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<float>(shape.volume());

    for (float& i : data) {
        i = static_cast<float>(static_cast<int8_t>(static_cast<int>(i)));
    }
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    auto source_spec = TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<float>(data, source_spec);

    auto result = to_dtype(source, DataType::INT8);

    auto expected_spec = TensorSpec(shape, TensorLayout(DataType::INT8, PageConfig(Layout::ROW_MAJOR), memory_config));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::INT8);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);

    auto result_data = result.to_vector<int8_t>();
    EXPECT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(result_data[i], static_cast<int8_t>(data[i]));
    }
}

TEST(HostTensorToDtype, Int8ToInt32RowMajorValueCheck) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<int8_t>(shape.volume());
    data[0] = -128;
    data[1] = 127;
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    auto source_spec = TensorSpec(shape, TensorLayout(DataType::INT8, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<int8_t>(data, source_spec);

    auto result = to_dtype(source, DataType::INT32);

    auto expected_spec = TensorSpec(shape, TensorLayout(DataType::INT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::INT32);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);

    auto result_data = result.to_vector<int32_t>();
    EXPECT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(result_data[i], static_cast<int32_t>(data[i]));
    }
}

TEST(HostTensorToDtype, Int32ToInt8RowMajorValueCheck) {
    const Shape shape{32, 64};
    auto data = CMAKE_UNIQUE_NAMESPACE::make_ramp<int32_t>(shape.volume());

    for (int32_t& i : data) {
        // Intentional signed wrap into the INT8 range so the INT32->INT8 conversion stays in-range
        // and covers the -128/127 edges; the signed-char narrowing here is deliberate.
        // NOLINTNEXTLINE(bugprone-signed-char-misuse,cert-str34-c)
        i = static_cast<int32_t>(static_cast<int8_t>(i));
    }
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    auto source_spec = TensorSpec(shape, TensorLayout(DataType::INT32, PageConfig(Layout::ROW_MAJOR), memory_config));
    auto source = HostTensor::from_vector<int32_t>(data, source_spec);

    auto result = to_dtype(source, DataType::INT8);

    auto expected_spec = TensorSpec(shape, TensorLayout(DataType::INT8, PageConfig(Layout::ROW_MAJOR), memory_config));
    EXPECT_TRUE(CMAKE_UNIQUE_NAMESPACE::exact_spec_match(result.tensor_spec(), expected_spec));
    EXPECT_EQ(result.dtype(), DataType::INT8);
    EXPECT_EQ(result.layout(), Layout::ROW_MAJOR);

    auto result_data = result.to_vector<int8_t>();
    EXPECT_EQ(result_data.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(result_data[i], static_cast<int8_t>(data[i]));
    }
}

TEST(HostTensorToDtype, Float32ToInt32SaturatesOutOfRange) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(CMAKE_UNIQUE_NAMESPACE::boundary_floats(), shape.volume());
    auto source =
        HostTensor::from_vector<float>(data, CMAKE_UNIQUE_NAMESPACE::row_major_spec(shape, DataType::FLOAT32));

    auto result = to_dtype(source, DataType::INT32);

    constexpr int32_t int32_max = std::numeric_limits<int32_t>::max();
    const std::vector<int32_t> expected_pattern = {
        0,                                    // 0.0f
        -1,                                   // -1.0f
        2147483520,                           // largest float below 2^31, exactly representable
        int32_max,                            // 2^31
        int32_max,                            // largest float below 2^32
        int32_max,                            // 2^32
        int32_max,                            // +inf
        std::numeric_limits<int32_t>::min(),  // -inf
        0,                                    // NaN
    };
    EXPECT_THAT(
        result.to_vector<int32_t>(),
        Pointwise(Eq(), CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(expected_pattern, shape.volume())));
}

TEST(HostTensorToDtype, Float32ToUint32SaturatesOutOfRange) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(CMAKE_UNIQUE_NAMESPACE::boundary_floats(), shape.volume());
    auto source =
        HostTensor::from_vector<float>(data, CMAKE_UNIQUE_NAMESPACE::row_major_spec(shape, DataType::FLOAT32));

    auto result = to_dtype(source, DataType::UINT32);

    constexpr uint32_t uint32_max = std::numeric_limits<uint32_t>::max();
    const std::vector<uint32_t> expected_pattern = {
        0u,           // 0.0f
        0u,           // -1.0f: the standard invalid-index sentinel is not expressible here
        2147483520u,  // largest float below 2^31
        2147483648u,  // 2^31, in range for UINT32
        4294967040u,  // largest float below 2^32, exactly representable
        uint32_max,   // 2^32
        uint32_max,   // +inf
        0u,           // -inf
        0u,           // NaN
    };
    EXPECT_THAT(
        result.to_vector<uint32_t>(),
        Pointwise(Eq(), CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(expected_pattern, shape.volume())));
}

TEST(HostTensorToDtype, Float32ToUint16SaturatesOutOfRange) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(CMAKE_UNIQUE_NAMESPACE::boundary_floats(), shape.volume());
    auto source =
        HostTensor::from_vector<float>(data, CMAKE_UNIQUE_NAMESPACE::row_major_spec(shape, DataType::FLOAT32));

    auto result = to_dtype(source, DataType::UINT16);

    constexpr uint16_t uint16_max = std::numeric_limits<uint16_t>::max();
    const std::vector<uint16_t> expected_pattern = {
        0,           // 0.0f
        0,           // -1.0f
        uint16_max,  // largest float below 2^31
        uint16_max,  // 2^31
        uint16_max,  // largest float below 2^32
        uint16_max,  // 2^32
        uint16_max,  // +inf
        0,           // -inf
        0,           // NaN
    };
    EXPECT_THAT(
        result.to_vector<uint16_t>(),
        Pointwise(Eq(), CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(expected_pattern, shape.volume())));
}

TEST(HostTensorToDtype, Float32ToUint8SaturatesOutOfRange) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(CMAKE_UNIQUE_NAMESPACE::boundary_floats(), shape.volume());
    auto source =
        HostTensor::from_vector<float>(data, CMAKE_UNIQUE_NAMESPACE::row_major_spec(shape, DataType::FLOAT32));

    auto result = to_dtype(source, DataType::UINT8);

    constexpr uint8_t uint8_max = std::numeric_limits<uint8_t>::max();
    const std::vector<uint8_t> expected_pattern = {0, 0, uint8_max, uint8_max, uint8_max, uint8_max, uint8_max, 0, 0};
    EXPECT_THAT(
        result.to_vector<uint8_t>(),
        Pointwise(Eq(), CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(expected_pattern, shape.volume())));
}

TEST(HostTensorToDtype, Float32ToInt8SaturatesOutOfRange) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(CMAKE_UNIQUE_NAMESPACE::boundary_floats(), shape.volume());
    auto source =
        HostTensor::from_vector<float>(data, CMAKE_UNIQUE_NAMESPACE::row_major_spec(shape, DataType::FLOAT32));

    auto result = to_dtype(source, DataType::INT8);

    constexpr int8_t int8_max = std::numeric_limits<int8_t>::max();
    constexpr int8_t int8_min = std::numeric_limits<int8_t>::min();
    const std::vector<int8_t> expected_pattern = {0, -1, int8_max, int8_max, int8_max, int8_max, int8_max, int8_min, 0};
    EXPECT_THAT(
        result.to_vector<int8_t>(),
        Pointwise(Eq(), CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(expected_pattern, shape.volume())));
}

TEST(HostTensorToDtype, Bfloat16ToInt32SaturatesOutOfRange) {
    const Shape shape{32, 32};
    auto data = CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(CMAKE_UNIQUE_NAMESPACE::boundary_bfloat16s(), shape.volume());
    auto source =
        HostTensor::from_vector<bfloat16>(data, CMAKE_UNIQUE_NAMESPACE::row_major_spec(shape, DataType::BFLOAT16));

    auto result = to_dtype(source, DataType::INT32);

    constexpr int32_t int32_max = std::numeric_limits<int32_t>::max();
    const std::vector<int32_t> expected_pattern = {
        0,                                    // 0.0
        -1,                                   // -1.0
        int32_max,                            // largest finite bfloat16
        int32_max,                            // 2^31
        int32_max,                            // +inf
        std::numeric_limits<int32_t>::min(),  // -inf
        0,                                    // NaN
    };
    EXPECT_THAT(
        result.to_vector<int32_t>(),
        Pointwise(Eq(), CMAKE_UNIQUE_NAMESPACE::repeat_to_volume(expected_pattern, shape.volume())));
}

}  // namespace
}  // namespace tt::tt_metal
