// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <cstdint>

#include "assert.hpp"
#include "buffer_constants.hpp"
#include "shape2d.hpp"
#include "tests/ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include "ttnn/tensor/xtensor/xtensor_all_includes.hpp"

namespace ttnn {
namespace {

using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;

template <typename... Args>
testing::Matcher<ttnn::Shape> ShapeIs(Args... args) {
    return testing::Eq(ttnn::Shape({args...}));
}

const std::vector<ttnn::Shape>& get_shapes_for_test() {
    static auto* shapes = new std::vector<ttnn::Shape>{
        ttnn::Shape{1},
        ttnn::Shape{1, 1, 1, 1},
        ttnn::Shape{1, 1, 1, 10},
        ttnn::Shape{1, 32, 32, 16},
        ttnn::Shape{1, 40, 3, 128},
        ttnn::Shape{2, 2},
        ttnn::Shape{1, 1, 1, 1, 10},
    };
    return *shapes;
}

TensorSpec get_tensor_spec_with_memory(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout = Layout::ROW_MAJOR,
    TensorMemoryLayout mem_layout = TensorMemoryLayout::SINGLE_BANK) {
    return TensorSpec(
        shape,
        TensorLayout(
            dtype,
            layout,
            MemoryConfig{
                .memory_layout = mem_layout,
                .buffer_type = BufferType::DRAM,
                .shard_spec = ShardSpec{
                    ttnn::CoreRangeSet{ttnn::CoreRange{ttnn::CoreRange{ttnn::CoreCoord{0, 0}, ttnn::CoreCoord{4, 1}}}},
                    {32, 64},
                    ShardOrientation::ROW_MAJOR,
                    ShardMode::LOGICAL}}));
}

TensorSpec get_tensor_spec(const ttnn::Shape& shape, DataType dtype, Layout layout = Layout::ROW_MAJOR) {
    return TensorSpec(shape, TensorLayout(dtype, layout, MemoryConfig{}));
}

template <typename T>
std::vector<T> arange(int64_t start, int64_t end, int64_t step, std::optional<int64_t> cap = std::nullopt) {
    std::vector<T> result;
    for (int el : xt::arange<int64_t>(start, end, step)) {
        int capped_el = cap ? el % *cap : el;
        if constexpr (std::is_same_v<T, ::bfloat16>) {
            result.push_back(T(static_cast<float>(capped_el)));
        } else {
            result.push_back(static_cast<T>(capped_el));
        }
    }
    return result;
}

template <typename T>
class VectorConversionTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, bfloat16, uint8_t, uint16_t, uint32_t, int32_t>;
TYPED_TEST_SUITE(VectorConversionTest, TestTypes);

TYPED_TEST(VectorConversionTest, TensorShape) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input = arange<TypeParam>(0, static_cast<int64_t>(shape.volume()), 1);
        Tensor output = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>()));

        EXPECT_THAT(output.get_tensor_spec().logical_shape(), Eq(shape)) << "for shape: " << shape;
    }
}

TYPED_TEST(VectorConversionTest, Roundtrip) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input = arange<TypeParam>(0, static_cast<int64_t>(shape.volume()), 1);
        auto tensor = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>()));

        TensorSpec tensor_spec = tensor.get_tensor_spec();

        EXPECT_THAT(tensor_spec.logical_shape(), Eq(shape)) << "for shape: " << shape;
        EXPECT_THAT(tensor_spec.data_type(), Eq(convert_to_data_type<TypeParam>()));

        auto output = tensor.template to_vector<TypeParam>();

        EXPECT_THAT(output, Pointwise(Eq(), input)) << "for shape: " << shape;
    }
}

TYPED_TEST(VectorConversionTest, InvalidSize) {
    ttnn::Shape shape{32, 32};
    auto input = arange<TypeParam>(0, 42, 1);

    ASSERT_NE(input.size(), shape.volume());
    EXPECT_ANY_THROW(Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>())));
}

TYPED_TEST(VectorConversionTest, OddshapeRoundtripTilizedLayout) {
    ttnn::Shape shape{1, 40, 3, 121};

    auto input = arange<TypeParam>(0, shape.volume(), 1);

    auto tensor = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>(), Layout::TILE));

    ASSERT_NE(tensor.tensor_spec().padded_shape(), tensor.tensor_spec().logical_shape());

    EXPECT_THAT(tensor.tensor_spec().logical_shape(), ShapeIs(1, 40, 3, 121));
    EXPECT_THAT(tensor.get_padded_shape(), ShapeIs(1, 40, 32, 128));

    auto output = tensor.template to_vector<TypeParam>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TYPED_TEST(VectorConversionTest, RoundtripTilezedLayout) {
    ttnn::Shape shape{128, 128};

    auto input = arange<TypeParam>(0, shape.volume(), 1);

    auto output = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>(), Layout::TILE))
                      .template to_vector<TypeParam>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TYPED_TEST(VectorConversionTest, InvalidDtype) {
    ttnn::Shape shape{32, 32};
    auto input = arange<TypeParam>(0, shape.volume(), 1);

    EXPECT_ANY_THROW(Tensor::from_vector(
        input,
        get_tensor_spec(
            shape,
            // Use INT32 for verification, except for when the actual type is int32_t.
            (std::is_same_v<TypeParam, int32_t> ? DataType::FLOAT32 : DataType::INT32))));
}

TEST(FloatVectorConversionTest, RoundtripBfloat16) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input_bf16 = arange<bfloat16>(0, static_cast<int64_t>(shape.volume()), 1);
        std::vector<float> input_ft;
        input_ft.reserve(input_bf16.size());
        std::transform(input_bf16.begin(), input_bf16.end(), std::back_inserter(input_ft), [](bfloat16 bf) {
            return bf.to_float();
        });

        auto output_bf16 =
            Tensor::from_vector(input_ft, get_tensor_spec(shape, DataType::BFLOAT16)).to_vector<bfloat16>();
        EXPECT_THAT(output_bf16, Pointwise(Eq(), input_bf16)) << "for shape: " << shape;

        auto output_ft = Tensor::from_vector(input_bf16, get_tensor_spec(shape, DataType::BFLOAT16)).to_vector<float>();
        EXPECT_THAT(output_ft, Pointwise(Eq(), input_ft)) << "for shape: " << shape;
    }
}

class BlockFloatVectorConversionTest : public ::testing::TestWithParam<DataType> {};

TEST_P(BlockFloatVectorConversionTest, InvalidLayout) {
    ttnn::Shape shape{32, 32};
    // Block float types are only supported in TILE layout.
    EXPECT_ANY_THROW(
        Tensor::from_vector(std::vector<float>(shape.volume()), get_tensor_spec(shape, GetParam(), Layout::ROW_MAJOR)));
}

TEST_P(BlockFloatVectorConversionTest, Roundtrip) {
    ttnn::Shape shape{32, 32};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    auto output = Tensor::from_vector(input, get_tensor_spec(shape, GetParam(), Layout::TILE)).to_vector<float>();
    EXPECT_THAT(output, Pointwise(FloatNear(4.0f), input));
}

TEST_P(BlockFloatVectorConversionTest, RoundtripWithPadding) {
    ttnn::Shape shape{14, 47};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    auto output = Tensor::from_vector(input, get_tensor_spec(shape, GetParam(), Layout::TILE));

    EXPECT_THAT(output.get_logical_shape(), ShapeIs(14, 47));
    EXPECT_THAT(output.get_padded_shape(), ShapeIs(32, 64));

    EXPECT_THAT(output.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

TEST_P(BlockFloatVectorConversionTest, RoundtripWithPaddingAndCustomTile) {
    ttnn::Shape shape{14, 47};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    TensorSpec spec(shape, TensorLayout(GetParam(), PageConfig(Layout::TILE, Tile({16, 16})), MemoryConfig{}));
    auto output = Tensor::from_vector(input, spec);

    EXPECT_THAT(output.get_logical_shape(), ShapeIs(14, 47));
    EXPECT_THAT(output.get_padded_shape(), ShapeIs(16, 48));

    EXPECT_THAT(output.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

INSTANTIATE_TEST_SUITE_P(
    BlockFloatVectorConversionTest,
    BlockFloatVectorConversionTest,
    ::testing::Values(DataType::BFLOAT4_B, DataType::BFLOAT8_B));

using DeviceVectorConversionTest = TTNNFixtureWithDevice;

TEST_F(DeviceVectorConversionTest, RoundtripWithMemoryConfig) {
    ttnn::Shape shape{128, 128};

    auto input = arange<float>(0, shape.volume(), 1);

    TensorSpec spec(
        shape, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{.buffer_type = BufferType::L1}));
    auto output = Tensor::from_vector(input, spec, device_);

    EXPECT_TRUE(is_device_tensor(output));
    EXPECT_TRUE(output.memory_config().is_l1());

    EXPECT_THAT(output.to_vector<float>(), Pointwise(Eq(), input));
}

// TODO: simplify into two layered class TYPED_TEST_P
class ShardVectorConversionTest : public ::testing::TestWithParam<TensorMemoryLayout> {};

TEST_P(ShardVectorConversionTest, UInt32RoundtripRowMajShardMapping) {
    ttnn::Shape shape{121, 128};

    auto input = arange<uint32_t>(0, shape.volume(), 1);

    auto tensor =
        Tensor::from_vector(input, get_tensor_spec_with_memory(shape, DataType::UINT32, Layout::ROW_MAJOR, GetParam()));

    auto output = tensor.template to_vector<uint32_t>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TEST_P(ShardVectorConversionTest, UInt32RoundtripTilizedShardMapping) {
    ttnn::Shape shape{121, 128};

    auto input = arange<uint32_t>(0, shape.volume(), 1);

    auto tensor =
        Tensor::from_vector(input, get_tensor_spec_with_memory(shape, DataType::UINT32, Layout::TILE, GetParam()));

    EXPECT_THAT(tensor.get_logical_shape(), ShapeIs(121, 128));
    EXPECT_THAT(tensor.get_padded_shape(), ShapeIs(128, 128));

    auto output = tensor.template to_vector<uint32_t>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TEST_P(ShardVectorConversionTest, FloatRoundtripRowMajShardMapping) {
    ttnn::Shape shape{121, 128};

    auto input = arange<float>(0, shape.volume(), 1);

    auto tensor = Tensor::from_vector(
        input, get_tensor_spec_with_memory(shape, DataType::FLOAT32, Layout::ROW_MAJOR, GetParam()));

    auto output = tensor.template to_vector<float>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TEST_P(ShardVectorConversionTest, FloatRoundtripTilizedShardMapping) {
    ttnn::Shape shape{121, 128};

    auto input = arange<float>(0, shape.volume(), 1);

    auto tensor =
        Tensor::from_vector(input, get_tensor_spec_with_memory(shape, DataType::FLOAT32, Layout::TILE, GetParam()));

    EXPECT_THAT(tensor.get_logical_shape(), ShapeIs(121, 128));
    EXPECT_THAT(tensor.get_padded_shape(), ShapeIs(128, 128));

    auto output = tensor.template to_vector<float>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TEST_P(ShardVectorConversionTest, Bfloat16RoundtripRowMajShardMapping) {
    ttnn::Shape shape{121, 128};

    auto input_bf16 = arange<bfloat16>(0, static_cast<int64_t>(shape.volume()), 1);

    std::vector<float> input_ft;
    input_ft.reserve(input_bf16.size());
    std::transform(
        input_bf16.begin(), input_bf16.end(), std::back_inserter(input_ft), [](bfloat16 bf) { return bf.to_float(); });

    auto tensor_bf = Tensor::from_vector(
        input_ft, get_tensor_spec_with_memory(shape, DataType::BFLOAT16, Layout::ROW_MAJOR, GetParam()));

    auto output_ft = tensor_bf.to_vector<float>();

    EXPECT_THAT(output_ft, Pointwise(Eq(), input_ft));
}

TEST_P(ShardVectorConversionTest, Bfloat16RoundtripTilizedShardMapping) {
    ttnn::Shape shape{121, 128};

    auto input_bf16 = arange<bfloat16>(0, static_cast<int64_t>(shape.volume()), 1);

    std::vector<float> input_ft;
    input_ft.reserve(input_bf16.size());
    std::transform(
        input_bf16.begin(), input_bf16.end(), std::back_inserter(input_ft), [](bfloat16 bf) { return bf.to_float(); });

    auto tensor_bf =
        Tensor::from_vector(input_ft, get_tensor_spec_with_memory(shape, DataType::BFLOAT16, Layout::TILE, GetParam()));

    EXPECT_THAT(tensor_bf.get_logical_shape(), ShapeIs(121, 128));
    EXPECT_THAT(tensor_bf.get_padded_shape(), ShapeIs(128, 128));

    auto output_ft = tensor_bf.to_vector<float>();

    EXPECT_THAT(output_ft, Pointwise(FloatNear(4.0f), input_ft));
}

TEST_P(ShardVectorConversionTest, BlockfloatRoundtripRowMajShardMapping) {
    ttnn::Shape shape{121, 128};

    auto input = arange<float>(0, shape.volume(), 1, 32);

    auto tensor =
        Tensor::from_vector(input, get_tensor_spec_with_memory(shape, DataType::BFLOAT8_B, Layout::TILE, GetParam()));

    EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

TEST_P(ShardVectorConversionTest, BlockfloatRoundtripTilizedShardMapping) {
    ttnn::Shape shape{121, 128};

    auto input = arange<float>(0, shape.volume(), 1, 32);

    auto tensor =
        Tensor::from_vector(input, get_tensor_spec_with_memory(shape, DataType::BFLOAT8_B, Layout::TILE, GetParam()));

    EXPECT_THAT(tensor.get_logical_shape(), ShapeIs(121, 128));
    EXPECT_THAT(tensor.get_padded_shape(), ShapeIs(128, 128));

    EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

INSTANTIATE_TEST_SUITE_P(
    ShardVectorConversionTests,
    ShardVectorConversionTest,
    ::testing::Values(
        TensorMemoryLayout::INTERLEAVED,
        TensorMemoryLayout::SINGLE_BANK,
        TensorMemoryLayout::HEIGHT_SHARDED,
        TensorMemoryLayout::WIDTH_SHARDED,
        TensorMemoryLayout::BLOCK_SHARDED));

}  // namespace

}  // namespace ttnn
