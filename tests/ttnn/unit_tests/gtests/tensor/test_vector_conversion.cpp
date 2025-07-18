// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xiterator.hpp>
#include <xtensor/core/xlayout.hpp>
#include <xtensor/utils/xtensor_simd.hpp>
#include <xtl/xiterator_base.hpp>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include "gmock/gmock.h"
#include <tt-metalium/shape.hpp>
#include <tt_stl/span.hpp>
#include "tests/ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp"
#include <tt-metalium/tile.hpp>
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

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

TensorSpec get_tensor_spec(
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout = Layout::ROW_MAJOR,
    const MemoryConfig& memory_config = MemoryConfig{}) {
    return TensorSpec(shape, TensorLayout(dtype, layout, memory_config));
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

TYPED_TEST(VectorConversionTest, InvalidSize) {
    ttnn::Shape shape{32, 32};
    auto input = arange<TypeParam>(0, 42, 1);

    ASSERT_NE(input.size(), shape.volume());
    EXPECT_ANY_THROW((void)Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>())));
}

TYPED_TEST(VectorConversionTest, InvalidDtype) {
    ttnn::Shape shape{32, 32};
    auto input = arange<TypeParam>(0, shape.volume(), 1);

    EXPECT_ANY_THROW((void)Tensor::from_vector(
        input,
        get_tensor_spec(
            shape,
            // Use INT32 for verification, except for when the actual type is int32_t.
            (std::is_same_v<TypeParam, int32_t> ? DataType::FLOAT32 : DataType::INT32))));
}

TYPED_TEST(VectorConversionTest, Roundtrip) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input = arange<TypeParam>(0, shape.volume(), 1);
        auto tensor = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>()));

        EXPECT_THAT(tensor.logical_shape(), Eq(shape)) << "for shape: " << shape;
        EXPECT_THAT(tensor.dtype(), Eq(convert_to_data_type<TypeParam>())) << "for shape: " << shape;

        auto output = tensor.template to_vector<TypeParam>();

        EXPECT_THAT(output, Pointwise(Eq(), input)) << "for shape: " << shape;
    }
}

TYPED_TEST(VectorConversionTest, RoundtripTilizedLayout) {
    ttnn::Shape shape{128, 128};
    auto input = arange<TypeParam>(0, shape.volume(), 1);
    auto tensor = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>(), Layout::TILE));

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(128, 128));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(128, 128));

    auto output = tensor.template to_vector<TypeParam>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TYPED_TEST(VectorConversionTest, RoundtripTilizedLayoutOddShape) {
    ttnn::Shape shape{1, 40, 3, 121};
    auto input = arange<TypeParam>(0, shape.volume(), 1);
    auto tensor = Tensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>(), Layout::TILE));

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(1, 40, 3, 121));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(1, 40, 32, 128));

    auto output = tensor.template to_vector<TypeParam>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TYPED_TEST(VectorConversionTest, RoundtripWithShardedLayout) {
    ttnn::Shape shape{56, 56, 30};
    auto input = arange<TypeParam>(0, shape.volume(), 1);
    auto tensor = Tensor::from_vector(
        input,
        get_tensor_spec(
            shape,
            convert_to_data_type<TypeParam>(),
            Layout::TILE,
            MemoryConfig{
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                ShardSpec{
                    ttnn::CoreRangeSet{ttnn::CoreRange{ttnn::CoreCoord{0, 0}, ttnn::CoreCoord{63, 63}}},
                    /*shard_shape_=*/{49, 30},
                    ShardOrientation::ROW_MAJOR,
                    ShardMode::LOGICAL}}));

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(56, 56, 30));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(56, 64, 32));

    auto output = tensor.template to_vector<TypeParam>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TEST(FloatVectorConversionTest, Float32Bfloat16Interop) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input_bf16 = arange<bfloat16>(0, shape.volume(), 1);
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

template <typename T>
class BorrowedStorageVectorConversionTest : public ::testing::Test {};

TYPED_TEST_SUITE(BorrowedStorageVectorConversionTest, TestTypes);

TYPED_TEST(BorrowedStorageVectorConversionTest, InvalidSize) {
    ttnn::Shape shape{32, 32};
    auto input = arange<TypeParam>(0, 42, 1);

    ASSERT_NE(input.size(), shape.volume());
    EXPECT_ANY_THROW((void)Tensor::from_borrowed_data(
        tt::stl::Span<TypeParam>(input),
        shape,
        /*on_creation_callback=*/[]() {},
        /*on_destruction_callback=*/[]() {}));
}

TYPED_TEST(BorrowedStorageVectorConversionTest, Roundtrip) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input = arange<TypeParam>(0, shape.volume(), 1);

        int ctor_count = 0;
        int dtor_count = 0;
        auto tensor = Tensor::from_borrowed_data(
            tt::stl::Span<TypeParam>(input),
            shape,
            /*on_creation_callback=*/[&]() { ctor_count++; },
            /*on_destruction_callback=*/[&]() { dtor_count++; });

        EXPECT_EQ(ctor_count, 1);
        EXPECT_EQ(dtor_count, 0);
        {
            Tensor copy(
                tensor.storage(), tensor.tensor_spec(), tensor.distributed_tensor_config(), tensor.tensor_topology());
            EXPECT_EQ(ctor_count, 2);
            EXPECT_EQ(dtor_count, 0);
        }
        EXPECT_EQ(ctor_count, 2);
        EXPECT_EQ(dtor_count, 1);

        EXPECT_THAT(tensor.logical_shape(), Eq(shape)) << "for shape: " << shape;
        EXPECT_THAT(tensor.dtype(), Eq(convert_to_data_type<TypeParam>())) << "for shape: " << shape;
        EXPECT_THAT(tensor.layout(), Eq(Layout::ROW_MAJOR)) << "for shape: " << shape;

        auto output = tensor.template to_vector<TypeParam>();

        EXPECT_THAT(output, Pointwise(Eq(), input)) << "for shape: " << shape;
    }
}

TYPED_TEST(BorrowedStorageVectorConversionTest, Callbacks) {
    ttnn::Shape shape{32, 32};
    auto input = arange<TypeParam>(0, shape.volume(), 1);

    int ctor_count = 0;
    int dtor_count = 0;
    auto tensor = Tensor::from_borrowed_data(
        tt::stl::Span<TypeParam>(input),
        shape,
        /*on_creation_callback=*/[&]() { ctor_count++; },
        /*on_destruction_callback=*/[&]() { dtor_count++; });

    EXPECT_EQ(ctor_count, 1);
    EXPECT_EQ(dtor_count, 0);
    {
        Tensor copy(
            tensor.storage(), tensor.tensor_spec(), tensor.distributed_tensor_config(), tensor.tensor_topology());
        EXPECT_EQ(ctor_count, 2);
        EXPECT_EQ(dtor_count, 0);
    }
    EXPECT_EQ(ctor_count, 2);
    EXPECT_EQ(dtor_count, 1);
}

TYPED_TEST(BorrowedStorageVectorConversionTest, CustomTile) {
    ttnn::Shape shape{32, 32};
    auto input = arange<TypeParam>(0, shape.volume(), 1);

    auto tensor = Tensor::from_borrowed_data(
        tt::stl::Span<TypeParam>(input),
        shape,
        /*on_creation_callback=*/[]() {},
        /*on_destruction_callback=*/[]() {},
        /*tile=*/Tile({16, 16}));

    // Retain row major layout, but use custom tile.
    // TODO: #18536 - this should be illegal.
    EXPECT_EQ(tensor.tensor_spec().layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.tensor_spec().tile(), Tile({16, 16}));
}

class BlockFloatVectorConversionTest : public ::testing::TestWithParam<DataType> {};

TEST_P(BlockFloatVectorConversionTest, InvalidLayout) {
    ttnn::Shape shape{32, 32};
    // Block float types are only supported in TILE layout.
    EXPECT_ANY_THROW((void)Tensor::from_vector(
        std::vector<float>(shape.volume()), get_tensor_spec(shape, GetParam(), Layout::ROW_MAJOR)));
}

TEST_P(BlockFloatVectorConversionTest, Roundtrip) {
    ttnn::Shape shape{32, 32};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    auto tensor = Tensor::from_vector(input, get_tensor_spec(shape, GetParam(), Layout::TILE));

    EXPECT_THAT(tensor.logical_shape(), Eq(shape));
    EXPECT_THAT(tensor.dtype(), Eq(GetParam()));
    EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

TEST_P(BlockFloatVectorConversionTest, RoundtripWithPadding) {
    ttnn::Shape shape{14, 47};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    auto tensor = Tensor::from_vector(input, get_tensor_spec(shape, GetParam(), Layout::TILE));

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(14, 47));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(32, 64));
    EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

TEST_P(BlockFloatVectorConversionTest, RoundtripWithPaddingAndCustomTile) {
    ttnn::Shape shape{14, 47};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    TensorSpec spec(shape, TensorLayout(GetParam(), PageConfig(Layout::TILE, Tile({16, 16})), MemoryConfig{}));
    auto tensor = Tensor::from_vector(input, spec);

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(14, 47));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(16, 48));
    EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
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
        shape,
        TensorLayout(
            DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));
    auto output = Tensor::from_vector(input, spec, device_);

    EXPECT_TRUE(is_device_tensor(output));
    EXPECT_TRUE(output.memory_config().is_l1());

    EXPECT_THAT(output.to_vector<float>(), Pointwise(Eq(), input));
}

}  // namespace

}  // namespace ttnn
