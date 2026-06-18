// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;

template <typename... Args>
testing::Matcher<Shape> ShapeIs(Args... args) {
    return testing::Eq(Shape({args...}));
}

const std::vector<Shape>& get_shapes_for_test() {
    static auto* shapes = new std::vector<Shape>{
        Shape{1},
        Shape{1, 1, 1, 1},
        Shape{1, 1, 1, 10},
        Shape{1, 32, 32, 16},
        Shape{1, 40, 3, 128},
        Shape{2, 2},
        Shape{1, 1, 1, 1, 10},
    };
    return *shapes;
}

TensorSpec get_tensor_spec(
    const Shape& shape,
    DataType dtype,
    Layout layout = Layout::ROW_MAJOR,
    const MemoryConfig& memory_config = MemoryConfig{}) {
    return TensorSpec(shape, TensorLayout(dtype, layout, memory_config));
}

template <typename T>
std::vector<T> arange(int64_t start, int64_t end, int64_t step, std::optional<int64_t> cap = std::nullopt) {
    std::vector<T> result;
    for (int64_t el = start; el < end; el += step) {
        int64_t capped_el = cap ? el % *cap : el;
        result.push_back(static_cast<T>(capped_el));
    }
    return result;
}

template <typename T>
class VectorConversionTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, bfloat16, uint8_t, uint16_t, uint32_t, int32_t>;
TYPED_TEST_SUITE(VectorConversionTest, TestTypes);

TYPED_TEST(VectorConversionTest, InvalidSize) {
    Shape shape{32, 32};
    auto input = arange<TypeParam>(0, 42, 1);

    ASSERT_NE(input.size(), shape.volume());
    EXPECT_ANY_THROW((void)HostTensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>())));
}

TYPED_TEST(VectorConversionTest, Roundtrip) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input = arange<TypeParam>(0, shape.volume(), 1);
        auto tensor = HostTensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>()));

        EXPECT_THAT(tensor.logical_shape(), Eq(shape)) << "for shape: " << shape;
        EXPECT_THAT(tensor.dtype(), Eq(convert_to_data_type<TypeParam>())) << "for shape: " << shape;

        auto output = tensor.template to_vector<TypeParam>();

        EXPECT_THAT(output, Pointwise(Eq(), input)) << "for shape: " << shape;
    }
}

TYPED_TEST(VectorConversionTest, RoundtripTilizedLayout) {
    Shape shape{128, 128};
    auto input = arange<TypeParam>(0, shape.volume(), 1);
    auto tensor =
        HostTensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>(), Layout::TILE));

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(128, 128));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(128, 128));

    auto output = tensor.template to_vector<TypeParam>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TYPED_TEST(VectorConversionTest, RoundtripTilizedLayoutOddShape) {
    Shape shape{1, 40, 3, 121};
    auto input = arange<TypeParam>(0, shape.volume(), 1);
    auto tensor =
        HostTensor::from_vector(input, get_tensor_spec(shape, convert_to_data_type<TypeParam>(), Layout::TILE));

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(1, 40, 3, 121));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(1, 40, 32, 128));

    auto output = tensor.template to_vector<TypeParam>();

    EXPECT_THAT(output, Pointwise(Eq(), input));
}

TEST(FloatVectorConversionTest, Float32Bfloat16Interop) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input_bf16 = arange<bfloat16>(0, shape.volume(), 1);
        std::vector<float> input_ft;
        input_ft.reserve(input_bf16.size());
        std::transform(input_bf16.begin(), input_bf16.end(), std::back_inserter(input_ft), [](bfloat16 bf) {
            return static_cast<float>(bf);
        });

        auto output_bf16 =
            HostTensor::from_vector(input_ft, get_tensor_spec(shape, DataType::BFLOAT16)).to_vector<bfloat16>();
        EXPECT_THAT(output_bf16, Pointwise(Eq(), input_bf16)) << "for shape: " << shape;

        auto output_ft =
            HostTensor::from_vector(input_bf16, get_tensor_spec(shape, DataType::BFLOAT16)).to_vector<float>();
        EXPECT_THAT(output_ft, Pointwise(Eq(), input_ft)) << "for shape: " << shape;
    }
}

template <typename T>
class BorrowedStorageVectorConversionTest : public ::testing::Test {};

TYPED_TEST_SUITE(BorrowedStorageVectorConversionTest, TestTypes);

TYPED_TEST(BorrowedStorageVectorConversionTest, InvalidSize) {
    Shape shape{32, 32};
    auto input = arange<TypeParam>(0, 42, 1);

    ASSERT_NE(input.size(), shape.volume());
    EXPECT_ANY_THROW((void)HostTensor::from_borrowed_data(
        std::span<TypeParam>(input),
        shape,
        MemoryPin(/*increment_ref_count=*/[]() {}, /*decrement_ref_count=*/[]() {})));
}

TYPED_TEST(BorrowedStorageVectorConversionTest, Roundtrip) {
    for (const auto& shape : get_shapes_for_test()) {
        auto input = arange<TypeParam>(0, shape.volume(), 1);

        int ctor_count = 0;
        int dtor_count = 0;
        auto tensor = HostTensor::from_borrowed_data(
            std::span<TypeParam>(input),
            shape,
            MemoryPin(
                /*increment_ref_count=*/[&]() { ctor_count++; },
                /*decrement_ref_count=*/[&]() { dtor_count++; }));

        EXPECT_EQ(ctor_count, 1);
        EXPECT_EQ(dtor_count, 0);
        {
            HostTensor copy(tensor.buffer(), tensor.tensor_spec(), tensor.tensor_topology());
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
    Shape shape{32, 32};
    auto input = arange<TypeParam>(0, shape.volume(), 1);

    int ctor_count = 0;
    int dtor_count = 0;
    auto tensor = HostTensor::from_borrowed_data(
        std::span<TypeParam>(input),
        shape,
        MemoryPin(
            /*increment_ref_count=*/[&]() { ctor_count++; },
            /*decrement_ref_count=*/[&]() { dtor_count++; }));

    EXPECT_EQ(ctor_count, 1);
    EXPECT_EQ(dtor_count, 0);
    {
        HostTensor copy(tensor.buffer(), tensor.tensor_spec(), tensor.tensor_topology());
        EXPECT_EQ(ctor_count, 2);
        EXPECT_EQ(dtor_count, 0);
    }
    EXPECT_EQ(ctor_count, 2);
    EXPECT_EQ(dtor_count, 1);
}

TYPED_TEST(BorrowedStorageVectorConversionTest, CustomTile) {
    Shape shape{32, 32};
    auto input = arange<TypeParam>(0, shape.volume(), 1);

    auto tensor = HostTensor::from_borrowed_data(
        std::span<TypeParam>(input),
        shape,
        MemoryPin(/*increment_ref_count=*/[]() {}, /*decrement_ref_count=*/[]() {}),
        /*tile=*/Tile({16, 16}));

    // Retain row major layout, but use custom tile.
    // TODO: #18536 - this should be illegal.
    EXPECT_EQ(tensor.tensor_spec().layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.tensor_spec().tile(), Tile({16, 16}));
}

class BlockFloatVectorConversionTest : public ::testing::TestWithParam<DataType> {};

TEST_P(BlockFloatVectorConversionTest, InvalidLayout) {
    Shape shape{32, 32};
    // Block float types are only supported in TILE layout.
    EXPECT_ANY_THROW((void)HostTensor::from_vector(
        std::vector<float>(shape.volume()), get_tensor_spec(shape, GetParam(), Layout::ROW_MAJOR)));
}

TEST_P(BlockFloatVectorConversionTest, Roundtrip) {
    Shape shape{32, 32};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    auto tensor = HostTensor::from_vector(input, get_tensor_spec(shape, GetParam(), Layout::TILE));

    EXPECT_THAT(tensor.logical_shape(), Eq(shape));
    EXPECT_THAT(tensor.dtype(), Eq(GetParam()));
    EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

TEST_P(BlockFloatVectorConversionTest, RoundtripWithPadding) {
    Shape shape{14, 47};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    auto tensor = HostTensor::from_vector(input, get_tensor_spec(shape, GetParam(), Layout::TILE));

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(14, 47));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(32, 64));
    EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

TEST_P(BlockFloatVectorConversionTest, RoundtripWithPaddingAndCustomTile) {
    Shape shape{14, 47};
    std::vector<float> input = arange<float>(0, shape.volume(), 1, /*cap=*/32);

    TensorSpec spec(shape, TensorLayout(GetParam(), PageConfig(Layout::TILE, Tile({16, 16})), MemoryConfig{}));
    auto tensor = HostTensor::from_vector(input, spec);

    EXPECT_THAT(tensor.logical_shape(), ShapeIs(14, 47));
    EXPECT_THAT(tensor.padded_shape(), ShapeIs(16, 48));
    EXPECT_THAT(tensor.to_vector<float>(), Pointwise(FloatNear(4.0f), input));
}

INSTANTIATE_TEST_SUITE_P(
    BlockFloatVectorConversionTest,
    BlockFloatVectorConversionTest,
    ::testing::Values(DataType::BFLOAT4_B, DataType::BFLOAT8_B));

using DeviceVectorConversionTest = MeshDevice1x1Fixture;

TEST_F(DeviceVectorConversionTest, RoundtripWithMemoryConfig) {
    Shape shape{128, 128};

    auto input = arange<float>(0, shape.volume(), 1);

    TensorSpec spec(
        shape,
        TensorLayout(
            DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));
    MemoryConfig mem_cfg{TensorMemoryLayout::INTERLEAVED, BufferType::L1};

    auto host = HostTensor::from_vector(input, spec);
    auto mesh = enqueue_write_tensor(mesh_device_->mesh_command_queue(), host, *mesh_device_, mem_cfg);

    EXPECT_TRUE(mesh.memory_config().is_l1());

    auto readback = enqueue_read_tensor(mesh_device_->mesh_command_queue(), mesh);

    EXPECT_THAT(readback.to_vector<float>(), Pointwise(Eq(), input));
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace tt::tt_metal
