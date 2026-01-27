// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "memory_pin.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/tensor.hpp"

/**
 * @brief Unit tests Tensor class to verify it's behavior when constructed on host.
 *
 * This is used to verify parity between TTNN::Tensor vs tt_metal::HostTensor in the future.
 */

namespace ttnn {
namespace {

using TensorUnderTest = tt::tt_metal::Tensor;

using PageConfig = tt::tt_metal::PageConfig;
using MemoryConfig = tt::tt_metal::MemoryConfig;
using TensorLayout = tt::tt_metal::TensorLayout;
using DataType = tt::tt_metal::DataType;
using Layout = tt::tt_metal::Layout;
using Shape = tt::tt_metal::Shape;
using HostBuffer = tt::tt_metal::HostBuffer;
using MemoryPin = tt::tt_metal::MemoryPin;
using TensorMemoryLayout = tt::tt_metal::TensorMemoryLayout;
using BufferType = tt::tt_metal::BufferType;

template <typename T>
std::vector<T> create_test_vector(std::size_t size) {
    std::vector<T> vec(size);
    for (std::size_t i = 0; i < size; i++) {
        vec[i] = static_cast<T>(i);
    }
    return vec;
}

// Special Member functions

TEST(TensorHostBehaviorTest, DefaultConstruction) {
    // Note: while ttnn Tensor can be default constructed, it's attributes are not accessible.
    // (nullptr segfault on access)
    TensorUnderTest tensor;
    (void)tensor;
}

TEST(TensorHostBehaviorTest, Copy_DefaultConstructed) {
    TensorUnderTest tensor;
    TensorUnderTest tensor_copy(tensor);
    (void)tensor_copy;
}

TEST(TensorHostBehaviorTest, Copy) {
    {
        auto vec = create_test_vector<float>(32 * 4);
        TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
        TensorUnderTest tensor_copy(tensor);
        EXPECT_EQ(tensor.tensor_spec(), tensor_copy.tensor_spec());
        EXPECT_EQ(tensor.to_vector<float>(), tensor_copy.to_vector<float>());
        (void)tensor_copy;
    }

    // Different Tensor Spec
    {
        auto vec = create_test_vector<int>(32 * 32);
        TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 32}), DataType::INT32, Layout::TILE);
        TensorUnderTest tensor_copy(tensor);
        EXPECT_EQ(tensor.tensor_spec(), tensor_copy.tensor_spec());
        EXPECT_EQ(tensor.to_vector<int>(), tensor_copy.to_vector<int>());
        (void)tensor_copy;
    }
}

TEST(TensorHostBehaviorTest, Copy_BorrowedData) {
    // Borrowed data is not deep copied
    auto vec = create_test_vector<float>(32 * 4);
    auto tensor = TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({32, 4}), tt::tt_metal::MemoryPin());
    TensorUnderTest tensor_copy(tensor);
    vec[0] = 100;
    EXPECT_EQ(tensor.to_vector<float>(), tensor_copy.to_vector<float>());
    (void)tensor_copy;
}

TEST(TensorHostBehaviorTest, CopyAssignment) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    TensorUnderTest tensor_copy;
    tensor_copy = tensor;
    EXPECT_EQ(tensor.tensor_spec(), tensor_copy.tensor_spec());
    EXPECT_EQ(tensor.to_vector<float>(), tensor_copy.to_vector<float>());
    (void)tensor_copy;
}

// Move is trivially handled by the smart pointers in the impl.

// HostBuffer constructor

TEST(TensorHostBehaviorTest, HostBufferConstructor_Overloads) {
    // Shape overload: HostBuffer size matches shape
    {
        auto vec = create_test_vector<float>(32 * 4);
        TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
        EXPECT_EQ(tensor.logical_shape(), Shape({32, 4}));
        EXPECT_EQ(tensor.padded_shape(), Shape({32, 4}));
        EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
        EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    }

    // Padded shape overload: HostBuffer size matches padded shape
    {
        auto vec = create_test_vector<float>(64 * 8);
        TensorUnderTest tensor(
            HostBuffer(std::move(vec)), Shape({32, 4}), Shape({64, 8}), DataType::FLOAT32, Layout::ROW_MAJOR);
        EXPECT_EQ(tensor.logical_shape(), Shape({32, 4}));
        EXPECT_EQ(tensor.padded_shape(), Shape({64, 8}));
        EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
        EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    }

    // TensorSpec overload: HostBuffer size matches TensorSpec
    {
        auto vec = create_test_vector<float>(32 * 4);
        TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
        TensorSpec spec(Shape({32, 4}), tensor_layout);
        TensorUnderTest tensor(HostBuffer(std::move(vec)), spec);
        EXPECT_EQ(tensor.tensor_spec(), spec);
        EXPECT_EQ(tensor.logical_shape(), Shape({32, 4}));
        EXPECT_EQ(tensor.padded_shape(), Shape({32, 4}));
        EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
        EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    }
}

TEST(TensorHostBehaviorTest, HostBufferConstructor_EmptyBuffer) {
    TensorUnderTest tensor(HostBuffer(), Shape({0}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), Shape({0}));
    EXPECT_EQ(tensor.padded_shape(), Shape({0}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

// Tensor currently does not validate the size of the HostBuffer against the shape or padded shape.
TEST(TensorHostBehaviorTest, HostBufferConstructor_SizeMismatch_TooFewElements) {
    // HostBuffer has fewer elements than shape requires - does not throw (no validation)
    auto vec = create_test_vector<float>(16);  // Only 16 elements, shape requires 128
    EXPECT_NO_THROW(
        (void)TensorUnderTest(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR));
}

TEST(TensorHostBehaviorTest, HostBufferConstructor_SizeMismatch_TooManyElements) {
    // HostBuffer has more elements than shape requires - does not throw (no validation)
    auto vec = create_test_vector<float>(256);  // 256 elements, shape requires 128
    EXPECT_NO_THROW(
        (void)TensorUnderTest(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR));
}

// Tensor currently does not validate the size of the HostBuffer against the shape or padded shape.
TEST(TensorHostBehaviorTest, HostBufferConstructor_DTypeMismatch) {
    // HostBuffer contains floats but DataType specifies INT32 - does not throw (no validation)
    auto vec = create_test_vector<float>(32 * 4);
    EXPECT_NO_THROW(
        (void)TensorUnderTest(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::INT32, Layout::ROW_MAJOR));
}

// from_borrowed_data

TEST(TensorHostBehaviorTest, from_borrowed_data) {
    // It is assumed for the borrowed data to be in row-major order.
    auto vec = create_test_vector<float>(32 * 4);
    auto tensor = TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({32, 4}), MemoryPin());
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.padded_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);

    // assert no copy is made
    vec[0] = 100;
    EXPECT_EQ(tensor.to_vector<float>(), vec);
}

TEST(TensorHostBehaviorTest, from_borrowed_data_EmptyTensor) {
    std::vector<float> vec;
    auto tensor = TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({0}), MemoryPin());
    EXPECT_EQ(tensor.logical_shape(), Shape({0}));
    EXPECT_EQ(tensor.padded_shape(), Shape({0}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

// from_borrowed_data validates the buffer size against the shape.
TEST(TensorHostBehaviorTest, from_borrowed_data_SizeMismatch_TooFewElements) {
    auto vec = create_test_vector<float>(16);  // Only 16 elements, shape requires 128
    EXPECT_ANY_THROW((void)TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({32, 4}), MemoryPin()));
}

TEST(TensorHostBehaviorTest, from_borrowed_data_SizeMismatch_TooManyElements) {
    auto vec = create_test_vector<float>(256);  // 256 elements, shape requires 128
    EXPECT_ANY_THROW((void)TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({32, 4}), MemoryPin()));
}

// from_span

TEST(TensorHostBehaviorTest, from_span) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    auto tensor = TensorUnderTest::from_span(std::span<const float>(vec), spec);
    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.padded_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.to_vector<float>(), vec);
}

TEST(TensorHostBehaviorTest, from_span_EmptyTensor) {
    std::vector<float> vec;
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({0}), tensor_layout);
    auto tensor = TensorUnderTest::from_span(std::span<const float>(vec), spec);
    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({0}));
    EXPECT_EQ(tensor.padded_shape(), Shape({0}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

TEST(TensorHostBehaviorTest, from_span_SizeMismatch_TooFewElements) {
    auto vec = create_test_vector<float>(16);  // Only 16 elements, shape requires 128
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_span(std::span<const float>(vec), spec));
}

TEST(TensorHostBehaviorTest, from_span_SizeMismatch_TooManyElements) {
    auto vec = create_test_vector<float>(256);  // 256 elements, shape requires 128
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_span(std::span<const float>(vec), spec));
}

TEST(TensorHostBehaviorTest, from_span_Tile_BFloat16) {
    // BFLOAT16 uses float for input/output, TILE layout requires tile-aligned shape
    auto vec = create_test_vector<float>(32 * 32);
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{});
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_span(std::span<const float>(vec), spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor.layout(), Layout::TILE);
    // Round-trip check (values may lose precision due to bfloat16)
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result.size(), vec.size());
}

// from_vector

TEST(TensorHostBehaviorTest, from_vector) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.padded_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.to_vector<float>(), vec);
}

TEST(TensorHostBehaviorTest, from_vector_EmptyTensor) {
    std::vector<float> vec;
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({0}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({0}));
    EXPECT_EQ(tensor.padded_shape(), Shape({0}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

TEST(TensorHostBehaviorTest, from_vector_SizeMismatch_TooFewElements) {
    auto vec = create_test_vector<float>(16);  // Only 16 elements, shape requires 128
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_vector(vec, spec));
}

TEST(TensorHostBehaviorTest, from_vector_SizeMismatch_TooManyElements) {
    auto vec = create_test_vector<float>(256);  // 256 elements, shape requires 128
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_vector(vec, spec));
}

TEST(TensorHostBehaviorTest, from_vector_Tile_BFloat16) {
    // BFLOAT16 uses float for input/output, TILE layout requires tile-aligned shape
    auto vec = create_test_vector<float>(32 * 32);
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{});
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor.layout(), Layout::TILE);
    // Round-trip check (values may lose precision due to bfloat16)
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result.size(), vec.size());
}

// from_vector rvalue overload

TEST(TensorHostBehaviorTest, from_vector_Rvalue) {
    auto vec = create_test_vector<float>(32 * 4);
    auto vec_copy = vec;  // Keep a copy for comparison
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(std::move(vec), spec);
    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.padded_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.to_vector<float>(), vec_copy);
}

TEST(TensorHostBehaviorTest, from_vector_Rvalue_EmptyTensor) {
    std::vector<float> vec;
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({0}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(std::move(vec), spec);
    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({0}));
    EXPECT_EQ(tensor.padded_shape(), Shape({0}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

TEST(TensorHostBehaviorTest, from_vector_Rvalue_SizeMismatch_TooFewElements) {
    auto vec = create_test_vector<float>(16);  // Only 16 elements, shape requires 128
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_vector(std::move(vec), spec));
}

TEST(TensorHostBehaviorTest, from_vector_Rvalue_SizeMismatch_TooManyElements) {
    auto vec = create_test_vector<float>(256);  // 256 elements, shape requires 128
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_vector(std::move(vec), spec));
}

TEST(TensorHostBehaviorTest, from_vector_Rvalue_Tile_BFloat16) {
    // BFLOAT16 uses float for input/output, TILE layout requires tile-aligned shape
    auto vec = create_test_vector<float>(32 * 32);
    auto vec_copy = vec;  // Keep a copy for size comparison
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{});
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(std::move(vec), spec);
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor.layout(), Layout::TILE);
    // Round-trip check (values may lose precision due to bfloat16)
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result.size(), vec_copy.size());
}

// to_vector

TEST(TensorHostBehaviorTest, to_vector_RowMajor_Float32) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(vec), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result, vec);
}

TEST(TensorHostBehaviorTest, to_vector_RowMajor_BFloat16) {
    // BFLOAT16 uses float for input/output but stores as bfloat16
    auto vec = create_test_vector<float>(32 * 4);
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    // Note: values may lose precision due to bfloat16 conversion
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result.size(), vec.size());
}

TEST(TensorHostBehaviorTest, to_vector_Tile_Float32) {
    // TILE layout requires shape to be tile-aligned (32x32)
    // Use from_vector to ensure proper row-major to tile conversion
    auto vec = create_test_vector<float>(32 * 32);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::TILE), MemoryConfig{});
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result, vec);
}

TEST(TensorHostBehaviorTest, to_vector_EmptyTensor) {
    // Use from_vector for empty tensor to avoid segfault with default HostBuffer
    std::vector<float> vec;
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({0}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

TEST(TensorHostBehaviorTest, to_vector_DTypeMismatch_Float32ToInt32) {
    // Tensor is FLOAT32, but requesting INT32 - should throw
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(vec), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_ANY_THROW((void)tensor.to_vector<int32_t>());
}

// is_sharded
// Note: is_sharded() returns false for all host tensors
// Even if the TensorSpec has a sharded MemoryConfig, the tensor itself is not sharded until on device.

TEST(TensorHostBehaviorTest, is_sharded_HostTensor_WithShardedMemoryConfig) {
    // Even with a sharded MemoryConfig, host tensors are not sharded
    auto vec = create_test_vector<float>(32 * 32);
    MemoryConfig sharded_mem_config(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), sharded_mem_config);
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    // Verify MemoryConfig is sharded
    EXPECT_TRUE(sharded_mem_config.is_sharded());
    // But host tensor is not sharded
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_FALSE(tensor.is_sharded());
}

}  // namespace
}  // namespace ttnn
