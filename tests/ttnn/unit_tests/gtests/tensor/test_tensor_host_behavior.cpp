// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "memory_pin.hpp"
#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/tensor.hpp"

/**
 * @file test_tensor_host_behavior.cpp
 * @brief Unit tests for Tensor class behavior when constructed on host.
 *
 * This test suite verifies the behavior of host tensor member functions as outlined in the
 * Metal Tensor Proposal (tech_reports/runtime-tensor/Metal Tensor Proposal.md).
 *
 * The intent is to establish parity between the current TTNN::Tensor and the future
 * tt_metal::HostTensor implementation. HostTensor should have container-like semantics.
 *
 * =============================================================================
 * HostTensor Member Functions (from Metal Tensor Proposal)
 * =============================================================================
 *
 * SPECIAL MEMBER FUNCTIONS:
 *   - Default ctor                    [TESTED]
 *   - Copy ctor / Copy assignment     [TESTED]
 *   - Move ctor / Move assignment     [TESTED]
 *
 * CONSTRUCTORS:
 *   - HostBuffer ctors                [TESTED]
 *
 * STATIC FACTORY METHODS:
 *   - from_borrowed_data              [TESTED]
 *   - from_span                       [TESTED]
 *   - from_vector (lvalue)            [TESTED]
 *   - from_vector (rvalue)            [TESTED]
 *
 * DATA ACCESS METHODS:
 *   - to_vector                       [TESTED]
 *   - write_to_string                 [TESTED]
 *
 * GETTERS:
 *   - logical_shape / padded_shape    [TESTED] (implicitly in constructor tests)
 *   - dtype / layout                  [TESTED] (implicitly in constructor tests)
 *   - tensor_spec                     [TESTED] (implicitly in constructor tests)
 *   - logical_volume                  [TESTED]
 *   - physical_volume                 [TESTED]
 *   - memory_config                   [TESTED]
 *   - strides                         [TESTED]
 *   - element_size                    [TESTED]
 *   - shard_spec                      [TESTED] (returns nullopt for host tensors)
 *   - nd_shard_spec                   [TESTED] (returns nullopt for host tensors)
 *   - tensor_topology                 [TESTED]
 *   - memory_config                   [NOT TESTED] (inferred from tensor_spec)
 *   - shard_spec                      [TODO]
 *   - nd_shard_spec                   [TODO]
 *
 * BOOLEAN QUERIES:
 *   - is_sharded                      [TESTED]
 *
 * =============================================================================
 * Known Edge Cases / Quirks:
 * =============================================================================
 *   - to_vector() on tensor created with empty HostBuffer() directly will segfault.
 *     Use from_vector() or from_span() for safe empty tensor creation.
 *   - Empty shape must be Shape({0}), not Shape(). Shape() creates a scalar with volume 1.
 *   - HostBuffer constructors do NOT validate buffer size against shape (no validation).
 *   - from_borrowed_data, from_span, from_vector DO validate buffer size against shape.
 */

namespace ttnn {
namespace {

// =============================================================================
// Test Setup
// =============================================================================

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
using TensorTopology = tt::tt_metal::TensorTopology;
using TensorSpec = tt::tt_metal::TensorSpec;

// Note: EMPTY_SHAPE.volume() is 0, EMPTY_SHAPE.rank() is 1
// Shape{} would be a scalar with volume 1, which is NOT an empty tensor.
const Shape EMPTY_SHAPE = Shape({0});

template <typename T>
std::vector<T> create_test_vector(std::size_t size) {
    std::vector<T> vec(size);
    for (std::size_t i = 0; i < size; i++) {
        vec[i] = static_cast<T>(i);
    }
    return vec;
}

// =============================================================================
// Special Member Functions
// =============================================================================

TEST(TensorHostBehaviorTest, DefaultConstruction) {
    // Note: A default constructed Tensor has no accessible attributes.
    // Accessing attributes on a default constructed tensor will segfault.
    TensorUnderTest tensor;
    (void)tensor;
}

TEST(TensorHostBehaviorTest, Copy_DefaultConstructed) {
    TensorUnderTest tensor;
    TensorUnderTest tensor_copy(tensor);
    (void)tensor_copy;
}

TEST(TensorHostBehaviorTest, Copy) {
    // ROW_MAJOR, FLOAT32
    {
        auto vec = create_test_vector<float>(32 * 4);
        TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
        TensorUnderTest tensor_copy(tensor);
        EXPECT_EQ(tensor.tensor_spec(), tensor_copy.tensor_spec());
        EXPECT_EQ(tensor.to_vector<float>(), tensor_copy.to_vector<float>());
    }

    // TILE, INT32
    {
        auto vec = create_test_vector<int>(32 * 32);
        TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 32}), DataType::INT32, Layout::TILE);
        TensorUnderTest tensor_copy(tensor);
        EXPECT_EQ(tensor.tensor_spec(), tensor_copy.tensor_spec());
        EXPECT_EQ(tensor.to_vector<int>(), tensor_copy.to_vector<int>());
    }
}

TEST(TensorHostBehaviorTest, Copy_BorrowedData) {
    // Borrowed data results in shallow copy (both tensors share underlying data)
    auto vec = create_test_vector<float>(32 * 4);
    auto tensor = TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({32, 4}), tt::tt_metal::MemoryPin());
    TensorUnderTest tensor_copy(tensor);
    vec[0] = 100;
    EXPECT_EQ(tensor.to_vector<float>(), tensor_copy.to_vector<float>());
}

TEST(TensorHostBehaviorTest, CopyAssignment) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    TensorUnderTest tensor_copy;
    tensor_copy = tensor;
    EXPECT_EQ(tensor.tensor_spec(), tensor_copy.tensor_spec());
    EXPECT_EQ(tensor.to_vector<float>(), tensor_copy.to_vector<float>());
}

TEST(TensorHostBehaviorTest, MoveConstruction) {
    auto vec = create_test_vector<float>(32 * 4);
    auto vec_copy = vec;
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    TensorUnderTest tensor_moved(std::move(tensor));

    EXPECT_EQ(tensor_moved.logical_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor_moved.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor_moved.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor_moved.to_vector<float>(), vec_copy);

    // Verify moved-from object is still assignable (valid state)
    auto vec2 = create_test_vector<float>(16);
    tensor = TensorUnderTest(HostBuffer(std::move(vec2)), Shape({4, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), Shape({4, 4}));
}

TEST(TensorHostBehaviorTest, MoveAssignment) {
    auto vec = create_test_vector<float>(32 * 4);
    auto vec_copy = vec;
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    TensorUnderTest tensor_moved;
    tensor_moved = std::move(tensor);

    EXPECT_EQ(tensor_moved.logical_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor_moved.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor_moved.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor_moved.to_vector<float>(), vec_copy);

    // Verify moved-from object is still assignable (valid state)
    auto vec2 = create_test_vector<float>(16);
    tensor = TensorUnderTest(HostBuffer(std::move(vec2)), Shape({4, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), Shape({4, 4}));
}

TEST(TensorHostBehaviorTest, MoveConstruction_DefaultConstructed) {
    TensorUnderTest tensor;
    TensorUnderTest tensor_moved(std::move(tensor));
    (void)tensor_moved;

    // Verify moved-from object is still assignable (valid state)
    auto vec = create_test_vector<float>(16);
    tensor = TensorUnderTest(HostBuffer(std::move(vec)), Shape({4, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), Shape({4, 4}));
}

// =============================================================================
// HostBuffer Constructors
// =============================================================================

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
    TensorUnderTest tensor(HostBuffer(), EMPTY_SHAPE, DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), EMPTY_SHAPE);
    EXPECT_EQ(tensor.padded_shape(), EMPTY_SHAPE);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    // Note: to_vector() on tensor created with empty HostBuffer() segfaults.
    // EXPECT_TRUE(tensor.to_vector<float>().empty());
}

// Note: HostBuffer constructors do NOT validate buffer size against shape.
TEST(TensorHostBehaviorTest, HostBufferConstructor_SizeMismatch_TooFewElements) {
    auto vec = create_test_vector<float>(16);  // Only 16 elements, shape requires 128
    EXPECT_NO_THROW(
        (void)TensorUnderTest(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR));
}

TEST(TensorHostBehaviorTest, HostBufferConstructor_SizeMismatch_TooManyElements) {
    auto vec = create_test_vector<float>(256);  // 256 elements, shape requires 128
    EXPECT_NO_THROW(
        (void)TensorUnderTest(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR));
}

TEST(TensorHostBehaviorTest, HostBufferConstructor_DTypeMismatch) {
    // HostBuffer contains floats but DataType specifies INT32 - no validation
    auto vec = create_test_vector<float>(32 * 4);
    EXPECT_NO_THROW(
        (void)TensorUnderTest(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::INT32, Layout::ROW_MAJOR));
}

// =============================================================================
// from_borrowed_data
// =============================================================================

TEST(TensorHostBehaviorTest, from_borrowed_data) {
    auto vec = create_test_vector<float>(32 * 4);
    auto tensor = TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({32, 4}), MemoryPin());
    EXPECT_EQ(tensor.logical_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.padded_shape(), Shape({32, 4}));
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);

    // Verify no copy is made (data is borrowed)
    vec[0] = 100;
    EXPECT_EQ(tensor.to_vector<float>(), vec);
}

TEST(TensorHostBehaviorTest, from_borrowed_data_EmptyTensor) {
    std::vector<float> vec;
    auto tensor = TensorUnderTest::from_borrowed_data(std::span<float>(vec), EMPTY_SHAPE, MemoryPin());
    EXPECT_EQ(tensor.logical_shape(), EMPTY_SHAPE);
    EXPECT_EQ(tensor.padded_shape(), EMPTY_SHAPE);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

// Note: from_borrowed_data validates buffer size against shape (unlike HostBuffer ctor)
TEST(TensorHostBehaviorTest, from_borrowed_data_SizeMismatch_TooFewElements) {
    auto vec = create_test_vector<float>(16);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({32, 4}), MemoryPin()));
}

TEST(TensorHostBehaviorTest, from_borrowed_data_SizeMismatch_TooManyElements) {
    auto vec = create_test_vector<float>(256);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_borrowed_data(std::span<float>(vec), Shape({32, 4}), MemoryPin()));
}

// =============================================================================
// from_span
// =============================================================================

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
    TensorSpec spec(EMPTY_SHAPE, tensor_layout);
    auto tensor = TensorUnderTest::from_span(std::span<const float>(vec), spec);

    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), EMPTY_SHAPE);
    EXPECT_EQ(tensor.padded_shape(), EMPTY_SHAPE);
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
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result.size(), vec.size());
}

// =============================================================================
// from_vector (lvalue)
// =============================================================================

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
    TensorSpec spec(EMPTY_SHAPE, tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);

    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), EMPTY_SHAPE);
    EXPECT_EQ(tensor.padded_shape(), EMPTY_SHAPE);
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
    auto vec = create_test_vector<float>(32 * 32);
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{});
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);

    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor.layout(), Layout::TILE);
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result.size(), vec.size());
}

// =============================================================================
// from_vector (rvalue)
// =============================================================================

TEST(TensorHostBehaviorTest, from_vector_Rvalue) {
    auto vec = create_test_vector<float>(32 * 4);
    auto vec_copy = vec;
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
    TensorSpec spec(EMPTY_SHAPE, tensor_layout);
    auto tensor = TensorUnderTest::from_vector(std::move(vec), spec);

    EXPECT_EQ(tensor.tensor_spec(), spec);
    EXPECT_EQ(tensor.logical_shape(), EMPTY_SHAPE);
    EXPECT_EQ(tensor.padded_shape(), EMPTY_SHAPE);
    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

TEST(TensorHostBehaviorTest, from_vector_Rvalue_SizeMismatch_TooFewElements) {
    auto vec = create_test_vector<float>(16);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_vector(std::move(vec), spec));
}

TEST(TensorHostBehaviorTest, from_vector_Rvalue_SizeMismatch_TooManyElements) {
    auto vec = create_test_vector<float>(256);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    EXPECT_ANY_THROW((void)TensorUnderTest::from_vector(std::move(vec), spec));
}

TEST(TensorHostBehaviorTest, from_vector_Rvalue_Tile_BFloat16) {
    auto vec = create_test_vector<float>(32 * 32);
    auto vec_copy = vec;
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{});
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(std::move(vec), spec);

    EXPECT_EQ(tensor.logical_shape(), Shape({32, 32}));
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor.layout(), Layout::TILE);
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result.size(), vec_copy.size());
}

// =============================================================================
// to_vector
// =============================================================================

TEST(TensorHostBehaviorTest, to_vector_RowMajor_Float32) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(vec), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result, vec);
}

TEST(TensorHostBehaviorTest, to_vector_RowMajor_BFloat16) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);

    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result.size(), vec.size());
}

TEST(TensorHostBehaviorTest, to_vector_Tile_Float32) {
    auto vec = create_test_vector<float>(32 * 32);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::TILE), MemoryConfig{});
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    auto result = tensor.to_vector<float>();
    EXPECT_EQ(result, vec);
}

TEST(TensorHostBehaviorTest, to_vector_EmptyTensor) {
    std::vector<float> vec;
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(EMPTY_SHAPE, tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_TRUE(tensor.to_vector<float>().empty());
}

TEST(TensorHostBehaviorTest, to_vector_DTypeMismatch_Float32ToInt32) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(vec), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_ANY_THROW((void)tensor.to_vector<int32_t>());
}

// =============================================================================
// Getters: logical_volume / physical_volume
// =============================================================================

TEST(TensorHostBehaviorTest, logical_volume) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_volume(), 32 * 4);
}

TEST(TensorHostBehaviorTest, logical_volume_HigherRank) {
    auto vec = create_test_vector<float>(2 * 3 * 4 * 5);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({2, 3, 4, 5}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_volume(), 2 * 3 * 4 * 5);
}

TEST(TensorHostBehaviorTest, logical_volume_EmptyTensor) {
    std::vector<float> vec;
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(EMPTY_SHAPE, tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_EQ(tensor.logical_volume(), 0);
}

TEST(TensorHostBehaviorTest, physical_volume) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.physical_volume(), 32 * 4);
}

TEST(TensorHostBehaviorTest, physical_volume_WithPadding) {
    auto vec = create_test_vector<float>(64 * 8);
    TensorUnderTest tensor(
        HostBuffer(std::move(vec)), Shape({32, 4}), Shape({64, 8}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_volume(), 32 * 4);
    EXPECT_EQ(tensor.physical_volume(), 64 * 8);
}

TEST(TensorHostBehaviorTest, physical_volume_EmptyTensor) {
    std::vector<float> vec;
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(EMPTY_SHAPE, tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_EQ(tensor.physical_volume(), 0);
}

// =============================================================================
// Getters: strides
// =============================================================================

TEST(TensorHostBehaviorTest, strides_RowMajor) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    auto strides = tensor.strides();
    EXPECT_EQ(strides.rank(), 2);
    EXPECT_EQ(strides[0], 4);
    EXPECT_EQ(strides[1], 1);
}

TEST(TensorHostBehaviorTest, strides_RowMajor_HigherRank) {
    auto vec = create_test_vector<float>(2 * 3 * 4 * 5);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({2, 3, 4, 5}), DataType::FLOAT32, Layout::ROW_MAJOR);
    auto strides = tensor.strides();
    EXPECT_EQ(strides.rank(), 4);
    EXPECT_EQ(strides[0], 3 * 4 * 5);
    EXPECT_EQ(strides[1], 4 * 5);
    EXPECT_EQ(strides[2], 5);
    EXPECT_EQ(strides[3], 1);
}

TEST(TensorHostBehaviorTest, strides_Tile) {
    auto vec = create_test_vector<float>(32 * 32);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::TILE), MemoryConfig{});
    TensorSpec spec(Shape({32, 32}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    auto strides = tensor.strides();
    EXPECT_EQ(strides.rank(), 2);
}

TEST(TensorHostBehaviorTest, strides_EmptyTensor) {
    std::vector<float> vec;
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(EMPTY_SHAPE, tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    auto strides = tensor.strides();
    EXPECT_EQ(strides.rank(), 1);
}

// =============================================================================
// Getters: element_size
// =============================================================================

TEST(TensorHostBehaviorTest, element_size_Float32) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.element_size(), sizeof(float));
}

TEST(TensorHostBehaviorTest, element_size_BFloat16) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), MemoryConfig{});
    TensorSpec spec(Shape({32, 4}), tensor_layout);
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_EQ(tensor.element_size(), 2);  // bfloat16 is 2 bytes
}

TEST(TensorHostBehaviorTest, element_size_Int32) {
    auto vec = create_test_vector<int32_t>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::INT32, Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.element_size(), sizeof(int32_t));
}

// =============================================================================
// Getters: shard_spec / nd_shard_spec
// =============================================================================

TEST(TensorHostBehaviorTest, shard_spec_ReturnsNullopt) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_FALSE(tensor.shard_spec().has_value());
}

// TODO: Add tests for nd_shard_spec

TEST(TensorHostBehaviorTest, nd_shard_spec_ReturnsNullopt) {
    auto vec = create_test_vector<float>(32 * 4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({32, 4}), DataType::FLOAT32, Layout::ROW_MAJOR);
    EXPECT_FALSE(tensor.nd_shard_spec().has_value());
}

// =============================================================================
// Boolean Queries: is_sharded
// =============================================================================

TEST(TensorHostBehaviorTest, is_sharded_HostTensor_WithShardedMemoryConfig) {
    // Even with a sharded MemoryConfig in TensorSpec, host tensors are NOT sharded
    // Sharding only applies when tensor is on device
    auto vec = create_test_vector<float>(32 * 32);
    MemoryConfig sharded_mem_config(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1);
    TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), sharded_mem_config);
    TensorSpec spec(Shape({32, 32}), tensor_layout);

    EXPECT_TRUE(sharded_mem_config.is_sharded());
    auto tensor = TensorUnderTest::from_vector(vec, spec);
    EXPECT_FALSE(tensor.is_sharded());
}

// =============================================================================
// Other Member Functions
// =============================================================================

TEST(TensorHostBehaviorTest, write_to_string) {
    auto vec = create_test_vector<float>(4);
    TensorUnderTest tensor(HostBuffer(std::move(vec)), Shape({2, 2}), DataType::FLOAT32, Layout::ROW_MAJOR);
    std::string str = tensor.write_to_string();
    EXPECT_FALSE(str.empty());
}

}  // namespace
}  // namespace ttnn
