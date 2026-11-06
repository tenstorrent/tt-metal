// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/utils/xexception.hpp>
#include <xtensor/core/xiterator.hpp>
#include <xtensor/core/xlayout.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/containers/xstorage.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/utils/xtensor_simd.hpp>
#include <xtensor/utils/xutils.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <bit>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <tt-metalium/float8.hpp>
#include <tuple>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include <csignal>
#include <csetjmp>

#include "ttnn/tensor/xtensor/partition.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"

namespace ttnn {
namespace {

using ::testing::SizeIs;
using ttnn::experimental::xtensor::chunk;
using ttnn::experimental::xtensor::chunk_ndim;
using ttnn::experimental::xtensor::concat;
using ttnn::experimental::xtensor::concat_ndim;

TEST(PartitionTest, ChunkBasicNonDivisible3) {
    // Create a 1D tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xt::xarray<float> tensor = xt::arange<float>(10);

    // Chunk into 3 parts along dimension 0
    auto chunks = chunk(tensor, 3, 0);

    ASSERT_THAT(chunks, SizeIs(3));
    EXPECT_EQ(chunks[0].shape()[0], 4u);  // first chunk size 4
    EXPECT_EQ(chunks[1].shape()[0], 4u);  // next chunk size 4
    EXPECT_EQ(chunks[2].shape()[0], 2u);  // last chunk size 2
}

TEST(PartitionTest, ChunkBasicLessChunksThanProvided) {
    // Create a 1D tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12]
    xt::xarray<float> tensor = xt::arange<float>(13);

    // Chunk into 6 parts along dimension 0
    auto chunks = chunk(tensor, 6, 0);

    ASSERT_THAT(chunks, SizeIs(5));
    EXPECT_EQ(chunks[0].shape()[0], 3u);  // first chunk size 3
    EXPECT_EQ(chunks[1].shape()[0], 3u);  // next chunk size 3
    EXPECT_EQ(chunks[2].shape()[0], 3u);  // next chunk size 3
    EXPECT_EQ(chunks[3].shape()[0], 3u);  // next chunk size 3
    EXPECT_EQ(chunks[4].shape()[0], 1u);  // last chunk size 1
}

TEST(PartitionTest, ChunkFewerChunksThanRequested) {
    xt::xarray<float> tensor = xt::arange<float>(5);

    auto chunks = chunk(tensor, 7, 0);

    ASSERT_THAT(chunks, SizeIs(5));
    EXPECT_EQ(chunks[0].shape()[0], 1u);
    EXPECT_EQ(chunks[1].shape()[0], 1u);
    EXPECT_EQ(chunks[2].shape()[0], 1u);
    EXPECT_EQ(chunks[3].shape()[0], 1u);
    EXPECT_EQ(chunks[4].shape()[0], 1u);
}

TEST(PartitionTest, ChunkNegativeDim) {
    xt::xarray<float> tensor = xt::arange<float>(20);
    tensor.reshape({2, 10});

    // Chunk into 3 parts along dimension -1 (i.e., dimension 1)
    auto chunks_neg = chunk(tensor, 3, -1);
    auto chunks_pos = chunk(tensor, 3, 1);

    ASSERT_THAT(chunks_neg, SizeIs(3));
    ASSERT_THAT(chunks_pos, SizeIs(3));

    EXPECT_TRUE(xt::allclose(chunks_neg[0], chunks_pos[0]));
    EXPECT_TRUE(xt::allclose(chunks_neg[1], chunks_pos[1]));
    EXPECT_TRUE(xt::allclose(chunks_neg[2], chunks_pos[2]));
}

TEST(PartitionTest, DefaultAxis) {
    xt::xarray<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    xt::xarray<double> b = {{5.0, 6.0}, {7.0, 8.0}};
    std::vector<xt::xarray<double>> input = {a, b};

    auto result = concat(input);  // axis=0 by default
    xt::xarray<double> expected = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};

    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, AxisOne) {
    xt::xarray<int> x = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> y = {{7, 8, 9}, {10, 11, 12}};
    std::vector<xt::xarray<int>> input = {x, y};

    auto result = concat(input, 1);
    xt::xarray<int> expected = {{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};

    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, ConcatNegativeDim) {
    xt::xarray<int> x = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> y = {{7, 8, 9}, {10, 11, 12}};
    std::vector<xt::xarray<int>> input = {x, y};

    auto result = concat(input, -1);
    xt::xarray<int> expected = {{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};

    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, MultipleArraysAxis0) {
    xt::xarray<float> a = {1.0f, 2.0f};
    xt::xarray<float> b = {3.0f, 4.0f};
    xt::xarray<float> c = {5.0f, 6.0f};
    std::vector<xt::xarray<float>> input = {a, b, c};

    auto result = concat(input, 0);
    xt::xarray<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, EmptyArray) {
    xt::xarray<int> a = {{1, 2}, {3, 4}};
    xt::xarray<int> b;  // Empty
    std::vector<xt::xarray<int>> input = {a, b};

    EXPECT_ANY_THROW({ auto result = concat(input, 0); });
}

TEST(PartitionTest, HigherDimensions) {
    xt::xarray<int> arr1 = xt::arange<int>(1, 9);  // 1 to 8
    arr1.reshape({2, 2, 2});
    xt::xarray<int> arr2 = xt::arange<int>(9, 17);  // 9 to 16
    arr2.reshape({2, 2, 2});

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    auto result = concat(input, 0);

    // Expected: shape (4,2,2) with arr1 stacked over arr2 along axis 0
    xt::xarray<int> expected = xt::concatenate(xt::xtuple(arr1, arr2), 0);

    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, HigherAxis) {
    xt::xarray<int> arr1 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}};
    // Both have shape (2,2,2)

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    auto result = concat(input, 2);
    // Expected shape: (2,2,4)
    xt::xarray<int> expected = {{{1, 2, 9, 10}, {3, 4, 11, 12}}, {{5, 6, 13, 14}, {7, 8, 15, 16}}};

    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, UnmatchedDimensions) {
    xt::xarray<int> arr1 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}, {{17, 18}, {19, 20}}};
    // arr1 has shape (2,2,2), arr2 has shape (3,2,2)

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    EXPECT_ANY_THROW(concat(input, 0));
    EXPECT_ANY_THROW(concat(input, 1));
    EXPECT_ANY_THROW(concat(input, 2));
}

TEST(PartitionTest, UnmatchedDimensionCount) {
    xt::xarray<int> arr1 = {{1, 2}, {3, 4}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}};
    // arr1 has shape (2,2), arr2 has shape (2,2,2)

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    EXPECT_ANY_THROW(concat(input, 0));
    EXPECT_ANY_THROW(concat(input, 1));
    EXPECT_ANY_THROW(concat(input, 2));
}

TEST(PartitionTest, DimensionOutofRange) {
    xt::xarray<int> arr1 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}};

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    EXPECT_ANY_THROW(concat(input, 6));   // Dimension out of range
    EXPECT_ANY_THROW(concat(input, -6));  // Negative dimension out of range
}

TEST(PartitionTest, EmptyInput) {
    std::vector<xt::xarray<int>> input;
    EXPECT_NO_THROW(concat(input, 0));
    EXPECT_TRUE(concat(input, 0).data().empty());
}

TEST(PartitionTest, ChunkDoesNotAccessData) {
    //  Create a read-protected memory region, and point `ttsl::Span` to it.
    //  `chunk` should not access the data, and should only calculate offsets and shapes.
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_NE(page_size, -1);

    constexpr int kDim0Size = 10;
    constexpr int kDim1Size = 17;
    const ttnn::Shape shape({kDim0Size, kDim1Size, page_size});
    const size_t total_size = shape.volume();

    // With `PROT_NONE`, the mapped memory cannot be accessed.
    void* mapped_mem = mmap(
        /*addr=*/nullptr, total_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
    ASSERT_NE(mapped_mem, MAP_FAILED);

    // Set up signal handler to verify segmentation faults.
    static sigjmp_buf jmp_env;
    struct Handler {
        static void segfault_handler(int /*signum*/) { siglongjmp(jmp_env, /*val=*/1); }
    };
    struct sigaction old_action{};
    struct sigaction new_action{};
    memset(&new_action, 0, sizeof(new_action));
    new_action.sa_handler = Handler::segfault_handler;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;
    ASSERT_EQ(sigaction(SIGSEGV, &new_action, &old_action), 0);

    ttsl::Span<const uint8_t> protected_span(static_cast<uint8_t*>(mapped_mem), total_size);
    auto xexpr = xt::adapt(
        protected_span.data(), total_size, xt::no_ownership(), std::vector<size_t>(shape.cbegin(), shape.cend()));

    // Verify that our set up actually works by attempting to read the protected memory region, and catching a segfault.
    bool segfault_occurred = false;
    if (sigsetjmp(jmp_env, /*savemask=*/1) == 0) {
        // `volatile` ensures the read is not optimized away.
        [[maybe_unused]] volatile uint8_t x = protected_span[0];
    } else {
        segfault_occurred = true;
    }
    EXPECT_TRUE(segfault_occurred);

    if (sigsetjmp(jmp_env, /*savemask=*/1) == 0) {
        auto chunks = chunk(xexpr, /*num_chunks=*/kDim1Size, /*dim=*/1);

        EXPECT_THAT(chunks, SizeIs(kDim1Size));
        for (const auto& chunked_xexpr : chunks) {
            EXPECT_THAT(chunked_xexpr, SizeIs(kDim0Size * page_size));
            EXPECT_EQ(
                ttnn::experimental::xtensor::get_shape_from_xarray(chunked_xexpr),
                ttnn::Shape({kDim0Size, 1, page_size}));
        }
    } else {
        FAIL() << "segfault occurred when calling `chunk`";
    }

    // Cleanup.
    ASSERT_EQ(sigaction(SIGSEGV, &old_action, nullptr), 0);
    ASSERT_EQ(munmap(mapped_mem, total_size), 0);
}

TEST(PartitionTest, ChunkNdimEmpty) {
    xt::xarray<float> tensor = xt::arange<float>(24);
    tensor.reshape({2, 3, 4});

    auto chunks = chunk_ndim(tensor, {}, {});

    ASSERT_THAT(chunks, SizeIs(1));
    EXPECT_TRUE(xt::allclose(chunks[0], tensor));
}

TEST(PartitionTest, ChunkNdimMismatchedSizes) {
    xt::xarray<float> tensor = xt::arange<float>(24);
    tensor.reshape({2, 3, 4});

    EXPECT_ANY_THROW(chunk_ndim(tensor, {2, 3}, {0}));
    EXPECT_ANY_THROW(chunk_ndim(tensor, {2}, {0, 1}));
}

TEST(PartitionTest, ChunkNdimNegativeChunks) {
    xt::xarray<float> tensor = xt::arange<float>(24);
    tensor.reshape({2, 3, 4});

    EXPECT_ANY_THROW(chunk_ndim(tensor, {-1, 2}, {0, 1}));
    EXPECT_ANY_THROW(chunk_ndim(tensor, {2, 0}, {0, 1}));
}

TEST(PartitionTest, ChunkNdimNonUniqueDims) {
    xt::xarray<float> tensor = xt::arange<float>(24);
    tensor.reshape({2, 3, 4});

    EXPECT_ANY_THROW(chunk_ndim(tensor, {2, 2}, {0, 0}));
    EXPECT_ANY_THROW(chunk_ndim(tensor, {2, 2}, {2, -1}));
    EXPECT_ANY_THROW(chunk_ndim(tensor, {2, 2, 3}, {0, 1, 0}));
}

TEST(PartitionTest, ChunkNdimDimsOutOfRange) {
    xt::xarray<float> tensor = xt::arange<float>(24);
    tensor.reshape({2, 3, 4});

    EXPECT_ANY_THROW(chunk_ndim(tensor, {2}, {3}));
    EXPECT_ANY_THROW(chunk_ndim(tensor, {2}, {-5}));
    EXPECT_ANY_THROW(chunk_ndim(tensor, {2, 2}, {0, 5}));
}

TEST(PartitionTest, ChunkNdimRowMajorOrder) {
    xt::xarray<float> tensor = xt::arange<float>(24);
    tensor.reshape({2, 3, 4});

    auto chunks = chunk_ndim(tensor, {2, 2}, {0, 1});

    ASSERT_THAT(chunks, SizeIs(4));

    // Expected order: [0,0], [0,1], [1,0], [1,1] in row-major
    // [0,0]: first half of dim 0, first half of dim 1
    EXPECT_TRUE(xt::allclose(chunks[0], xt::view(tensor, xt::range(0, 1), xt::range(0, 2), xt::all())));
    // [0,1]: first half of dim 0, second half of dim 1
    EXPECT_TRUE(xt::allclose(chunks[1], xt::view(tensor, xt::range(0, 1), xt::range(2, 3), xt::all())));
    // [1,0]: second half of dim 0, first half of dim 1
    EXPECT_TRUE(xt::allclose(chunks[2], xt::view(tensor, xt::range(1, 2), xt::range(0, 2), xt::all())));
    // [1,1]: second half of dim 0, second half of dim 1
    EXPECT_TRUE(xt::allclose(chunks[3], xt::view(tensor, xt::range(1, 2), xt::range(2, 3), xt::all())));
}

TEST(PartitionTest, ChunkNdimFewerChunksThanRequested) {
    xt::xarray<float> tensor = xt::arange<float>(12);
    tensor.reshape({3, 4});

    // Request 5 chunks along dim 0 (size 3), 6 chunks along dim 1 (size 4)
    auto chunks = chunk_ndim(tensor, {5, 6}, {0, 1});

    // Should get 3*4 = 12 chunks (capped by actual dimension sizes)
    ASSERT_THAT(chunks, SizeIs(12));

    // Each chunk should be 1x1
    for (const auto& chunk : chunks) {
        EXPECT_EQ(chunk.shape()[0], 1u);
        EXPECT_EQ(chunk.shape()[1], 1u);
    }

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            size_t chunk_idx = (i * 4) + j;
            EXPECT_FLOAT_EQ(chunks[chunk_idx](0, 0), tensor(i, j));
        }
    }
}

TEST(PartitionTest, ChunkNdimBasicMultiDim) {
    xt::xarray<int> tensor = xt::arange<int>(60);
    tensor.reshape({3, 4, 5});

    auto chunks = chunk_ndim(tensor, {2, 2}, {0, 2});

    ASSERT_THAT(chunks, SizeIs(4));

    // Check shapes: dim 0 split into [2,1], dim 2 split into [3,2]
    EXPECT_EQ(chunks[0].shape()[0], 2u);  // [0,0]
    EXPECT_EQ(chunks[0].shape()[2], 3u);

    EXPECT_EQ(chunks[1].shape()[0], 2u);  // [0,1]
    EXPECT_EQ(chunks[1].shape()[2], 2u);

    EXPECT_EQ(chunks[2].shape()[0], 1u);  // [1,0]
    EXPECT_EQ(chunks[2].shape()[2], 3u);

    EXPECT_EQ(chunks[3].shape()[0], 1u);  // [1,1]
    EXPECT_EQ(chunks[3].shape()[2], 2u);

    EXPECT_TRUE(xt::allclose(chunks[0], xt::view(tensor, xt::range(0, 2), xt::all(), xt::range(0, 3))));
    EXPECT_TRUE(xt::allclose(chunks[1], xt::view(tensor, xt::range(0, 2), xt::all(), xt::range(3, 5))));
    EXPECT_TRUE(xt::allclose(chunks[2], xt::view(tensor, xt::range(2, 3), xt::all(), xt::range(0, 3))));
    EXPECT_TRUE(xt::allclose(chunks[3], xt::view(tensor, xt::range(2, 3), xt::all(), xt::range(3, 5))));
}

TEST(PartitionTest, ChunkNdimSingleDimension) {
    xt::xarray<float> tensor = xt::arange<float>(10);

    auto chunks = chunk_ndim(tensor, {3}, {0});

    ASSERT_THAT(chunks, SizeIs(3));
    EXPECT_EQ(chunks[0].shape()[0], 4u);
    EXPECT_EQ(chunks[1].shape()[0], 4u);
    EXPECT_EQ(chunks[2].shape()[0], 2u);

    EXPECT_TRUE(xt::allclose(chunks[0], xt::view(tensor, xt::range(0, 4))));
    EXPECT_TRUE(xt::allclose(chunks[1], xt::view(tensor, xt::range(4, 8))));
    EXPECT_TRUE(xt::allclose(chunks[2], xt::view(tensor, xt::range(8, 10))));
}

TEST(PartitionTest, ChunkNdimThreeDimensions) {
    xt::xarray<double> tensor = xt::arange<double>(24);
    tensor.reshape({2, 3, 4});

    auto chunks = chunk_ndim(tensor, {2, 2, 2}, {0, 1, 2});

    ASSERT_THAT(chunks, SizeIs(8));

    // In row-major order with dims {0, 1, 2}, chunks are ordered as:
    // [0]: (0,0,0), [1]: (0,0,1), [2]: (0,1,0), [3]: (0,1,1)
    // [4]: (1,0,0), [5]: (1,0,1), [6]: (1,1,0), [7]: (1,1,1)

    // Verify dimension 0 chunks
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(chunks[i].shape()[0], 1u);      // first half of dim 0
        EXPECT_EQ(chunks[i + 4].shape()[0], 1u);  // second half of dim 0
    }

    // Verify dimension 1 chunks - every 2 chunks alternate between dim1=0 and dim1=1
    for (size_t i = 0; i < 8; i += 4) {
        EXPECT_EQ(chunks[i].shape()[1], 2u);      // (x,0,0) - first half of dim 1
        EXPECT_EQ(chunks[i + 1].shape()[1], 2u);  // (x,0,1) - first half of dim 1
        EXPECT_EQ(chunks[i + 2].shape()[1], 1u);  // (x,1,0) - second half of dim 1
        EXPECT_EQ(chunks[i + 3].shape()[1], 1u);  // (x,1,1) - second half of dim 1
    }

    // Verify dimension 2 chunks - alternates every chunk
    for (size_t i = 0; i < 8; i += 2) {
        EXPECT_EQ(chunks[i].shape()[2], 2u);      // (x,x,0) - first half of dim 2
        EXPECT_EQ(chunks[i + 1].shape()[2], 2u);  // (x,x,1) - second half of dim 2
    }

    EXPECT_TRUE(xt::allclose(chunks[0], xt::view(tensor, xt::range(0, 1), xt::range(0, 2), xt::range(0, 2))));
    EXPECT_TRUE(xt::allclose(chunks[1], xt::view(tensor, xt::range(0, 1), xt::range(0, 2), xt::range(2, 4))));
    EXPECT_TRUE(xt::allclose(chunks[2], xt::view(tensor, xt::range(0, 1), xt::range(2, 3), xt::range(0, 2))));
    EXPECT_TRUE(xt::allclose(chunks[3], xt::view(tensor, xt::range(0, 1), xt::range(2, 3), xt::range(2, 4))));
    EXPECT_TRUE(xt::allclose(chunks[4], xt::view(tensor, xt::range(1, 2), xt::range(0, 2), xt::range(0, 2))));
    EXPECT_TRUE(xt::allclose(chunks[5], xt::view(tensor, xt::range(1, 2), xt::range(0, 2), xt::range(2, 4))));
    EXPECT_TRUE(xt::allclose(chunks[6], xt::view(tensor, xt::range(1, 2), xt::range(2, 3), xt::range(0, 2))));
    EXPECT_TRUE(xt::allclose(chunks[7], xt::view(tensor, xt::range(1, 2), xt::range(2, 3), xt::range(2, 4))));
}

TEST(PartitionTest, ChunkNdimNonContiguousDims) {
    xt::xarray<int> tensor = xt::arange<int>(120);
    tensor.reshape({2, 3, 4, 5});

    auto chunks = chunk_ndim(tensor, {2, 2}, {0, 2});

    ASSERT_THAT(chunks, SizeIs(4));

    // Verify dimensions 1 and 3 remain unchanged
    for (const auto& chunk : chunks) {
        EXPECT_EQ(chunk.shape()[1], 3u);
        EXPECT_EQ(chunk.shape()[3], 5u);
    }

    EXPECT_TRUE(xt::allclose(chunks[0], xt::view(tensor, xt::range(0, 1), xt::all(), xt::range(0, 2), xt::all())));
    EXPECT_TRUE(xt::allclose(chunks[1], xt::view(tensor, xt::range(0, 1), xt::all(), xt::range(2, 4), xt::all())));
    EXPECT_TRUE(xt::allclose(chunks[2], xt::view(tensor, xt::range(1, 2), xt::all(), xt::range(0, 2), xt::all())));
    EXPECT_TRUE(xt::allclose(chunks[3], xt::view(tensor, xt::range(1, 2), xt::all(), xt::range(2, 4), xt::all())));
}

TEST(PartitionTest, ConcatNdimBasic) {
    // Create 4 2x3 arrays to concatenate in a 2x2 grid
    xt::xarray<float> a = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    xt::xarray<float> b = {{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}};
    xt::xarray<float> c = {{13.0f, 14.0f, 15.0f}, {16.0f, 17.0f, 18.0f}};
    xt::xarray<float> d = {{19.0f, 20.0f, 21.0f}, {22.0f, 23.0f, 24.0f}};

    std::vector<xt::xarray<float>> expressions = {a, b, c, d};

    // Concatenate along dimensions [0, 1] with 2 chunks each
    auto result = concat_ndim(expressions, {2, 2}, {0, 1});

    // Expected shape: (4, 6)
    EXPECT_EQ(result.expr().shape()[0], 4u);
    EXPECT_EQ(result.expr().shape()[1], 6u);

    // Verify content - should be arranged as:
    // a b
    // c d
    xt::xarray<float> expected = {
        {1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f},
        {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f},
        {13.0f, 14.0f, 15.0f, 19.0f, 20.0f, 21.0f},
        {16.0f, 17.0f, 18.0f, 22.0f, 23.0f, 24.0f}};
    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, ConcatNdimEmpty) {
    std::vector<xt::xarray<int>> expressions;
    auto result = concat_ndim(expressions, {}, {});
    EXPECT_EQ(result.expr().size(), 0u);
}

TEST(PartitionTest, ConcatNdimSingleExpression) {
    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    std::vector<xt::xarray<int>> expressions = {a};

    auto result = concat_ndim(expressions, {}, {});
    EXPECT_TRUE(xt::allclose(result.expr(), a));
}

TEST(PartitionTest, ConcatNdimMismatchedSizes) {
    xt::xarray<float> a = {1.0f, 2.0f};
    std::vector<xt::xarray<float>> expressions = {a};

    // Mismatched num_chunks and dims sizes
    EXPECT_ANY_THROW(concat_ndim(expressions, {2, 2}, {0}));
    EXPECT_ANY_THROW(concat_ndim(expressions, {2}, {0, 1}));
}

TEST(PartitionTest, ConcatNdimInvalidNumChunks) {
    xt::xarray<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<xt::xarray<double>> expressions = {a};

    // Zero or negative num_chunks
    EXPECT_ANY_THROW(concat_ndim(expressions, {0}, {0}));
    EXPECT_ANY_THROW(concat_ndim(expressions, {-1}, {0}));
    EXPECT_ANY_THROW(concat_ndim(expressions, {2, 0}, {0, 1}));
}

TEST(PartitionTest, ConcatNdimWrongNumberOfExpressions) {
    xt::xarray<int> a = {1, 2, 3};
    xt::xarray<int> b = {4, 5, 6};
    std::vector<xt::xarray<int>> expressions = {a, b};

    // Expected 4 expressions (2*2) but only have 2
    EXPECT_ANY_THROW(concat_ndim(expressions, {2, 2}, {0, 1}));
}

TEST(PartitionTest, ConcatNdimMismatchedShapes) {
    xt::xarray<float> a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    xt::xarray<float> b = {5.0f, 6.0f, 7.0f};  // Different shape
    std::vector<xt::xarray<float>> expressions = {a, b};

    EXPECT_ANY_THROW(concat_ndim(expressions, {2}, {0}));
}

TEST(PartitionTest, ConcatNdimNegativeDims) {
    xt::xarray<int> a = {{1, 2}};
    xt::xarray<int> b = {{3, 4}};
    xt::xarray<int> c = {{5, 6}};
    xt::xarray<int> d = {{7, 8}};
    std::vector<xt::xarray<int>> expressions = {a, b, c, d};

    // Use negative dimensions
    auto result = concat_ndim(expressions, {2, 2}, {-2, -1});

    xt::xarray<int> expected = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, ConcatNdimNonUniqueDims) {
    xt::xarray<int> a = {1, 2, 3};
    std::vector<xt::xarray<int>> expressions = {a};

    // Duplicate dimensions
    EXPECT_ANY_THROW(concat_ndim(expressions, {2, 2}, {0, 0}));
    EXPECT_ANY_THROW(concat_ndim(expressions, {2, 2}, {1, -1}));  // Both refer to last dim
}

TEST(PartitionTest, ConcatNdimDimsOutOfRange) {
    xt::xarray<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<xt::xarray<double>> expressions = {a};

    EXPECT_ANY_THROW(concat_ndim(expressions, {2}, {3}));
    EXPECT_ANY_THROW(concat_ndim(expressions, {2}, {-3}));
}

TEST(PartitionTest, ConcatNdimSingleDimension) {
    // Should behave like regular concat
    xt::xarray<int> a = {1, 2, 3};
    xt::xarray<int> b = {4, 5, 6};
    xt::xarray<int> c = {7, 8, 9};
    std::vector<xt::xarray<int>> expressions = {a, b, c};

    auto result_ndim = concat_ndim(expressions, {3}, {0});
    auto result_regular = concat(expressions, 0);

    EXPECT_TRUE(xt::allclose(result_ndim.expr(), result_regular.expr()));
}

TEST(PartitionTest, ConcatNdimThreeDimensions) {
    // Create 8 small tensors of shape (1, 1, 2)
    std::vector<xt::xarray<int>> expressions;
    for (int i = 0; i < 8; ++i) {
        xt::xarray<int> tensor = {{{i * 2, (i * 2) + 1}}};
        expressions.push_back(tensor);
    }

    // Concatenate along all 3 dimensions with 2 chunks each
    auto result = concat_ndim(expressions, {2, 2, 2}, {0, 1, 2});

    // Expected shape: (2, 2, 4)
    EXPECT_EQ(result.expr().shape()[0], 2u);
    EXPECT_EQ(result.expr().shape()[1], 2u);
    EXPECT_EQ(result.expr().shape()[2], 4u);

    // Verify row-major ordering
    xt::xarray<int> expected = {{{0, 1, 2, 3}, {4, 5, 6, 7}}, {{8, 9, 10, 11}, {12, 13, 14, 15}}};
    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

TEST(PartitionTest, ConcatNdimNonContiguousDims) {
    // Create 4 tensors of shape (1, 2, 1, 3)
    xt::xarray<float> a = {{{{1.0f, 2.0f, 3.0f}}, {{4.0f, 5.0f, 6.0f}}}};
    xt::xarray<float> b = {{{{7.0f, 8.0f, 9.0f}}, {{10.0f, 11.0f, 12.0f}}}};
    xt::xarray<float> c = {{{{13.0f, 14.0f, 15.0f}}, {{16.0f, 17.0f, 18.0f}}}};
    xt::xarray<float> d = {{{{19.0f, 20.0f, 21.0f}}, {{22.0f, 23.0f, 24.0f}}}};

    std::vector<xt::xarray<float>> expressions = {a, b, c, d};

    // Concatenate along dimensions 0 and 2 (skipping 1 and 3)
    auto result = concat_ndim(expressions, {2, 2}, {0, 2});

    // Expected shape: (2, 2, 2, 3)
    EXPECT_EQ(result.expr().shape()[0], 2u);
    EXPECT_EQ(result.expr().shape()[1], 2u);
    EXPECT_EQ(result.expr().shape()[2], 2u);
    EXPECT_EQ(result.expr().shape()[3], 3u);

    // Verify dimensions 1 and 3 remain unchanged
    EXPECT_TRUE(xt::allclose(xt::view(result.expr(), xt::range(0, 1), xt::all(), xt::range(0, 1), xt::all()), a));
    EXPECT_TRUE(xt::allclose(xt::view(result.expr(), xt::range(0, 1), xt::all(), xt::range(1, 2), xt::all()), b));
    EXPECT_TRUE(xt::allclose(xt::view(result.expr(), xt::range(1, 2), xt::all(), xt::range(0, 1), xt::all()), c));
    EXPECT_TRUE(xt::allclose(xt::view(result.expr(), xt::range(1, 2), xt::all(), xt::range(1, 2), xt::all()), d));
}

TEST(PartitionTest, ConcatNdimInverseOfChunkNdim) {
    // Create a tensor and chunk it
    xt::xarray<double> original = xt::arange<double>(120);
    original.reshape({4, 5, 6});

    // Chunk along dimensions 0 and 2
    auto chunks = chunk_ndim(original, {2, 3}, {0, 2});

    // Convert to vector of xarrays for concat_ndim
    std::vector<xt::xarray<double>> expressions;
    for (const auto& chunk : chunks) {
        expressions.push_back(xt::xarray<double>(chunk));
    }

    // Concatenate back
    auto reconstructed = concat_ndim(expressions, {2, 3}, {0, 2});

    // Should get back the original
    EXPECT_TRUE(xt::allclose(reconstructed.expr(), original));
}

TEST(PartitionTest, ConcatNdimRowMajorOrder) {
    // Create 6 small tensors of shape (1, 2)
    std::vector<xt::xarray<int>> expressions;
    for (int i = 0; i < 6; ++i) {
        xt::xarray<int> tensor = {{i * 2, (i * 2) + 1}};
        expressions.push_back(tensor);
    }

    // Concatenate with 2 chunks along dim 0, 3 chunks along dim 1
    auto result = concat_ndim(expressions, {2, 3}, {0, 1});

    // Expected row-major order:
    // 0, 1, 2, 3, 4, 5
    // 6, 7, 8, 9, 10, 11
    xt::xarray<int> expected = {{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}};
    EXPECT_TRUE(xt::allclose(result.expr(), expected));
}

template <typename T>
T concat_payload(size_t index) {
    static_assert(std::is_trivially_copyable_v<T>);
    if constexpr (sizeof(T) == 1) {
        return std::bit_cast<T>(static_cast<uint8_t>(index));
    } else if constexpr (sizeof(T) == 2) {
        constexpr uint16_t bits[] = {
            0x0000,
            0x8000,
            0x7f80,
            0xff80,
            0x7fc1,
            0x7f81,
            0xffd5,
            0x0001,
            0x8001,
            0x3f80,
            0xbf80,
            0xffff,
            0x7fff,
            0x0080,
            0x8080,
            0x1234};
        return std::bit_cast<T>(bits[index % std::size(bits)]);
    } else if constexpr (sizeof(T) == 4) {
        constexpr uint32_t bits[] = {
            0x00000000,
            0x80000000,
            0x7f800000,
            0xff800000,
            0x7fc01234,
            0x7f800001,
            0xffc055aa,
            0x00000001,
            0x80000001,
            0x3f800000,
            0xbf800000,
            0xffffffff,
            0x7fffffff,
            0x00800000,
            0x80800000,
            0x12345678};
        return std::bit_cast<T>(bits[index % std::size(bits)]);
    } else {
        static_assert(sizeof(T) == 8);
        constexpr uint64_t bits[] = {
            0x0000000000000000,
            0x8000000000000000,
            0x7ff0000000000000,
            0xfff0000000000000,
            0x7ff8000000001234,
            0x7ff0000000000001,
            0xfff80000000055aa,
            0x0000000000000001,
            0x8000000000000001,
            0x3ff0000000000000,
            0xbff0000000000000,
            0xffffffffffffffff,
            0x7fffffffffffffff,
            0x0010000000000000,
            0x8010000000000000,
            0x123456789abcdef0};
        return std::bit_cast<T>(bits[index % std::size(bits)]);
    }
}

// Compute expected values from logical coordinates, independently of the copy blocks.
template <typename T>
void check_concat_payloads(
    const std::vector<size_t>& shape, const ttsl::SmallVector<int>& counts, const ttsl::SmallVector<int>& axes) {
    using namespace ttnn::experimental::xtensor;
    using PayloadBytes = std::array<std::byte, sizeof(T)>;
    const size_t volume = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    const size_t pieces = std::accumulate(counts.begin(), counts.end(), size_t{1}, std::multiplies<size_t>());
    std::vector<size_t> output_shape = shape;
    std::vector<size_t> normalized_axes;
    for (size_t i = 0; i < axes.size(); ++i) {
        const size_t axis = axes[i] < 0 ? axes[i] + static_cast<int>(shape.size()) : axes[i];
        normalized_axes.push_back(axis);
        output_shape[axis] *= counts[i];
    }

    // Every adapter points into a guarded allocation, not its beginning. Both
    // mutable and const pointers must refer to the logical start of the span.
    constexpr size_t guard = 5;
    std::vector<std::vector<T>> buffers(pieces, std::vector<T>(volume + 2 * guard));
    std::vector<xt::xarray<T>> arrays;
    std::vector<AdaptedView<T>> mutable_views;
    std::vector<AdaptedView<const T>> const_views;
    for (size_t piece = 0; piece < pieces; ++piece) {
        for (size_t i = 0; i < buffers[piece].size(); ++i) {
            buffers[piece][i] = concat_payload<T>(piece * 7 + i);
        }
        auto array = xt::xarray<T>::from_shape(shape);
        if (volume != 0) {
            std::memcpy(array.data(), buffers[piece].data() + guard, volume * sizeof(T));
        }
        arrays.push_back(std::move(array));
        mutable_views.push_back(adapt(ttsl::Span<T>(buffers[piece].data() + guard, volume), shape));
        const_views.push_back(adapt(ttsl::Span<const T>(buffers[piece].data() + guard, volume), shape));
    }
    const auto saved_buffers = buffers;
    const size_t output_volume =
        std::accumulate(output_shape.begin(), output_shape.end(), size_t{1}, std::multiplies<size_t>());

    const auto verify = [&](const auto& output) {
        EXPECT_EQ(std::vector<size_t>(output.expr().shape().begin(), output.expr().shape().end()), output_shape);
        ASSERT_EQ(output.data().size(), output_volume);
        for (size_t linear = 0; linear < output_volume; ++linear) {
            std::vector<size_t> coords(shape.size());
            size_t remaining = linear;
            for (size_t d = shape.size(); d-- > 0;) {
                coords[d] = remaining % output_shape[d];
                remaining /= output_shape[d];
            }
            size_t piece = 0;
            for (size_t i = 0; i < axes.size(); ++i) {
                const size_t axis = normalized_axes[i];
                piece = piece * counts[i] + coords[axis] / shape[axis];
                coords[axis] %= shape[axis];
            }
            size_t source = 0;
            for (size_t d = 0; d < shape.size(); ++d) {
                source = source * shape[d] + coords[d];
            }
            // Compare representations so NaN payloads and signed zero must survive unchanged.
            ASSERT_EQ(
                std::bit_cast<PayloadBytes>(output.data()[linear]),
                std::bit_cast<PayloadBytes>(saved_buffers[piece][guard + source]))
                << "linear=" << linear << " piece=" << piece << " source=" << source;
        }
    };
    const auto check = [&](const auto& expressions) {
        verify(concat_ndim(expressions, counts, axes));
        if (axes.size() == 1) {
            verify(concat(expressions, axes.front()));
        }
    };
    check(arrays);
    check(mutable_views);
    check(const_views);
    for (size_t piece = 0; piece < pieces; ++piece) {
        EXPECT_EQ(
            std::memcmp(buffers[piece].data(), saved_buffers[piece].data(), buffers[piece].size() * sizeof(T)), 0);
        if (volume != 0) {
            EXPECT_EQ(std::memcmp(arrays[piece].data(), saved_buffers[piece].data() + guard, volume * sizeof(T)), 0);
        }
    }
}

template <typename T>
class PartitionPayloadTest : public ::testing::Test {};
using PartitionPayloadTypes =
    ::testing::Types<bfloat16, float, double, int32_t, int8_t, uint8_t, uint16_t, uint32_t, float8_e4m3>;
TYPED_TEST_SUITE(PartitionPayloadTest, PartitionPayloadTypes);

TYPED_TEST(PartitionPayloadTest, SingleAxisCoordinateAndByteOracle) {
    for (int axis : {0, 1, 2, -1, -2, -3}) {
        for (int pieces : {1, 2, 3, 4}) {
            SCOPED_TRACE(::testing::Message() << "axis=" << axis << " pieces=" << pieces);
            check_concat_payloads<TypeParam>({2, 3, 4}, {pieces}, {axis});
        }
    }
    check_concat_payloads<TypeParam>({2, 1, 3, 5}, {3}, {2});
    check_concat_payloads<TypeParam>({4, 1, 2, 3}, {3}, {1});
    check_concat_payloads<TypeParam>({1, 1, 17}, {4}, {-1});
    // Includes every possible 8-bit payload and all wider payload patterns.
    check_concat_payloads<TypeParam>({2, 256}, {3}, {-1});
}

TYPED_TEST(PartitionPayloadTest, MultiAxisFallbackByteOracle) {
    check_concat_payloads<TypeParam>({2, 3, 4}, {2, 3}, {2, 0});
    check_concat_payloads<TypeParam>({2, 3, 4}, {3, 2}, {-3, -1});
    check_concat_payloads<TypeParam>({2, 3, 4}, {2, 2, 2}, {0, 1, 2});
}

TYPED_TEST(PartitionPayloadTest, ZeroVolumeAndEmptyInputs) {
    for (const auto& shape : std::vector<std::vector<size_t>>{{0, 3, 4}, {2, 0, 4}, {2, 3, 0}}) {
        for (int axis : {0, 1, 2, -1}) {
            check_concat_payloads<TypeParam>(shape, {3}, {axis});
        }
        check_concat_payloads<TypeParam>(shape, {2, 3}, {0, 2});
    }
    const auto check_empty = [](const auto& inputs) {
        const auto result = concat(inputs, -1);
        EXPECT_TRUE(result.data().empty());
        EXPECT_EQ(
            std::vector<size_t>(result.expr().shape().begin(), result.expr().shape().end()), std::vector<size_t>{0});
    };
    check_empty(std::vector<xt::xarray<TypeParam>>{});
    check_empty(std::vector<ttnn::experimental::xtensor::AdaptedView<TypeParam>>{});
    check_empty(std::vector<ttnn::experimental::xtensor::AdaptedView<const TypeParam>>{});
}

TYPED_TEST(PartitionPayloadTest, NoAxesIdentity) {
    check_concat_payloads<TypeParam>({2, 3, 4}, {}, {});
    check_concat_payloads<TypeParam>({}, {}, {});
}

TEST(PartitionTest, SupportedAdapterLayoutAndOffsetContract) {
    using namespace ttnn::experimental::xtensor;
    std::vector<int32_t> values(24);
    auto adapted = adapt(ttsl::Span<int32_t>(values.data() + 4, 12), {3, 4});
    static_assert(decltype(adapted)::static_layout == xt::layout_type::row_major);
    EXPECT_EQ(adapted.data(), values.data() + 4);
    EXPECT_EQ(adapted.data_offset(), 0u);
    EXPECT_TRUE(adapted.is_contiguous());
    const std::vector<size_t> shape{3, 4};
    const typename decltype(adapted)::strides_type strides{1, 3};
    EXPECT_THROW(adapted.resize(shape, strides), std::runtime_error);
    auto offset_view = xt::strided_view(adapted, {xt::range(1, 3), xt::all()});
    auto strided_view = xt::strided_view(adapted, {xt::all(), xt::range(0, 4, 2)});
    static_assert(!std::is_same_v<decltype(offset_view), AdaptedView<int32_t>>);
    static_assert(!std::is_same_v<decltype(strided_view), AdaptedView<int32_t>>);
    EXPECT_NE(offset_view.data_offset(), 0u);
    EXPECT_FALSE(strided_view.is_contiguous());
    // These xstrided_view types have no concat instantiation in partition.cpp.
}

}  // namespace
}  // namespace ttnn
