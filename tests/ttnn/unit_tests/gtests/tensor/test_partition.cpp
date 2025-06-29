// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

#include <cstdint>
#include <tuple>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>

#include "ttnn/tensor/xtensor/partition.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
namespace tt {
namespace tt_metal {
class Tensor;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace {

using ::testing::SizeIs;
using ::tt::tt_metal::Tensor;
using ::ttnn::experimental::xtensor::chunk;
using ::ttnn::experimental::xtensor::chunk_ndim;
using ::ttnn::experimental::xtensor::concat;
using ::ttnn::experimental::xtensor::concat_ndim;

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
    EXPECT_TRUE(xt::allclose(concat(input, 0).expr(), xt::xarray<int>{}));
}

TEST(PartitionTest, ChunkDoesNotAccessData) {
    //  Create a read-protected memory region, and point `tt::stl::Span` to it.
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
        static void segfault_handler(int signum) { siglongjmp(jmp_env, /*val=*/1); }
    };
    struct sigaction old_action;
    struct sigaction new_action;
    memset(&new_action, 0, sizeof(new_action));
    new_action.sa_handler = Handler::segfault_handler;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;
    ASSERT_EQ(sigaction(SIGSEGV, &new_action, &old_action), 0);

    tt::stl::Span<const uint8_t> protected_span(static_cast<uint8_t*>(mapped_mem), total_size);
    auto xexpr = xt::adapt(
        protected_span.data(), total_size, xt::no_ownership(), std::vector<size_t>(shape.cbegin(), shape.cend()));

    // Verify that our set up actually works by attempting to read the protected memory region, and catching a segfault.
    bool segfault_occurred = false;
    if (sigsetjmp(jmp_env, /*savemask=*/1) == 0) {
        // `volatile` ensures the read is not optimized away.
        volatile uint8_t x = protected_span[0];
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
                experimental::xtensor::get_shape_from_xarray(chunked_xexpr), ttnn::Shape({kDim0Size, 1, page_size}));
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
            size_t chunk_idx = i * 4 + j;
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
        xt::xarray<int> tensor = {{{i * 2, i * 2 + 1}}};
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
        xt::xarray<int> tensor = {{i * 2, i * 2 + 1}};
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

}  // namespace
}  // namespace ttnn
