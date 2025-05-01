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
using ::ttnn::experimental::xtensor::concat;

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

TEST(PartitionTest, DefaultAxis) {
    xt::xarray<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    xt::xarray<double> b = {{5.0, 6.0}, {7.0, 8.0}};
    std::vector<xt::xarray<double>> input = {a, b};

    xt::xarray<double> result = concat(input);  // axis=0 by default
    xt::xarray<double> expected = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};

    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST(PartitionTest, AxisOne) {
    xt::xarray<int> x = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> y = {{7, 8}, {9, 10}};
    std::vector<xt::xarray<int>> input = {x, y};

    xt::xarray<int> result = concat(input, 1);
    xt::xarray<int> expected = {{1, 2, 3, 7, 8}, {4, 5, 6, 9, 10}};

    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST(PartitionTest, MultipleArraysAxis0) {
    xt::xarray<float> a = {1.0f, 2.0f};
    xt::xarray<float> b = {3.0f, 4.0f};
    xt::xarray<float> c = {5.0f, 6.0f};
    std::vector<xt::xarray<float>> input = {a, b, c};

    xt::xarray<float> result = concat(input, 0);
    xt::xarray<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST(PartitionTest, EmptyArray) {
    xt::xarray<int> a = {{1, 2}, {3, 4}};
    xt::xarray<int> b;  // Empty
    std::vector<xt::xarray<int>> input = {a, b};

    EXPECT_ANY_THROW({ xt::xarray<int> result = concat(input, 0); });
}

TEST(PartitionTest, HigherDimensions) {
    xt::xarray<int> arr1 = xt::arange<int>(1, 9);  // 1 to 8
    arr1.reshape({2, 2, 2});
    xt::xarray<int> arr2 = xt::arange<int>(9, 17);  // 9 to 16
    arr2.reshape({2, 2, 2});

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    xt::xarray<int> result = concat(input, 0);

    // Expected: shape (4,2,2) with arr1 stacked over arr2 along axis 0
    xt::xarray<int> expected = xt::concatenate(xt::xtuple(arr1, arr2), 0);

    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST(PartitionTest, HigherAxis) {
    xt::xarray<int> arr1 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}};
    // Both have shape (2,2,2)

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    xt::xarray<int> result = concat(input, 2);
    // Expected shape: (2,2,4)
    xt::xarray<int> expected = {{{1, 2, 9, 10}, {3, 4, 11, 12}}, {{5, 6, 13, 14}, {7, 8, 15, 16}}};

    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST(PartitionTest, UnmatchedDimensions) {
    xt::xarray<int> arr1 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}, {{17, 18}, {19, 20}}};
    // arr1 has shape (2,2,2), arr2 has shape (3,2,2)
    xt::xarray<int> result_dim0 = xt::concatenate(xt::xtuple(arr1, arr2), 0);

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    EXPECT_NO_THROW(concat(input, 0));
    EXPECT_TRUE(xt::allclose(concat(input, 0), result_dim0));

    EXPECT_ANY_THROW(concat(input, 1));  // Should throw for axis 1
    EXPECT_ANY_THROW(concat(input, 2));  // Should throw for axis 2
}

TEST(PartitionTest, UnmatchedDimensionCount) {
    xt::xarray<int> arr1 = {{1, 2}, {3, 4}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}};
    // arr1 has shape (2,2), arr2 has shape (2,2,2)

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    EXPECT_ANY_THROW(concat(input, 0));  // Should throw for axis 0
    EXPECT_ANY_THROW(concat(input, 1));  // Should throw for axis 1
    EXPECT_ANY_THROW(concat(input, 2));  // Should throw for axis 2
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
    EXPECT_TRUE(xt::allclose(concat(input, 0), xt::xarray<int>{}));
}

TEST(PartitionTest, ShardSpans) {
    constexpr size_t kNumChunks = 64;
    constexpr size_t kChunkSize = 4 << 10;
    std::vector<float> test_data;
    for (int i = 0; i < kNumChunks * kChunkSize; i++) {
        test_data.push_back(i);
    }

    auto chunks = chunk(tt::stl::Span(test_data), ttnn::Shape{kNumChunks, kChunkSize}, kNumChunks);

    EXPECT_THAT(chunks, SizeIs(kNumChunks));
    for (int i = 0; i < kNumChunks; i++) {
        const auto& [chunk_span, shape] = chunks[i];
        EXPECT_THAT(chunk_span, SizeIs(kChunkSize));
        EXPECT_EQ(shape, ttnn::Shape({1, kChunkSize}));
        for (int j = 0; j < kChunkSize; j++) {
            EXPECT_EQ(chunk_span[j], i * kChunkSize + j);
        }
    }
}

TEST(PartitionTest, ChunkDoesNotAccessData) {
    //  Create a read-protected memory region, and point `tt::stl::Span` to it.
    //  `chunk` should not access the data, and should only calculate offsets and shapes.
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_NE(page_size, -1);

    const size_t total_size = 10 * page_size;
    const int num_chunks = 10;

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
        auto chunks = chunk(protected_span, ttnn::Shape({total_size}), num_chunks);

        EXPECT_THAT(chunks, SizeIs(num_chunks));
        for (const auto& [chunk_span, chunk_shape] : chunks) {
            EXPECT_THAT(chunk_span, SizeIs(page_size));
            EXPECT_EQ(chunk_shape, ttnn::Shape({page_size}));
        }
    } else {
        FAIL() << "segfault occurred when calling `chunk`";
    }

    // Cleanup.
    ASSERT_EQ(sigaction(SIGSEGV, &old_action, nullptr), 0);
    ASSERT_EQ(munmap(mapped_mem, total_size), 0);
}

}  // namespace
}  // namespace ttnn
