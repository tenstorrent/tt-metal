// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xexception.hpp>
#include <xtensor/xiterator.hpp>
#include <xtensor/xlayout.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xstorage.hpp>
#include <xtensor/xtensor_forward.hpp>
#include <xtensor/xtensor_simd.hpp>
#include <xtensor/xutils.hpp>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
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

    xt::allclose(result, expected);
}

TEST(PartitionTest, AxisOne) {
    xt::xarray<int> x = {{1, 2, 3}, {4, 5, 6}};
    xt::xarray<int> y = {{7, 8}, {9, 10}};
    std::vector<xt::xarray<int>> input = {x, y};

    xt::xarray<int> result = concat(input, 1);
    xt::xarray<int> expected = {{1, 2, 3, 7, 8}, {4, 5, 6, 9, 10}};

    xt::allclose(result, expected);
}

TEST(PartitionTest, MultipleArraysAxis0) {
    xt::xarray<float> a = {1.0f, 2.0f};
    xt::xarray<float> b = {3.0f, 4.0f};
    xt::xarray<float> c = {5.0f, 6.0f};
    std::vector<xt::xarray<float>> input = {a, b, c};

    xt::xarray<float> result = concat(input, 0);
    xt::xarray<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    xt::allclose(result, expected);
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

    xt::allclose(result, expected);
}

TEST(PartitionTest, HigherAxis) {
    xt::xarray<int> arr1 = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    xt::xarray<int> arr2 = {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}};
    // Both have shape (2,2,2)

    std::vector<xt::xarray<int>> input = {arr1, arr2};
    xt::xarray<int> result = concat(input, 2);
    // Expected shape: (2,2,4)
    xt::xarray<int> expected = {{{1, 2, 9, 10}, {3, 4, 11, 12}}, {{5, 6, 13, 14}, {7, 8, 15, 16}}};

    xt::allclose(result, expected);
}

}  // namespace
}  // namespace ttnn
