// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>

#include <vector>

#include "ttnn/tensor/xtensor/conversion_utils.hpp"

namespace ttnn {
namespace {

using ::testing::ElementsAre;
using ::ttnn::experimental::xtensor::XtensorAdapter;

TEST(XtensorAdapterTest, BasicConstruction) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<size_t> shape = {2, 3};

    XtensorAdapter<float> adapter(std::move(data), shape);

    EXPECT_THAT(adapter.data(), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f));
    EXPECT_EQ(adapter.expr().shape()[0], 2u);
    EXPECT_EQ(adapter.expr().shape()[1], 3u);
    EXPECT_FLOAT_EQ(adapter.expr()(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(adapter.expr()(1, 2), 6.0f);
}

TEST(XtensorAdapterTest, CopyConstructor) {
    std::vector<int> data = {1, 2, 3, 4};
    std::vector<size_t> shape = {2, 2};

    XtensorAdapter<int> adapter1(std::move(data), shape);
    XtensorAdapter<int> adapter2(adapter1);

    EXPECT_EQ(adapter1.data().size(), adapter2.data().size());
    EXPECT_NE(&adapter1.data()[0], &adapter2.data()[0]);

    adapter2.data()[0] = 10;
    EXPECT_EQ(adapter2.expr()(0, 0), 10);
    EXPECT_EQ(adapter1.expr()(0, 0), 1);
}

TEST(XtensorAdapterTest, MoveConstructor) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<size_t> shape = {3, 2};

    XtensorAdapter<double> adapter1(std::move(data), shape);
    auto original_data = adapter1.data();
    XtensorAdapter<double> adapter2(std::move(adapter1));

    EXPECT_EQ(adapter2.data(), original_data);
    EXPECT_EQ(adapter2.expr().shape()[0], 3u);
    EXPECT_EQ(adapter2.expr().shape()[1], 2u);

    adapter2.data()[0] = 10.0;
    EXPECT_DOUBLE_EQ(adapter2.expr()(0, 0), 10.0);
}

TEST(XtensorAdapterTest, CopyAssignment) {
    std::vector<float> data1 = {1.0f, 2.0f};
    std::vector<float> data2 = {3.0f, 4.0f, 5.0f, 6.0f};

    XtensorAdapter<float> adapter1(std::move(data1), {2});
    XtensorAdapter<float> adapter2(std::move(data2), {2, 2});

    adapter1 = adapter2;

    EXPECT_EQ(adapter1.data().size(), 4u);
    EXPECT_EQ(adapter1.expr().shape()[0], 2u);
    EXPECT_EQ(adapter1.expr().shape()[1], 2u);

    adapter1.data()[0] = 10.0f;
    EXPECT_FLOAT_EQ(adapter1.expr()(0, 0), 10.0f);
    EXPECT_FLOAT_EQ(adapter2.expr()(0, 0), 3.0f);
}

TEST(XtensorAdapterTest, MoveAssignment) {
    std::vector<int> data1 = {1, 2, 3};
    std::vector<int> data2 = {4, 5, 6, 7, 8, 9};

    XtensorAdapter<int> adapter1(std::move(data1), {3});
    XtensorAdapter<int> adapter2(std::move(data2), {2, 3});

    adapter1 = std::move(adapter2);

    EXPECT_THAT(adapter1.data(), ElementsAre(4, 5, 6, 7, 8, 9));
    EXPECT_EQ(adapter1.expr().shape()[0], 2u);
    EXPECT_EQ(adapter1.expr().shape()[1], 3u);
}

TEST(XtensorAdapterTest, SelfAssignment) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    XtensorAdapter<double> adapter(std::move(data), {2, 2});

    auto* ptr = &adapter;
    adapter = *ptr;

    EXPECT_THAT(adapter.data(), ElementsAre(1.0, 2.0, 3.0, 4.0));
    EXPECT_DOUBLE_EQ(adapter.expr()(0, 0), 1.0);

    adapter = std::move(*ptr);

    EXPECT_THAT(adapter.data(), ElementsAre(1.0, 2.0, 3.0, 4.0));
    EXPECT_DOUBLE_EQ(adapter.expr()(1, 1), 4.0);
}

TEST(XtensorAdapterTest, ExpressionOperations) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    XtensorAdapter<double> adapter(std::move(data), {2, 2});

    // Use xtensor operations on expr()
    auto squared = xt::square(adapter.expr());
    EXPECT_DOUBLE_EQ(squared(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(squared(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(squared(1, 0), 9.0);
    EXPECT_DOUBLE_EQ(squared(1, 1), 16.0);

    // Modify through expr
    adapter.expr() *= 2.0;
    EXPECT_THAT(adapter.data(), ElementsAre(2.0, 4.0, 6.0, 8.0));
}

TEST(XtensorAdapterTest, EmptyAdapter) {
    std::vector<int> data;
    std::vector<size_t> shape = {0};

    XtensorAdapter<int> adapter(std::move(data), shape);

    EXPECT_EQ(adapter.data().size(), 0u);
    EXPECT_EQ(adapter.expr().size(), 0u);
}

TEST(XtensorAdapterTest, EmptyShape) {
    std::vector<int> data = {0};
    std::vector<size_t> shape = {};

    XtensorAdapter<int> adapter(std::move(data), shape);

    EXPECT_EQ(adapter.data().size(), 1u);
    EXPECT_EQ(adapter.expr().size(), 1u);
}

}  // namespace
}  // namespace ttnn
