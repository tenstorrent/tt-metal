// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/container/vector.hpp>
#include <gtest/gtest.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/containers/xcontainer.hpp>
#include <xtensor/core/xiterator.hpp>
#include <xtensor/core/xlayout.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xshape.hpp>
#include <xtensor/containers/xstorage.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/utils/xtensor_simd.hpp>
#include <cstddef>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include <tt-metalium/shape.hpp>
#include <tt_stl/span.hpp>
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"

namespace ttnn {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::ttnn::experimental::xtensor::from_xtensor;
using ::ttnn::experimental::xtensor::get_shape_from_xarray;
using ::ttnn::experimental::xtensor::span_to_xtensor_view;
using ::ttnn::experimental::xtensor::to_xtensor;
using ::ttnn::experimental::xtensor::xtensor_to_span;

TensorSpec get_tensor_spec(const ttnn::Shape& shape) {
    return TensorSpec(
        shape, TensorLayout(tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::ROW_MAJOR, MemoryConfig{}));
}

TEST(XtensorConversionTest, SpanToXtensor) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    tt::stl::Span<const int> data_span(data.data(), data.size());
    ttnn::Shape shape({2, 3});

    auto result = span_to_xtensor_view(data_span, shape);

    // Check shape
    EXPECT_THAT(result.shape(), ElementsAre(2, 3));

    // Check data
    int expected_val = 1;
    for (size_t i = 0; i < result.shape()[0]; ++i) {
        for (size_t j = 0; j < result.shape()[1]; ++j) {
            EXPECT_EQ(result(i, j), expected_val++);
        }
    }
}

TEST(XtensorConversionTest, XtensorToSpan) {
    xt::xarray<float> arr = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    EXPECT_THAT(xtensor_to_span(arr), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
}

TEST(XtensorConversionTest, GetShape) {
    EXPECT_THAT(get_shape_from_xarray(xt::xarray<int>::from_shape({2, 3, 4, 5, 6})), Eq(ttnn::Shape{2, 3, 4, 5, 6}));
    EXPECT_THAT(get_shape_from_xarray(xt::xarray<int>::from_shape({7})), Eq(ttnn::Shape{7}));
}

TEST(XtensorConversionTest, FromXtensorInvalidShape) {
    xt::xarray<float> arr = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    EXPECT_ANY_THROW(from_xtensor(arr, get_tensor_spec(ttnn::Shape{3, 3})));
}

TEST(XtensorConversionTest, Roundtrip) {
    const std::vector<ttnn::Shape> shapes{
        ttnn::Shape{1},
        ttnn::Shape{1, 1, 1, 1},
        ttnn::Shape{1, 1, 1, 10},
        ttnn::Shape{1, 32, 32, 16},
        ttnn::Shape{1, 40, 3, 128},
        ttnn::Shape{2, 2},
        ttnn::Shape{1, 1, 1, 1, 10},
    };

    for (const auto& shape : shapes) {
        const auto tensor_spec = get_tensor_spec(shape);
        xt::xarray<float> input = xt::arange<float>(shape.volume());
        xt::dynamic_shape<std::size_t> new_shape(shape.cbegin(), shape.cend());
        input.reshape(new_shape);

        auto output = to_xtensor<float>(from_xtensor(input, tensor_spec));
        EXPECT_TRUE(xt::allclose(input, output));
    }
}

}  // namespace
}  // namespace ttnn
