// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_all_includes.hpp>
#include <span>
#include <ttnn/tensor/shape/shape.hpp>

// TODO: decide if we want to use xarray everwhere or xtensor is ok
/*
Difference between xtensor and xarray:

xarray<T> : tensor that can be reshaped to any number of dimensions. xtensor<T, N> : tensor with a number of dimensions
set to N at compile time. xtensor_fixed<T, xshape<I, J, K> : tensor whose shape is fixed at compile time.
*/

namespace ttml::core {
template <class T>
auto span_to_xtensor(std::span<T>& vec, const ttnn::SimpleShape& shape) {
    return xt::adapt(vec, {shape.cbegin(), shape.cend()});
}
template <class T>
auto xtensor_to_span(const xt::xarray<T>& xtensor) {
    return std::span(xtensor.template begin<T>(), xtensor.template end<T>());
}

// TODO: decide if we want to keep this function with E or use the xtensor type directly
template <typename E>
std::array<uint32_t, 4> get_shape_4d(const E& expr) {
    const int max_dims = 4;
    // TODO: Ensure that E is an xtensor expression

    // Retrieve the shape of the tensor
    auto expr_shape = expr.shape();
    std::array<uint32_t, 4> shape4d = {1, 1, 1, 1};

    size_t dims = expr_shape.size();

    if (dims > max_dims) {
        throw std::runtime_error("Number of dimensions greater than 4");
    }

    // Copy the dimensions into the shape array
    for (size_t i = 0; i < dims; ++i) {
        shape4d[i] = static_cast<uint32_t>(expr_shape[i + max_dims - dims]);
    }

    return shape4d;
}
}  // namespace ttml::core
