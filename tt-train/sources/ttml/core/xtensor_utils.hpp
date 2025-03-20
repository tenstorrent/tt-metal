// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/core.h>

#include <core/ttnn_all_includes.hpp>
#include <span>
#include <sstream>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/xtensor/conversion_utils.hpp>
#include <ttnn/tensor/xtensor/partition.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

// TODO: decide if we want to use xarray everwhere or xtensor is ok
/*
Difference between xtensor and xarray:

xarray<T> : tensor that can be reshaped to any number of dimensions. xtensor<T, N> : tensor with a number of dimensions
set to N at compile time. xtensor_fixed<T, xshape<I, J, K> : tensor whose shape is fixed at compile time.
*/

namespace ttml::core {
template <class T>
xt::xarray<T> span_to_xtensor_view(std::span<T> vec, const ttnn::Shape& shape) {
    return ttnn::experimental::xtensor::span_to_xtensor_view(vec, shape);
}
template <class T>
auto xtensor_to_span(const xt::xarray<T>& xtensor) {
    return ttnn::experimental::xtensor::xtensor_to_span(xtensor);
}

template <typename T>
xt::xarray<T> concat(const std::vector<xt::xarray<T>>& v, size_t axis = 0) {
    return ttnn::experimental::xtensor::concat(v, axis);
}

}  // namespace ttml::core

template <typename T>
struct fmt::formatter<xt::xarray<T>> : fmt::formatter<std::string> {
    auto format(const xt::xarray<T>& arr, fmt::format_context& ctx) const {
        std::stringstream ss;
        ss << arr;
        return fmt::formatter<std::string>::format(ss.str(), ctx);
    }
};

// Add formatter for xarray_container
template <typename T, xt::layout_type L, class SC>
struct fmt::formatter<xt::xarray_container<T, L, SC>> : fmt::formatter<std::string> {
    auto format(const xt::xarray_container<T, L, SC>& arr, fmt::format_context& ctx) const {
        std::stringstream ss;
        ss << arr;
        return fmt::formatter<std::string>::format(ss.str(), ctx);
    }
};
