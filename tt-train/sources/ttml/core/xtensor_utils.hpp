// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <core/xtensor_all_includes.hpp>
#include <cstdint>
#include <span>
#include <ttnn/tensor/shape/shape.hpp>

#include "xtensor/xbuffer_adaptor.hpp"

// TODO: decide if we want to use xarray everwhere or xtensor is ok
/*
Difference between xtensor and xarray:

xarray<T> : tensor that can be reshaped to any number of dimensions. xtensor<T, N> : tensor with a number of dimensions
set to N at compile time. xtensor_fixed<T, xshape<I, J, K> : tensor whose shape is fixed at compile time.
*/

namespace ttml::core {
template <class T>
auto span_to_xtensor(std::span<T> vec, const ttnn::SimpleShape& shape) {
    std::vector<size_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(vec.data(), vec.size(), xt::acquire_ownership(), shape_vec);
}
template <class T>
auto xtensor_to_span(const xt::xarray<T>& xtensor) {
    auto adaptor = xt::adapt(xtensor.data(), xtensor.size(), xt::no_ownership());
    return std::span(adaptor.data(), adaptor.size());
}

// TODO: decide if we want to keep this function with E or use the xtensor type directly
template <typename E>
std::array<uint32_t, 4> get_shape_4d(const E& expr) {
    const int max_dims = 4;
    // TODO: Ensure that E is an xtensor expression

    // Retrieve the shape of the tensor
    auto& expr_shape = expr.shape();
    std::array<uint32_t, 4> shape4d = {1, 1, 1, 1};

    size_t dims = expr_shape.size();

    if (dims > max_dims) {
        throw std::runtime_error("Number of dimensions greater than 4");
    }

    // Copy the dimensions into the shape array
    for (size_t i = 0; i < dims; ++i) {
        shape4d[i + max_dims - dims] = static_cast<uint32_t>(expr_shape[i]);
    }

    return shape4d;
}

namespace detail {
template <typename S>
struct ShapeIndex {
    const S& shape;
    std::vector<size_t> index;

    explicit ShapeIndex(const S& shape);
    bool increment();
};

template <typename S>
ShapeIndex<S>::ShapeIndex(const S& shape) : shape(shape), index(shape.size(), 0) {
}
template <typename S>
bool ShapeIndex<S>::increment() {
    for (int dim = shape.size() - 1; dim >= 0; --dim) {
        if (index[dim] >= shape[dim] - 1)
            continue;

        ++index[dim];
        for (size_t lsd = dim + 1; lsd < index.size(); ++lsd)  // lsd = less significant dimension
            index[lsd] = 0;
        return true;
    }

    return false;
}
}  // namespace detail
template <typename T>
xt::xarray<T> concatenate(const std::vector<xt::xarray<T>>& v, const size_t axis = 0) {
    if (v.empty())
        return {};

    auto res_shape = v.front().shape();
    if (axis >= res_shape.size()) {
        throw std::out_of_range("axis is not a dimension of shape");
    }
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i].shape().size() != res_shape.size()) {
            throw std::logic_error("shapes have different dimensionalities");
        }
        for (size_t dim = 0; dim < res_shape.size(); ++dim) {
            if (dim == axis)
                continue;
            if (v[i].shape(dim) != res_shape[dim]) {
                throw std::logic_error("incompatible shapes");
            }
        }
        res_shape[axis] += v[i].shape(axis);
    }

    xt::xarray<T> result(res_shape);
    std::vector<size_t> dst_index(res_shape.size(), 0);
    for (size_t i = 0; i < v.size(); ++i) {
        const size_t axis_index_offset = dst_index[axis];
        detail::ShapeIndex src_index(v[i].shape());
        do {
            for (size_t dim = 0; dim < res_shape.size(); ++dim) {
                if (dim == axis)
                    dst_index[dim] = axis_index_offset + src_index.index[dim];
                else
                    dst_index[dim] = src_index.index[dim];
            }
            result[dst_index] = v[i][src_index.index];
        } while (src_index.increment());
        dst_index[axis] += v[i].shape(axis);
    }
    return result;
}

}  // namespace ttml::core
