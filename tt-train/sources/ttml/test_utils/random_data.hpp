// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <iterator>
#include <random>
#include <span>
#include <string>
#include <tt_stl/reflection.hpp>
#include <type_traits>
#include <vector>

#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/generators/xbuilder.hpp"

namespace ttml::test_utils {

template <typename T>
inline void fill_uniform(std::span<T> data, const T min, const T max, const uint32_t seed) {
    if constexpr (std::is_integral_v<T>) {
        ttml::core::parallel_generate(data, [min, max]() { return std::uniform_int_distribution<T>(min, max); }, seed);
    } else {
        using DistScalar = std::conditional_t<std::is_floating_point_v<T>, T, float>;
        const DistScalar min_v = static_cast<DistScalar>(min);
        const DistScalar max_v = static_cast<DistScalar>(max);
        ttml::core::parallel_generate(
            data, [min_v, max_v]() { return std::uniform_real_distribution<DistScalar>(min_v, max_v); }, seed);
    }
}

template <typename T>
inline std::vector<T> make_uniform_vector(const std::size_t vec_size, const T min, const T max, const uint32_t seed) {
    std::vector<T> data(vec_size);
    fill_uniform<T>(std::span<T>{data.data(), data.size()}, min, max, seed);
    return data;
}

template <typename T, typename Shape>
inline xt::xarray<T> make_uniform_xarray(const Shape& shape, const T min, const T max, const uint32_t seed) {
    std::vector<std::size_t> xt_shape;
    xt_shape.reserve(std::size(shape));
    for (const auto dim : shape) {
        xt_shape.push_back(static_cast<std::size_t>(dim));
    }
    xt::xarray<T> x = xt::empty<T>(xt_shape);
    fill_uniform<T>(std::span<T>{x.data(), x.size()}, min, max, seed);
    return x;
}

// Whole numbers drawn uniformly from [-bound, bound] and rounded. Small enough integers keep every product and partial
// sum exact in bf16, so a device result can be checked against a host reference bit for bit.
inline xt::xarray<float> make_small_ints(const std::vector<uint32_t>& shape, const float bound, const uint32_t seed) {
    return xt::round(make_uniform_xarray<float>(shape, -bound, bound, seed));
}

// Copy of t with its leading dims flattened into rows: [..., rows, cols] -> [volume / cols, cols].
inline xt::xarray<float> flatten_rows(const xt::xarray<float>& t) {
    xt::xarray<float> flat = t;
    flat.reshape(std::vector<std::size_t>{t.size() / t.shape().back(), t.shape().back()});
    return flat;
}

// A device op's name as the graph capture records it. It comes from the op type, so renaming the op breaks the build
// rather than silently counting zero.
template <typename Op>
std::string op_name() {
    return std::string(ttsl::get_type_name<Op>());
}

// Runs out->backward() under a NORMAL-mode graph capture and returns the calltrace. In NORMAL mode the ops really
// execute, so the grads they produce can be checked as well.
inline std::vector<std::string> backward_calltrace(const ttml::autograd::TensorPtr& out) {
    // The scoped capture ends even if backward throws, so a failing test cannot leave later tests capturing.
    ttnn::graph::ScopedGraphCapture capture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
    out->backward();
    return ttnn::graph::extract_calltrace(capture.end_graph_capture());
}

}  // namespace ttml::test_utils
