// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <iterator>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

#include "core/random.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/generators/xbuilder.hpp"

namespace ttml::test_utils {

template <typename T>
inline void fill_uniform(std::span<T> data, T min, T max, uint32_t seed) {
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
inline std::vector<T> make_uniform_vector(std::size_t count, T min, T max, uint32_t seed) {
    std::vector<T> data(count);
    fill_uniform<T>(std::span<T>{data.data(), data.size()}, min, max, seed);
    return data;
}

template <typename T, typename Shape>
inline xt::xarray<T> make_uniform_xarray(const Shape& shape, uint32_t seed, T min = T{-1}, T max = T{1}) {
    std::vector<std::size_t> xt_shape;
    xt_shape.reserve(std::size(shape));
    for (const auto dim : shape) {
        xt_shape.push_back(static_cast<std::size_t>(dim));
    }
    xt::xarray<T> x = xt::empty<T>(xt_shape);
    fill_uniform<T>(std::span<T>{x.data(), x.size()}, min, max, seed);
    return x;
}

}  // namespace ttml::test_utils
