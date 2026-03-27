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
        const float min_f = static_cast<float>(min);
        const float max_f = static_cast<float>(max);
        ttml::core::parallel_generate(
            data, [min_f, max_f]() { return std::uniform_real_distribution<float>(min_f, max_f); }, seed);
    }
}

template <typename T>
inline void fill_normal(std::span<T> data, float mean, float stddev, uint32_t seed) {
    ttml::core::parallel_generate(
        data, [mean, stddev]() { return std::normal_distribution<float>(mean, stddev); }, seed);
}

template <typename T>
inline std::vector<T> make_uniform_vector(std::size_t count, T min, T max, uint32_t seed) {
    std::vector<T> data(count);
    fill_uniform<T>(std::span<T>{data.data(), data.size()}, min, max, seed);
    return data;
}

template <typename T>
inline std::vector<T> make_normal_vector(std::size_t count, float mean, float stddev, uint32_t seed) {
    std::vector<T> data(count);
    fill_normal<T>(std::span<T>{data.data(), data.size()}, mean, stddev, seed);
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

template <typename T, typename Shape>
inline xt::xarray<T> make_normal_xarray(const Shape& shape, uint32_t seed, float mean = 0.0F, float stddev = 1.0F) {
    std::vector<std::size_t> xt_shape;
    xt_shape.reserve(std::size(shape));
    for (const auto dim : shape) {
        xt_shape.push_back(static_cast<std::size_t>(dim));
    }
    xt::xarray<T> x = xt::empty<T>(xt_shape);
    fill_normal<T>(std::span<T>{x.data(), x.size()}, mean, stddev, seed);
    return x;
}

}  // namespace ttml::test_utils
