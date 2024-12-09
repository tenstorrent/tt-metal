// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "xtensor_utils.hpp"

namespace ttml::core {
namespace detail {
template <typename T, std::size_t... Indices>
auto vector_to_tuple_helper(const std::vector<T>& v, std::index_sequence<Indices...>) {
    return std::make_tuple(v[Indices]...);
}

template <std::size_t N, typename T>
auto vector_to_tuple(const std::vector<T>& buffer) {
    assert(buffer.size() >= N);
    return vector_to_tuple_helper(buffer, std::make_index_sequence<N>());
}

template <typename T, int N>
xt::xarray<T> concat_helper(const std::vector<xt::xarray<T>>& v, size_t axis = 0) {
    constexpr int FIXED_N = N < 2 ? 2 : N;
    if (N < 2) {
        throw std::runtime_error("Tuple size in concatenate must be greater than 1");
    }
    auto tuple = detail::vector_to_tuple<FIXED_N>(v);
    return xt::concatenate(std::move(tuple), axis);
}

template <class T, size_t... I>
consteval auto create_array_impl(std::index_sequence<I...>) {
    return std::array<xt::xarray<T> (*)(const std::vector<xt::xarray<T>>& v, size_t axis), sizeof...(I)>{
        concat_helper<T, I>...};
}

template <class T, size_t max>
consteval auto create_array() {
    return create_array_impl<T>(std::make_index_sequence<max>());
}

}  // namespace detail

template <typename T>
xt::xarray<T> concatenate(const std::vector<xt::xarray<T>>& v, size_t axis) {
    constexpr size_t MAX_TUPLE_SIZE = 64;

    if (v.empty()) {
        return {};
    }
    if (v.size() == 1) {
        return v.front();
    }
    if (v.size() > MAX_TUPLE_SIZE) {
        throw std::runtime_error(
            fmt::format("Number of tensors to concatenate exceeds the maximum supported size {}", MAX_TUPLE_SIZE));
    }
    constexpr auto table = detail::create_array<T, MAX_TUPLE_SIZE>();
    return (*table[v.size()])(v, axis);
}

template xt::xarray<double> concatenate(const std::vector<xt::xarray<double>>& v, size_t axis);
template xt::xarray<float> concatenate(const std::vector<xt::xarray<float>>& v, size_t axis);
template xt::xarray<uint32_t> concatenate(const std::vector<xt::xarray<uint32_t>>& v, size_t axis);
template xt::xarray<int32_t> concatenate(const std::vector<xt::xarray<int32_t>>& v, size_t axis);
}  // namespace ttml::core
