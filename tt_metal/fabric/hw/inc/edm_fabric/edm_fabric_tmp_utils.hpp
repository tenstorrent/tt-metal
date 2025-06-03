// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

namespace tt::tt_fabric {

// a generic tuple‐for_each helper
template <typename Tuple, typename F, size_t... Is>
constexpr void tuple_for_each_impl(Tuple&& t, F&& f, std::index_sequence<Is...>) {
    // expansion: f(std::get<0>(t),0), f(std::get<1>(t),1), …
    (f(std::get<Is>(std::forward<Tuple>(t)), Is), ...);
}

template <typename Tuple, typename F>
constexpr void tuple_for_each(Tuple&& t, F&& f) {
    tuple_for_each_impl(
        std::forward<Tuple>(t),
        std::forward<F>(f),
        std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

}  // namespace tt::tt_fabric
