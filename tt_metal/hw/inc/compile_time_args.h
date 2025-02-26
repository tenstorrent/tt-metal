// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

namespace ckernel {
template <class T, class... Ts>
FORCE_INLINE constexpr std::array<T, sizeof...(Ts)> make_array(Ts... values) {
    return {T(values)...};
}

#if defined(KERNEL_COMPILE_TIME_ARGS)
constexpr auto kernel_compile_time_args = make_array<std::uint32_t>(KERNEL_COMPILE_TIME_ARGS);
#else
constexpr auto kernel_compile_time_args = make_array<std::uint32_t>();
#endif

#define get_compile_time_arg_val(arg_idx) kernel_compile_time_args[arg_idx]
}  // namespace ckernel
