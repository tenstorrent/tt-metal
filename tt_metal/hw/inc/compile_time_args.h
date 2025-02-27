// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

template <class T, class... Ts>
FORCE_INLINE constexpr std::array<T, sizeof...(Ts)> make_array(Ts... values) {
    return {T(values)...};
}

#if defined(KERNEL_COMPILE_TIME_ARGS)
constexpr auto kernel_compile_time_args = make_array<std::uint32_t>(KERNEL_COMPILE_TIME_ARGS);
#else
constexpr auto kernel_compile_time_args = make_array<std::uint32_t>();
#endif

// clang-format off
/**
 * Returns the value of a constexpr argument from kernel_compile_time_args array provided during kernel creation using
 * CreateKernel calls.
 *
 * Return value: constexpr uint32_t
 *
 * | Argument              | Description                        | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------|-----------------------|-------------|----------|
 * | arg_idx               | The index of the argument          | uint32_t              | 0 to 31     | True     |
 */
// clang-format on
#define get_compile_time_arg_val(arg_idx) \
    ((arg_idx) < kernel_compile_time_args.size() ? kernel_compile_time_args[(arg_idx)] : 0)
