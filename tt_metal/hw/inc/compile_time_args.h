// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#ifndef FORCE_INLINE
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

template <class T, class... Ts>
FORCE_INLINE constexpr std::array<T, sizeof...(Ts)> make_array(Ts... values) {
    return {T(values)...};
}

#ifndef KERNEL_COMPILE_TIME_ARGS
#define KERNEL_COMPILE_TIME_ARGS
#endif

constexpr auto kernel_compile_time_args = make_array<std::uint32_t>(KERNEL_COMPILE_TIME_ARGS);

template <uint32_t Idx>
constexpr uint32_t get_ct_arg() {
    static_assert(Idx < kernel_compile_time_args.size(), "Index out of range");
    return kernel_compile_time_args[Idx];
}

constexpr bool ct_streq(const char* a, const char* b) {
    for (int i = 0;; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
        if (a[i] == '\0') {
            return true;
        }
    }
}

constexpr uint32_t get_named_arg(const char* name) {
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
#define X(name_str, value)        \
    if (ct_streq(name, name_str)) \
        return value;
    KERNEL_COMPILE_TIME_ARG_MAP
#undef X
#endif
    return 0;
}

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
#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()

// clang-format off
/**
 * Returns the value of a named constexpr argument from kernel_compile_time_args array provided during kernel creation using
 * CreateKernel calls. The name-to-index mapping is defined via KERNEL_COMPILE_TIME_ARG_MAP.
 *
 * Return value: constexpr uint32_t
 *
 * | Argument              | Description                        | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------|-----------------------|-------------|----------|
 * | arg_name              | The name of the argument           | string literal        | defined names | True   |
 */
// clang-format on
#define get_compile_time_arg_val_by_name(arg_name) get_named_arg(arg_name)
