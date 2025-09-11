// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_METAL_COMPILE_TIME_ARGS_H
#define TT_METAL_COMPILE_TIME_ARGS_H

#include <array>
#include <cstdint>
#include <string_view>

#include "tt_metal/hw/inc/debug/assert.h"

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

#ifdef KERNEL_COMPILE_TIME_ARG_MAP
namespace {
constexpr std::pair<std::string_view, uint32_t> named_args_map[] = {KERNEL_COMPILE_TIME_ARG_MAP};
}
#endif

// TODO #28026: Migrate to C++20 standards when available, see related issue for more details.
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
constexpr uint32_t get_named_ct_arg(std::string_view name) {
    for (const auto& [arg_name, arg_value] : named_args_map) {
        if (name == arg_name) {
            return arg_value;
        }
    }
    // This should never be reached if the named argument is defined in KERNEL_COMPILE_TIME_ARG_MAP.
    // Upon reaching this point, compilation should fail.
    // Note: Compilation currently fails with a segfault.
    __builtin_unreachable();  // Invalid named compile time argument
}
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
#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()

// clang-format off
/**
 * Returns the value of a named constexpr argument from kernel_compile_time_args array provided during kernel creation using
 * CreateKernel calls. The name-to-index mapping is defined via KERNEL_COMPILE_TIME_ARG_MAP. Migrating all existing kernels to
 * use named compile time arguments is not a trivial task, so backward-compatibility is maintained by allowing the use of the
 * get_compile_time_arg_val function with an index. get_compile_time_arg_val can be deprecated in the future upon completion
 * of the migration and by request. See vecadd_multi_core.cpp for an example of how to use named compile time arguments.
 * Note: Return value must be stored in a constexpr variable to guarantee compile time evaluation.
 *
 * Return value: constexpr uint32_t
 *
 * | Argument              | Description                        | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------|-----------------------|-------------|----------|
 * | arg_name              | The name of the argument           | string literal        | defined names | True   |
 */
// clang-format on
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
constexpr uint32_t get_named_compile_time_arg_val(std::string_view name) { return get_named_ct_arg(name); }
#endif

#endif  // TT_METAL_COMPILE_TIME_ARGS_H
