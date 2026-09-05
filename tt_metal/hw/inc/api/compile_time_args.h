// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_METAL_COMPILE_TIME_ARGS_H
#define TT_METAL_COMPILE_TIME_ARGS_H

#include <array>
#include <cstdint>

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

#endif  // TT_METAL_COMPILE_TIME_ARGS_H

// The named-argument API lives in api/named_compile_time_args.h, which every
// definer of KERNEL_COMPILE_TIME_ARG_MAP includes itself, directly after the
// #define -- the generated map header and the emulator wrapper both do. This
// header deliberately does not tail-include it: under TT_METAL_JIT_PCH this
// header sits inside the precompiled prelude, where the macro does not exist
// yet, so an include from here could only ever bake in the disabled state.
