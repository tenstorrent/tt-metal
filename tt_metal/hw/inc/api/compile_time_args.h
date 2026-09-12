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

// Tensor-accessor templates in the shared prelude call this with dependent
// indices. Its definition and argument array arrive before instantiation.
template <uint32_t Idx>
constexpr uint32_t get_ct_arg();

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

#endif  // TT_METAL_COMPILE_TIME_ARGS_H

// Precompile the library headers and helper, then materialize the kernel's
// positional arguments after loading the PCH. Do not inspect the argument macro
// while building the PCH: even an #ifndef would constrain GCC's macro validation.
#if !defined(TT_METAL_PCH_BUILD) && !defined(TT_METAL_POSITIONAL_CT_ARGS_DEFINED)
#define TT_METAL_POSITIONAL_CT_ARGS_DEFINED

#ifndef KERNEL_COMPILE_TIME_ARGS
#define KERNEL_COMPILE_TIME_ARGS
#endif

constexpr auto kernel_compile_time_args = make_array<std::uint32_t>(KERNEL_COMPILE_TIME_ARGS);

template <uint32_t Idx>
constexpr uint32_t get_ct_arg() {
    static_assert(Idx < kernel_compile_time_args.size(), "Index out of range");
    return kernel_compile_time_args[Idx];
}

#endif  // positional arguments available

// Preserve the named API for callers that define the map and include this header
// directly. Keep this outside the positional API's guard so a map defined after
// PCH consumption can still declare the named API on a later include.
#include "api/named_compile_time_args.h"
