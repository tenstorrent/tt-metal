// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#pragma once
template <typename T, std::size_t N>
constexpr const T& get_kernel_compile_time_arg(const T (&array)[N], std::size_t arg_idx) {
    if (arg_idx > N) {
        static_assert(false, "Argument index out of bounds for kernel_compile_time_arg.");
    }
    return array[arg_idx];
}
namespace ckernel {

#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_##arg_idx
#define kernel_compile_time_arg(arg_idx) get_kernel_compile_time_arg(KERNEL_COMPILE_TIME_ARGS, arg_idx)

}  // namespace ckernel
