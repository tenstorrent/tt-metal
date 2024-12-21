// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_##arg_idx

#if defined(KERNEL_COMPILE_TIME_ARG_0)
#define KERNEL_COMPILE_TIME_ARGS \
    (int[]) { KERNEL_COMPILE_TIME_ARG_0 }
#endif

#if defined(KERNEL_COMPILE_TIME_ARG_1)
#undef KERNEL_COMPILE_TIME_ARGS
#define KERNEL_COMPILE_TIME_ARGS \
    (int[]) { KERNEL_COMPILE_TIME_ARG_0, KERNEL_COMPILE_TIME_ARG_1 }
#endif

#if defined(KERNEL_COMPILE_TIME_ARG_2)
#undef KERNEL_COMPILE_TIME_ARGS
#define KERNEL_COMPILE_TIME_ARGS \
    (int[]) { KERNEL_COMPILE_TIME_ARG_0, KERNEL_COMPILE_TIME_ARG_1, KERNEL_COMPILE_TIME_ARG_2 }
#endif
#define get_compile_time_arg_vals(arg_idx) KERNEL_COMPILE_TIME_ARGS[arg_idx]
}  // namespace ckernel
