// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
#define get_compile_time_arg_val(arg_idx) KERNEL_COMPILE_TIME_ARG_##arg_idx
}  // namespace ckernel
