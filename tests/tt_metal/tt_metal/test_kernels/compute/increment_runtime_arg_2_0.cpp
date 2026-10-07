// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 compute variant of increment_runtime_arg.cpp.

#include <cstdint>

#include "api/compute/common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_unique_rt_args = get_arg(args::num_unique_rt_args);
    constexpr uint32_t num_common_rt_args = get_arg(args::num_common_rt_args);
    constexpr uint32_t rt_args_base = get_arg(args::rt_args_base);
    constexpr uint32_t common_rt_args_base = get_arg(args::common_rt_args_base);
    constexpr uint32_t unique_arg_incr_val = 10;
    constexpr uint32_t common_arg_incr_val = 100;

    // Only one Tensix thread may publish the results.
#ifdef TRISC_PACK
    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        auto* arg_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt_args_base + (i * sizeof(uint32_t)));
        arg_ptr[0] = get_vararg(i) + unique_arg_incr_val;
    }

    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        auto* arg_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(common_rt_args_base + (i * sizeof(uint32_t)));
        arg_ptr[0] = get_common_vararg(i) + common_arg_incr_val;
    }
#endif
}
