// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 data-movement variant of increment_runtime_arg.cpp: takes the counts and result bases as
// named compile-time args and reads the runtime args as positional varargs. The compute variant is in
// test_kernels/compute/increment_runtime_arg_2_0.cpp.

#include <cstdint>

#include "internal/risc_attribs.h"
#include "api/core_local_mem.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_unique_rt_args = get_arg(args::num_unique_rt_args);
    constexpr uint32_t num_common_rt_args = get_arg(args::num_common_rt_args);
    constexpr uint32_t rt_args_base = get_arg(args::rt_args_base);
    constexpr uint32_t common_rt_args_base = get_arg(args::common_rt_args_base);
    constexpr uint32_t unique_arg_incr_val = 10;
    constexpr uint32_t common_arg_incr_val = 100;

    // The host reads these over the NOC at the cached address, so a DM core must store through the
    // uncached alias for them to be visible.
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    CoreLocalMem<uint32_t> unique_results(rt_args_base + MEM_L1_UNCACHED_BASE);
    CoreLocalMem<uint32_t> common_results(common_rt_args_base + MEM_L1_UNCACHED_BASE);
#else
    CoreLocalMem<uint32_t> unique_results(rt_args_base);
    CoreLocalMem<uint32_t> common_results(common_rt_args_base);
#endif

    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        unique_results[i] = get_vararg(i) + unique_arg_incr_val;
    }

    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        common_results[i] = get_common_vararg(i) + common_arg_incr_val;
    }
}
