// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#ifdef ARCH_QUASAR
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#endif

/**
 * add two ints
 * args are in L1
 * result is in L1
 */

void kernel_main() {
#ifdef ARCH_QUASAR
    uint32_t a = get_arg(args::a);
    uint32_t b = get_arg(args::b);
    constexpr uint32_t l1_address = get_arg(args::l1_address);
    CoreLocalMem<std::uint32_t> result(l1_address);
#else
    uint32_t a = get_arg_val<uint32_t>(0);
    uint32_t b = get_arg_val<uint32_t>(1);
    constexpr uint32_t l1_address = get_compile_time_arg_val(0);
    volatile tt_l1_ptr std::uint32_t* result = (tt_l1_ptr uint32_t*)(l1_address);
#endif

    result[0] = a + b;
    DPRINT("Adding two ints: {} + {} = {}\n", a, b, result[0]);
}
