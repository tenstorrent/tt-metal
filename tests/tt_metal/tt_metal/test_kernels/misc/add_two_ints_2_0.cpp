// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of add_two_ints.cpp. Reads two named runtime args
// and writes the sum to an L1 address bound via the named compile-time arg
// `l1_address`. The legacy positional-arg variant remains in add_two_ints.cpp
// for callers still on the legacy host API.

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

/**
 * add two ints
 * args are in L1
 * result is in L1
 */

void kernel_main() {
    uint32_t a = get_arg(args::a);
    uint32_t b = get_arg(args::b);
    constexpr uint32_t l1_address = get_arg(args::l1_address);
    CoreLocalMem<std::uint32_t> result(l1_address);

    result[0] = a + b;
    DPRINT("Adding two ints: {} + {} = {}\n", a, b, result[0]);
}
