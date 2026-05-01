// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "experimental/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"

/**
 * add two ints
 * args are in L1
 * result is in L1
 */

void kernel_main() {
    uint32_t val_a = get_vararg(0);
    uint32_t val_b = get_vararg(1);
    constexpr uint32_t l1_address = get_arg(args::l1_address);

    experimental::CoreLocalMem<std::uint32_t> result(l1_address);

    result[0] = val_a + val_b;
    DPRINT << "Adding two ints: " << val_a << " + " << val_b << " = " << result[0] << ENDL();
    DEVICE_PRINT("Adding two ints: {} + {} = {}\n", val_a, val_b, result[0]);
}
