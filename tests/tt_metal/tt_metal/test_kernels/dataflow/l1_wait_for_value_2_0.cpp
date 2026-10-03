// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

// Spins until the host writes the expected value to an L1 word.
void kernel_main() {
    std::uint32_t addr = get_arg(args::address);
    std::uint32_t value = get_arg(args::value);

    volatile tt_l1_ptr std::uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
    while (ptr[0] != value) {
        invalidate_l1_cache();
    }
}
