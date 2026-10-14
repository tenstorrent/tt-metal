// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writes one runtime-arg word at a runtime-arg L1 address: the smallest observable kernel.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t value = get_arg_val<uint32_t>(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value;
}
