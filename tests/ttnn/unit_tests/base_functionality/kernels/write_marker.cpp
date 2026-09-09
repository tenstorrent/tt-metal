// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The smallest observable side effect a launch can have: one word written to one L1 address.
// tests/ttnn/unit_tests/base_functionality/test_reload_host_support.py uses it to tell a program
// that was CONFIGURED (binary and launch message on the core) from one that RAN.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t value = get_arg_val<uint32_t>(1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value;
}
