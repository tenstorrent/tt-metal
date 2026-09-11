// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Fill `size` bytes of this core's L1 at `addr` with `value` (test scaffolding: a
// single-page interleaved L1 buffer lives in ONE bank/core, so host-side buffer
// writes do not reach the L1 of the core a kernel actually uses).
void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t size = get_arg_val<uint32_t>(1);
    const uint32_t value = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < size / 4; i++) {
        p[i] = value;
    }
}
