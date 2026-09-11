// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reads eth tile registers for the host: runtime args are the count, a register to write first (0 = none) with
// its value, then the addresses to read; the values land in L1 at the compile-time result address after a done
// marker carrying the count.

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"

constexpr uint32_t kResultAddr = get_compile_time_arg_val(0);

void kernel_main() {
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultAddr);
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t poke_addr = get_arg_val<uint32_t>(1);
    const uint32_t poke_val = get_arg_val<uint32_t>(2);
    if (poke_addr != 0) {
        *reinterpret_cast<volatile uint32_t*>(poke_addr) = poke_val;
    }
    for (uint32_t i = 0; i < n; i++) {
        out[1 + i] = *reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(3 + i));
    }
    out[0] = 0xD0E00000u | n;
}
