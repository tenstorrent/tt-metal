// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Hangs, keeping a program in flight for the L1 ECC halt tests.
// arg 0: nonzero raises IRQ 3 through the watchdog's software interrupt (WDT_CNTL[2]), the line the
//        L1 ECC halt handler is armed on, so the real handler runs without an ECC error.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t raise_irq3 = get_arg_val<uint32_t>(0);

    if (raise_irq3 != 0) {
        *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WDT_CNTL) = 1u << 2;
    }
    volatile uint32_t keep_spinning = 1;
    while (keep_spinning != 0) {
    }
}
