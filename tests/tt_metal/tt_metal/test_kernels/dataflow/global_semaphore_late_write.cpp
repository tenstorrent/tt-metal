// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Waits `delay_cycles`, then writes `value` to the local L1 word at `addr`: a late write to a semaphore
// that the host has already released.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t value = get_arg_val<uint32_t>(1);
    const uint32_t delay_cycles = get_arg_val<uint32_t>(2);

    riscv_wait(delay_cycles);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value;
}
