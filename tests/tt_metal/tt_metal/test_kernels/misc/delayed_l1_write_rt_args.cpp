// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Busy-waits a number of wall-clock cycles, then writes a value to an L1 address, so a test can keep a program in
// flight for a known time. Runtime args: address, value, delay_cycles.

#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t value = get_arg_val<uint32_t>(1);
    const uint32_t delay_cycles = get_arg_val<uint32_t>(2);

    riscv_wait(delay_cycles);

    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr) = value;
}
