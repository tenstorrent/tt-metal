// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Busy-waits `delay_cycles` wall-clock cycles, then writes `value` to `address`.
// Lets a test keep a program in flight for a known time.

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

void kernel_main() {
    uintptr_t dst_addr = get_arg(args::address);
    uint32_t value = get_arg(args::value);
    uint32_t delay_cycles = get_arg(args::delay_cycles);

    riscv_wait(delay_cycles);

    CoreLocalMem<uint32_t> buffer(dst_addr);
    buffer[0] = value;
    flush_l2_cache_line(dst_addr);
}
