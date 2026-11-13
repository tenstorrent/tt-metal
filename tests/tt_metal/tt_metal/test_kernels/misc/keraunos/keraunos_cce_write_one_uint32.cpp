// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes one uint32 into this core's own SRAM through its uncached alias.
//
// Compile-time args:
//   0: core-local SRAM offset to write (the kernel adds MEM_L1_UNCACHED_BASE; the host never does)
//   1: value

#include "api/compile_time_args.h"
#include "dev_mem_map.h"

void kernel_main() {
    constexpr uint32_t local_offset = get_compile_time_arg_val(0);
    constexpr uint32_t value = get_compile_time_arg_val(1);
    volatile uint32_t* result = reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + local_offset);
    result[0] = value;
}
