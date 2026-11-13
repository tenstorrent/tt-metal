// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Word-by-word CPU copy on a Keraunos CCE hart: volatile 32-bit loads from src, stores to dst, then a
// fence. Either side may be a core-local SRAM offset or a global (HSIO / D2D) address, so the same kernel
// probes CPU reads and CPU writes on every path. A core-local offset is accessed through this core's
// uncached alias (MEM_L1_UNCACHED_BASE), which the kernel adds; the host never does.
//
// Status words (uint32, uncached alias): 0 started, 1 index of the next word to load (so a hang shows how
// far the loads got), 4 done.
//
// Runtime args:
//   0, 1: source address, low and high 32 bits
//   2, 3: destination address, low and high 32 bits
//   4:    number of 32-bit words
//   5:    status block, local SRAM offset
//   6:    flags: bit 0 = source is a core-local offset, bit 1 = destination is a core-local offset

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

namespace {
uint64_t arg_u64(uint32_t index) {
    return (static_cast<uint64_t>(get_arg_val<uint32_t>(index + 1)) << 32) | get_arg_val<uint32_t>(index);
}

volatile uint32_t* resolve(uint64_t addr, bool is_local) {
    return reinterpret_cast<volatile uint32_t*>(is_local ? MEM_L1_UNCACHED_BASE + addr : addr);
}
}  // namespace

void kernel_main() {
    const uint32_t flags = get_arg_val<uint32_t>(6);
    volatile uint32_t* src = resolve(arg_u64(0), flags & 1);
    volatile uint32_t* dst = resolve(arg_u64(2), flags & 2);
    const uint32_t words = get_arg_val<uint32_t>(4);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(5));

    status[0] = 0x1D3A0001;
    for (uint32_t i = 0; i < words; i++) {
        status[1] = i;
        dst[i] = src[i];
    }
    asm volatile("fence rw, rw" ::: "memory");
    status[4] = 0x1D3A0003;
}
