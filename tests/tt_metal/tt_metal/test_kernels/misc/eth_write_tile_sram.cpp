// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Store one word into another ethernet core's SRAM.
//
// Runtime args:
//   0: index of the core this kernel runs on. That entry is left untouched.
//   1: number of cores in the table.
//   2: byte offset added to each core's global HSIO SRAM base.
//   3: magic. Core i receives magic | i.
//   then (base_lo, base_hi) pairs, one per core, in descriptor order.
// The bases are global HSIO SRAM addresses (0x1206800000 + tile). The kernel does not contain one.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t source_index = get_arg_val<uint32_t>(0);
    const uint32_t core_count = get_arg_val<uint32_t>(1);
    const uint32_t offset = get_arg_val<uint32_t>(2);
    const uint32_t magic = get_arg_val<uint32_t>(3);
    if (source_index >= core_count) {
        __builtin_trap();
    }

    for (uint32_t core_index = 0; core_index < core_count; ++core_index) {
        if (core_index == source_index) {
            continue;
        }
        const uint32_t base_lo = get_arg_val<uint32_t>(4 + core_index * 2);
        const uint32_t base_hi = get_arg_val<uint32_t>(5 + core_index * 2);
        const uint64_t target = (static_cast<uint64_t>(base_hi) << 32) | base_lo;
        const uint64_t address = target + offset;
        if (static_cast<uint64_t>(static_cast<uintptr_t>(address)) != address) {
            __builtin_trap();
        }
        volatile uint32_t* dest = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(address));
        dest[0] = magic | core_index;
        // The peer window is outside this core's cacheable SRAM. FLUSH64 only writebacks that local
        // alias, and a flush of the global HSIO address does not return. The store itself is the
        // access the remapper delivers; the fence orders it before kernel_main returns.
        asm volatile("fence" ::: "memory");
    }
}
