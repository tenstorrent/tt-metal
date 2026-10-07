// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

void kernel_main() {
    constexpr uint32_t cached_write = get_arg(args::cached_write);
    uintptr_t dst_addr = get_arg(args::address) + (cached_write ? 0 : MEM_L1_UNCACHED_BASE);
    uint32_t value = get_arg(args::value);

    // Write to cacheable L1 address
    CoreLocalMem<uint32_t> buffer(dst_addr);
    buffer[0] = value;

    if constexpr (cached_write) {
        // Flush the cache line to TL1 (node memory) so host can read it
        flush_l2_cache_line(dst_addr);
    }
}
