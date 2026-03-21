// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "risc_common.h"

void kernel_main() {
    uintptr_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t value = get_common_arg_val<uint32_t>(0);

    // Write to cacheable L1 address
    *((volatile uint32_t*)dst_addr) = value;

    // Flush the cache line to TL1 (node memory) so host can read it
    flush_l2_cache_line(dst_addr);
}
