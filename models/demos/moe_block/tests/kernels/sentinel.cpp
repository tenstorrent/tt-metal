// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Late-write detector: fills [target, target + words x 4) of this core with a pattern, waits `cycles`, then counts
// the words something else overwrote (a fabric write that landed after the program that owned the range ended) and
// writes the count to page `page` (64 B) of the DRAM result tensor.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/device/kernels/dram_pages.hpp"

void kernel_main() {
    const uint32_t target = get_arg_val<uint32_t>(0), words = get_arg_val<uint32_t>(1);
    const uint32_t pattern = get_arg_val<uint32_t>(2), cycles = get_arg_val<uint32_t>(3);
    const uint32_t out = get_arg_val<uint32_t>(4), page = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(target);
    for (uint32_t i = 0; i < words; ++i) {
        p[i] = pattern;
    }
    riscv_wait(cycles);
    invalidate_l1_cache();
    uint32_t bad = 0;
    for (uint32_t i = 0; i < words; ++i) {
        bad += p[i] != pattern;
    }
    p[0] = bad;
    p[1] = page;
    p[2] = 0x5E47;
    p[3] = 0;
    noc_async_write(target, dram_pages(out, 64).get_noc_addr(page), 16);
    noc_async_write_barrier();
}
