// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC kernel that issues NoC reads: copies one page from each listed source into its own L1, back
// to back from dst_l1_base, where the host reads them back.
//
// Compile-time args:
//   [0] dst_l1_base - DRISC L1 address the first page lands at
//   [1] page_bytes  - bytes per source page
//
// Runtime args: num_sources, then (virtual noc x, virtual noc y, address) per source.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// DRISC firmware does not define cb_interface (no CB infrastructure on DRAM cores), and
// dataflow_api.h references it.
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t dst_l1_base = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t kArgsPerSource = 3;

    const uint32_t num_sources = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < num_sources; ++i) {
        const uint32_t x = get_arg_val<uint32_t>(1 + i * kArgsPerSource);
        const uint32_t y = get_arg_val<uint32_t>(2 + i * kArgsPerSource);
        const uint32_t addr = get_arg_val<uint32_t>(3 + i * kArgsPerSource);
        const uint32_t noc_xy = uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc_index, x), DYNAMIC_NOC_Y(noc_index, y)));
        noc_async_read(get_noc_addr_helper(noc_xy, addr), dst_l1_base + i * page_bytes, page_bytes, noc_index);
    }
    noc_async_read_barrier(noc_index);
}
