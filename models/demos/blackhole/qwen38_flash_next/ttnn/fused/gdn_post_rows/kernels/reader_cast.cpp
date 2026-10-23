// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_cast`` reader: the scan's fp32 output ``o`` [12, T, 128] in 4-tile groups.  Group g of this core covers the
// four column tiles of one (value head, tile row): pages (start + g) * 4 .. + 3 of the head-major page order
// (page (h * NC + c) * 4 + d), so a group is four consecutive pages and a core's run is one contiguous read.
// CBs: CB_IN (0, fp32, 4-tile groups).
// Compile-time args: TensorAccessorArgs of o from 0.  Runtime args: 0 o address, 1 groups on this core, 2 first group.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_IN = 0;
constexpr uint32_t GROUP = 4;  // 128 / 32 column tiles of one head
constexpr uint32_t FP32_TILE = 4096;
}  // namespace

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    uint32_t arg = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t groups = get_arg_val<uint32_t>(arg++);
    const uint32_t first = get_arg_val<uint32_t>(arg++);
    const auto o = TensorAccessor(o_args, o_addr);

    for (uint32_t g = 0; g < groups; ++g) {
        FUSED_ZONE("fz_gpo_rc_group");
        const uint32_t page = (first + g) * GROUP;
        cb_reserve_back(CB_IN, GROUP);
        const uint32_t l1 = get_write_ptr(CB_IN);
        for (uint32_t d = 0; d < GROUP; ++d) {
            noc_async_read_page(page + d, o, l1 + d * FP32_TILE);
        }
        noc_async_read_barrier();
        cb_push_back(CB_IN, GROUP);
    }
}
