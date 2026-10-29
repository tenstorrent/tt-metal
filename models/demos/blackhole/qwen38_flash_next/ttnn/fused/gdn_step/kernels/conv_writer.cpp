// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Conv core of the split GDN step: sends the conv tile (CB_QKV page 0, bf16) to every head core that consumes it.  CB
// 11 is one CBDescriptor over the head and conv cores, so its L1 base is the same on both sides: the tile goes to base
// + slot * 2048 of the head core (slots: q 0-3, k 4-7, v 8-11).  Data first, write barrier, then the semaphore
// increment (the order of gdn_rows_scan's exchange).  Semaphore ids: 0 = q/k tiles (8 per head core), 1 = v tiles (4).
// Runtime args: 0 destination count, then per destination (noc_x, noc_y, slot, semaphore id).

#include "api/dataflow/dataflow_api.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_QKV = 11;
constexpr uint32_t BF16_TILE = 2048;
}  // namespace

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    FUSED_ZONE("fz_gcv_w");
#ifndef CS_NOWORK
    cb_wait_front(CB_QKV, 1);
#endif
    const uint32_t src = get_read_ptr(CB_QKV);  // page 0 of the program's only push: the CB base
    const uint32_t base = src;
    for (uint32_t d = 0; d < count; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(1 + 4 * d);
        const uint32_t y = get_arg_val<uint32_t>(2 + 4 * d);
        const uint32_t slot = get_arg_val<uint32_t>(3 + 4 * d);
#ifndef CS_NOWORK  // CS_NOWORK (dispatch-cost probe): the increments only, no data, no wait for a conv result
        noc_async_write(src, get_noc_addr(x, y, base + slot * BF16_TILE), BF16_TILE);
#endif
    }
    noc_async_write_barrier();
    for (uint32_t d = 0; d < count; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(1 + 4 * d);
        const uint32_t y = get_arg_val<uint32_t>(2 + 4 * d);
        const uint32_t sem = get_arg_val<uint32_t>(4 + 4 * d);
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem)), 1);
    }
    noc_async_atomic_barrier();
#ifndef CS_NOWORK
    cb_pop_front(CB_QKV, 1);
#endif
}
