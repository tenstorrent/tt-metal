// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Norm core of the GDN step split 4 (QWEN38_GDN_CONV_SPLIT=4), writer: sends the 4 fp32 tiles of CB_QROW to the CB
// dest_cb (CB_QROW = 15 for a q core, CB_KROW = 16 for a k core) of every head core of the key head, data first, write
// barrier, then a SEM_NORM increment on each.  CB 15 and 16 are one CBDescriptor each over the head and norm cores, so
// the L1 address is the same on both sides.  Runtime args: 0 dest_cb, 1 count, then per destination (noc_x, noc_y).

#include "api/dataflow/dataflow_api.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_QROW = 15, SEM_NORM = 3, HT = 4;
constexpr uint32_t FP32_TILE = 4096;
}  // namespace

void kernel_main() {
    const uint32_t dest_cb = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    FUSED_ZONE("fz_gv_n_w");
    cb_wait_front(CB_QROW, HT);
    const uint32_t src = get_read_ptr(CB_QROW);
    const uint32_t dst = get_write_ptr(dest_cb);
    for (uint32_t d = 0; d < count; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(2 + 2 * d);
        const uint32_t y = get_arg_val<uint32_t>(3 + 2 * d);
        noc_async_write(src, get_noc_addr(x, y, dst), HT * FP32_TILE);
    }
    noc_async_write_barrier();
    for (uint32_t d = 0; d < count; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(2 + 2 * d);
        const uint32_t y = get_arg_val<uint32_t>(3 + 2 * d);
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SEM_NORM)), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(CB_QROW, HT);
}
