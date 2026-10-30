// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Gate core of the GDN step split 4 (QWEN38_GDN_CONV_SPLIT=4), writer: sends CB_BETA and CB_DECAY (one fp32 tile each)
// to the same CBs of every block core of the head, data first, write barrier, then one SEM_GATE increment on each.  CB
// 18 and 19 are one CBDescriptor each over the head and gate cores (the same L1 address on both sides).  Runtime args:
// 0 count, then per destination (noc_x, noc_y).

#include "api/dataflow/dataflow_api.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_BETA = 18, CB_DECAY = 19, SEM_GATE = 4;
constexpr uint32_t FP32_TILE = 4096;
}  // namespace

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    FUSED_ZONE("fz_gv_g_w");
    cb_wait_front(CB_BETA, 1);
    cb_wait_front(CB_DECAY, 1);
    const uint32_t beta = get_read_ptr(CB_BETA);
    const uint32_t decay = get_read_ptr(CB_DECAY);
    for (uint32_t d = 0; d < count; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(1 + 2 * d);
        const uint32_t y = get_arg_val<uint32_t>(2 + 2 * d);
        noc_async_write(beta, get_noc_addr(x, y, beta), FP32_TILE);
        noc_async_write(decay, get_noc_addr(x, y, decay), FP32_TILE);
    }
    noc_async_write_barrier();
    for (uint32_t d = 0; d < count; ++d) {
        const uint32_t x = get_arg_val<uint32_t>(1 + 2 * d);
        const uint32_t y = get_arg_val<uint32_t>(2 + 2 * d);
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SEM_GATE)), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(CB_BETA, 1);
    cb_pop_front(CB_DECAY, 1);
}
