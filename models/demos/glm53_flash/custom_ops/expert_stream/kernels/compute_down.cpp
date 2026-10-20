// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert down projection summed over the experts (TRISC) for the GLM-5.3-Flash expert stream
// (custom_ops/expert_stream/op.py): for this core's `cols` output columns,
//   out[:, c] = sum_e act[e] @ W2[e][:, c]   (act[e] = D_e @ swiglu_e: the routing-weighted activations)
// with the `cols` output tiles held in DST (fp32) across all active experts (count from the BRISC mailbox) and
// packed once. HiFi2, fp32 accumulation.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "ckernel.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_out);
    matmul_init(cb_a, cb_w);

    uint32_t n = 0;
    UNPACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    if (n == 0) {
        return;
    }

    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_a, seg_tiles);
        cb_wait_front(cb_w, cols * seg_tiles);
        for (uint32_t c = 0; c < cols; ++c) {
            for (uint32_t k = 0; k < seg_tiles; ++k) {
                matmul_tiles(cb_a, cb_w, k, c * seg_tiles + k, c);
            }
        }
        cb_pop_front(cb_a, seg_tiles);
        cb_pop_front(cb_w, cols * seg_tiles);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, cols);
    tile_regs_wait();
    for (uint32_t c = 0; c < cols; ++c) {
        pack_tile(c, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, cols);
}
