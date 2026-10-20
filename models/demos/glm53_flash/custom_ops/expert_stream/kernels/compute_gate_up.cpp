// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert gate|up matmul + clamped SwiGLU + routing weight (TRISC) for the GLM-5.3-Flash expert stream
// (custom_ops/expert_stream/op.py).
//
// For each expert of this core's group (count from the BRISC mailbox) and this bank's (gate, up) column pair:
//   g = x @ W[:, gate col]    (kt 32x32 tile products accumulated in DST tile 0, fp32)
//   u = x @ W[:, up col]      (DST tile 1)
//   h = silu(min(g, 10)) * clamp(u, -10, 10)   (the fused clamped_silu_glu SFPU op, DST tile 2)
//   out = D_e @ h             (D_e = diag(routing weights of the tokens for expert e): row t scaled by w[t, e])
// HiFi2 with fp32 accumulation, like the sparse matmul it replaces.

#include <cstdint>

#include "api/compute/clamped_silu_glu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ckernel.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_d = get_compile_time_arg_val(2);
    constexpr uint32_t cb_h = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_init(cb_x, cb_w);

    uint32_t n = 0;
    UNPACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    MATH(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
    PACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)

    cb_wait_front(cb_x, kt);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        // DST tiles 0 / 1 are accumulators that are never packed (only the SwiGLU result is), so they are not
        // cleared by the packer: zero them before every expert.
        fill_tile_init();
        fill_tile(0, 0.0f);
        fill_tile(1, 0.0f);
        matmul_init(cb_x, cb_w);
        for (uint32_t c = 0; c < 2; ++c) {  // gate -> DST 0, up -> DST 1
            cb_wait_front(cb_w, kt);
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_tiles(cb_x, cb_w, k, k, c);
            }
            cb_pop_front(cb_w, kt);
        }
        clamped_silu_glu_tile_init();
        clamped_silu_glu_tile(0, 1, 2);
        tile_regs_commit();
        cb_reserve_back(cb_h, 1);
        tile_regs_wait();
        pack_tile(2, cb_h);
        tile_regs_release();
        cb_push_back(cb_h, 1);

        // routing weights: out = D_e @ h (BF16 x BF16 operands; srcA = in1, srcB = in0 as in the matmul kernels)
        reconfig_data_format(cb_w, cb_h, cb_x, cb_d);
        matmul_init(cb_d, cb_h);
        tile_regs_acquire();
        cb_wait_front(cb_d, 1);
        cb_wait_front(cb_h, 1);
        matmul_tiles(cb_d, cb_h, 0, 0, 3);
        cb_pop_front(cb_d, 1);
        cb_pop_front(cb_h, 1);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(3, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        reconfig_data_format(cb_h, cb_w, cb_d, cb_x);
    }
    cb_pop_front(cb_x, kt);
}
