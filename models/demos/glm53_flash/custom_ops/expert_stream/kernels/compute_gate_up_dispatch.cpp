// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tilize + gate|up matmul + clamped SwiGLU + routing weight (TRISC) of the per-expert dispatch op
// (custom_ops/expert_stream/op.py, ExpertGateUpDispatch).
//
// Per expert of this core (pass count from the BRISC mailbox, 0 = done) the bank's (gate, up) column pair stays
// resident over the expert's passes; per pass the writer's 32-row row-major block is tilized into the [32, K]
// activation tiles, then as compute_gate_up.cpp: g -> DST 0, u -> DST 1, clamped_silu_glu -> DST 2, D @ h -> DST 3.
// HiFi2, fp32 accumulation.

#include <cstdint>

#include "api/compute/clamped_silu_glu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "ckernel.h"

void kernel_main() {
    constexpr uint32_t cb_xrm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_d = get_compile_time_arg_val(3);
    constexpr uint32_t cb_h = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t kt = get_compile_time_arg_val(6);
    constexpr uint32_t ppb = get_compile_time_arg_val(7);  // column pairs per bank

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    matmul_init(cb_x, cb_w);

    while (true) {
        uint32_t passes = 0;
        UNPACK(passes = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(passes = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(passes = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        if (passes == 0) {
            break;
        }
        if constexpr (ppb == 1) {
            cb_wait_front(cb_w, 2 * kt);  // the pair, resident over the passes: gate tiles 0..kt-1, up kt..2kt-1
        }
        for (uint32_t p = 0; p < passes; ++p) {
            // the pass's token rows -> tiles
            cb_wait_front(cb_xrm, kt);
            cb_reserve_back(cb_x, kt);
            reconfig_data_format_srca(cb_xrm);
            pack_reconfig_data_format(cb_x);
            tilize_init(cb_xrm, kt, cb_x);
            tilize_block(cb_xrm, kt, cb_x);
            tilize_uninit(cb_xrm, cb_x);
            cb_push_back(cb_x, kt);
            cb_pop_front(cb_xrm, kt);
            cb_wait_front(cb_x, kt);

            cb_wait_front(cb_d, 1);  // the pass's routing weights, shared by the column pairs
            for (uint32_t q = 0; q < ppb; ++q) {
                if constexpr (ppb > 1) {
                    cb_wait_front(cb_w, 2 * kt);  // this pair's gate and up columns (streamed per pass)
                }
                tile_regs_acquire();
                fill_tile_init();
                fill_tile(0, 0.0f);
                fill_tile(1, 0.0f);
                reconfig_data_format<SrcOrder::Reverse>(cb_x, cb_w);
                matmul_init(cb_x, cb_w);
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_tiles(cb_x, cb_w, k, k, 0);
                }
                for (uint32_t k = 0; k < kt; ++k) {
                    matmul_tiles(cb_x, cb_w, k, kt + k, 1);
                }
                clamped_silu_glu_tile_init();
                clamped_silu_glu_tile(0, 1, 2);
                tile_regs_commit();
                cb_reserve_back(cb_h, 1);
                tile_regs_wait();
                pack_reconfig_data_format(cb_h);
                pack_tile(2, cb_h);
                tile_regs_release();
                cb_push_back(cb_h, 1);
                if constexpr (ppb > 1) {
                    cb_pop_front(cb_w, 2 * kt);
                }

                // routing weights: out = D @ h
                reconfig_data_format<SrcOrder::Reverse>(cb_d, cb_h);
                matmul_init(cb_d, cb_h);
                tile_regs_acquire();
                cb_wait_front(cb_h, 1);
                matmul_tiles(cb_d, cb_h, 0, 0, 3);
                cb_pop_front(cb_h, 1);
                tile_regs_commit();
                cb_reserve_back(cb_out, 1);
                tile_regs_wait();
                pack_reconfig_data_format(cb_out);
                pack_tile(3, cb_out);
                tile_regs_release();
                cb_push_back(cb_out, 1);
            }
            cb_pop_front(cb_d, 1);
            cb_pop_front(cb_x, kt);
        }
        if constexpr (ppb == 1) {
            cb_pop_front(cb_w, 2 * kt);
        }
    }
}
