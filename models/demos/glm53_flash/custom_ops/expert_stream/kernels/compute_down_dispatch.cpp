// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down projection per pass + gathered token sums (TRISC) of the per-expert dispatch op
// (custom_ops/expert_stream/op.py, ExpertDownDispatch), for this core's `cols` output tile columns.
//
// Phase 1 (per-expert pass counts from the writer's metadata page; the expert's W2 column block stays resident):
//   h_k = act[k] @ W2[e][:, cols]   (seg_tiles products per column into DST, fp32), packed BF16 into cb_h (resident)
// Phase 2 (per token row tile, used slot count from the mailbox):
//   out[r] = sum_j G[j]   as identity-tile matmuls accumulated in fp32 DST (exact), packed once.
// HiFi4 (BF16 x BF16 products exact).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ckernel.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_h = get_compile_time_arg_val(3);
    constexpr uint32_t cb_g = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t cols = get_compile_time_arg_val(7);
    constexpr uint32_t J = get_compile_time_arg_val(8);
    constexpr uint32_t RT = get_compile_time_arg_val(9);
    constexpr uint32_t cb_m = get_compile_time_arg_val(10);

    constexpr uint32_t slot_tiles = J * cols;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_h);
    matmul_init(cb_a, cb_w);

    // ---- phase 1: per active expert its W2 block stays resident over its passes; the counts come from the
    // metadata page (read by the unpack thread and forwarded to the math / pack threads)
    CircularBuffer meta(cb_m);
    meta.wait_front(1);
    const uint32_t n_active = meta.read_tile_value(0, 0);
    for (uint32_t i = 0; i < n_active; ++i) {
        const uint32_t passes = meta.read_tile_value(0, 1 + i);
        cb_wait_front(cb_w, cols * seg_tiles);
        for (uint32_t p = 0; p < passes; ++p) {
            cb_wait_front(cb_a, seg_tiles);
            tile_regs_acquire();
            fill_tile_init();
            for (uint32_t c = 0; c < cols; ++c) {
                fill_tile(c, 0.0f);
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_w);
            matmul_init(cb_a, cb_w);
            for (uint32_t c = 0; c < cols; ++c) {
                for (uint32_t k = 0; k < seg_tiles; ++k) {
                    matmul_tiles(cb_a, cb_w, k, c * seg_tiles + k, c);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_h, cols);
            tile_regs_wait();
            pack_reconfig_data_format(cb_h);
            for (uint32_t c = 0; c < cols; ++c) {
                pack_tile(c, cb_h);
            }
            tile_regs_release();
            cb_push_back(cb_h, cols);
            cb_pop_front(cb_a, seg_tiles);
        }
        cb_pop_front(cb_w, cols * seg_tiles);
    }

    // ---- phase 2
    cb_wait_front(cb_id, 1);
    for (uint32_t r = 0; r < RT; ++r) {
        uint32_t j_max = 0;
        UNPACK(j_max = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        MATH(j_max = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        PACK(j_max = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId);)
        cb_wait_front(cb_g, slot_tiles);
        tile_regs_acquire();
        fill_tile_init();
        for (uint32_t c = 0; c < cols; ++c) {
            fill_tile(c, 0.0f);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_id, cb_g);
        matmul_init(cb_id, cb_g);
        for (uint32_t j = 0; j < j_max; ++j) {
            for (uint32_t c = 0; c < cols; ++c) {
                matmul_tiles(cb_id, cb_g, 0, j * cols + c, c);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, cols);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        for (uint32_t c = 0; c < cols; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, cols);
        cb_pop_front(cb_g, slot_tiles);
    }
}
