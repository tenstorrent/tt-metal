// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Union-of-experts down projection (TRISC) (fused_decode/union.py ExpertDownUnion): one DST acquire holds this core's
// `cols` output tiles; per union expert e (one weight block of cols segments: this core's columns of the expert,
// contiguous, kernels/stream_down_union_reader.cpp) every column c accumulates in0 segment e . W_e[:, c] into DST c
// (custom_mm, [rows, 32] in0 tiles, LoFi, FP32 accumulation; split accumulators merged with the last expert).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    static_assert(cols <= 4, "the output tiles of a core stay in DST (4 FP32 tiles)");
#ifdef D_NCOLS_ARG
    const uint32_t ncols = get_arg_val<uint32_t>(0);  // balanced down layout: this core's columns (<= cols)
#else
    constexpr uint32_t ncols = cols;
#endif

    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, dest_layout>(cb_in0, cb_w, cb_out);

    uint32_t num_sel = 0;
    UNPACK(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    tile_regs_acquire();
    // union expert e (processing order) from weight CB wcb, accumulated into DST 0 .. cols - 1
    auto expert = [&](uint32_t e, uint32_t wcb) {
        cb_wait_front(cb_in0, seg_tiles);  // slot e's activation (a ring: the writer pushes slot by slot)
        cb_wait_front(wcb, cols * seg_tiles);
        for (uint32_t c = 0; c < ncols; ++c) {
            if (e + 1 == num_sel) {
                custom_mm_block<dest_layout>(cb_in0, wcb, 0, c * seg_tiles, c, seg_tiles);
            } else {
                custom_mm_block<dest_layout, false>(cb_in0, wcb, 0, c * seg_tiles, c, seg_tiles);
            }
        }
        cb_pop_front(wcb, cols * seg_tiles);
        cb_pop_front(cb_in0, seg_tiles);
    };
    uint32_t e = 0;
#ifdef MOE_PRED
    // the predicted experts' blocks (MOE_PRED_CB, prefetched before the union was known): hit bit k from the writer
    uint32_t hits = 0;
    UNPACK(hits = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(hits = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(hits = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    for (uint32_t k = 0; k < MOE_PRED; ++k) {
        if ((hits >> k) & 1) {
            expert(e, MOE_PRED_CB);
            ++e;
        } else {
            cb_wait_front(MOE_PRED_CB, cols * seg_tiles);
            cb_pop_front(MOE_PRED_CB, cols * seg_tiles);
        }
    }
#endif
    for (; e < num_sel; ++e) {
        expert(e, cb_w);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t c = 0; c < ncols; ++c) {
        cb_reserve_back(cb_out, 1);
        pack_tile(c, cb_out);
        cb_push_back(cb_out, 1);
    }
    tile_regs_release();
    custom_mm_block_uninit<dest_layout>(cb_out);
}
