// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail, fast sort-core form (QWEN38_ROUTER_TAIL_FAST=1), compute of core B (RTF_INCR / RTF_DIRECT) (rows == 1):
// today's insertion sort, on the logits (plus 128, see SORT_SHIFT_BITS), concurrently with core A's softmax.  Softmax
// is strictly monotone and keeps equal values equal, so every compare of the network (and every tie) comes out as it
// does on the probabilities (x18: the indices of the raw sort equal today's on 259 / 259 rows).  The logits are copied
// through the srcA / fp32-dest path the softmax's sub takes into cb_probs (x18's copy pass), shifted, then
// frt_load_tile / frt_sort64 run VERBATIM from ../../router_tail/kernels/compute_router_tail.cpp: the same 16 loads,
// the same 15 sorts of 64 in the same fold order, pass_mask 1 (token 0).  Only the transposed index tile is packed
// (cb_idx_t); the reader sends it to core A.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/topk.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../router_tail/kernels/topk_lanes.h"
#include "../../kernels/zones.h"

#ifndef RTF_IDXF
#define RTF_IDXF \
    0  // the index tiles are fp32 (exact integers), unpacked to DST like the logits: no srcA reconfig per tile
#endif

// One tile of the insertion chain into DST: compute_router_tail.cpp's, minus the srcA reconfigs when RTF_IDXF.
FORCE_INLINE void frt_load_tile(uint32_t cb_probs, uint32_t cb_index, uint32_t w, uint32_t slot) {
#if !RTF_IDXF
    reconfig_data_format_srca(cb_probs);
#endif
    transpose_init(cb_probs);
    transpose_tile(cb_probs, w, slot);
#if !RTF_IDXF
    reconfig_data_format_srca(cb_index);
#endif
    copy_init(cb_index);
    copy_tile(cb_index, w, slot + 2);
}

// verbatim from compute_router_tail.cpp
FORCE_INLINE void frt_sort64(uint32_t pass_mask) {
    if (pass_mask == 0) {
        ckernel::topk_local_sort<false>(0, 0 /* largest */, 5 /* end_phase */);
    } else {
        topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask);
    }
}

// The SFPSWAP's tie decision on equal NEGATIVE values differs from its decision on equal positive ones (device probe
// x36: tied negative logits come out in another order than the tied, positive, probabilities).  So the sort keys are
// logit + 128 (exact for bf16 logits of magnitude >= 2^-9, strictly monotone, ties kept), always positive for
// logits > -128, as the probabilities are.
constexpr uint32_t SORT_SHIFT_BITS = 0x43000000;  // 128.0f

// add_unary_tile on faces 0 and 2 only (VectorMode::C): the transposed tile's columns 0..15 hold tokens 0..15, and
// the sort of token 0 never reads the columns 16..31 (faces 1 and 3).
ALWI void add_shift_faces02(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ckernel::ADD_UNARY, 8 /* ITERATIONS */, DST_ACCUM_MODE),
        idst,
        VectorMode::C,
        SORT_SHIFT_BITS));
}

#ifndef RTF_INCR
#define RTF_INCR 0
#endif
#ifndef RTF_INLINE
#define RTF_INLINE 0  // with RTF_DIRECT: the +128 shift is applied to each transposed tile in DST, in the sort loop
#endif
#ifndef RTF_DIRECT
#define RTF_DIRECT 0  // timing probe only: sort cb_in0 as is (no +128, wrong for negative ties)
#endif

void kernel_main() {
    const uint32_t pass_mask = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_probs = get_named_compile_time_arg_val("cb_probs");
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t cb_vals_t = get_named_compile_time_arg_val("cb_vals_t");
    constexpr uint32_t cb_idx_t = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t ndst = 4;  // the copy pass's block size under fp32 dest
    static_assert(Wt % ndst == 0, "the copy blocks must tile the row");

    DataflowBuffer in0(cb_in0);
    DataflowBuffer probs(cb_probs);
    DataflowBuffer index(cb_index);
    DataflowBuffer idx_t(cb_idx_t);

#if !RTF_DIRECT
    {
        // x18's raw-logit copy pass (compute_router_tail.cpp, FRT_RAWSORT): the logits through srcA into DST, packed
        // fp32 into cb_probs
        FUSED_ZONE("fz_rt_c_softmax");
        compute_kernel_hw_startup(cb_in0, cb_in0, cb_probs);
        reconfig_data_format_srca(cb_in0);
        pack_reconfig_data_format(cb_probs);
        copy_init(cb_in0);
        binop_with_scalar_tile_init();
        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
#if RTF_INCR
            in0.wait_front(wt < 4 ? 4 : Wt);
#else
            if (wt == 0) {
                in0.wait_front(Wt);
            }
#endif
            tile_regs_acquire();
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                copy_tile(cb_in0, wt + wt8, wt8);
            }
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                add_unary_tile(wt8, SORT_SHIFT_BITS);  // logit + 128: every key positive, like the probabilities
            }
            probs.reserve_back(ndst);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t wt8 = 0; wt8 < ndst; wt8++) {
                pack_tile(wt8, cb_probs);
            }
            tile_regs_release();
            probs.push_back(ndst);
        }
        in0.pop_front(Wt);
    }

#endif

    // ---- today's insertion sort (topk.cpp single core, Wt tiles, largest, stable_sort false) ----
#if RTF_DIRECT
    constexpr uint32_t cb_src = cb_in0;
#else
    constexpr uint32_t cb_src = cb_probs;
#endif
    compute_kernel_hw_startup(cb_src, cb_index, cb_vals_t);
#if RTF_INLINE
    binop_with_scalar_tile_init();
#endif
    ckernel::topk_tile_init();
    DataflowBuffer src(cb_src);
    src.wait_front(Wt);
    index.wait_front(Wt);
    {
        FUSED_ZONE("fz_rt_c_topk_sort");
        tile_regs_acquire();
        for (uint32_t w = 0; w < Wt; ++w) {
            frt_load_tile(cb_src, cb_index, w, (w == 0) ? 0 : 1);
#if RTF_INLINE
            add_shift_faces02((w == 0) ? 0 : 1);
#endif
            if (w != 0) {
                frt_sort64(pass_mask);
            }
        }
        tile_regs_commit();
    }
    src.pop_front(Wt);
    index.pop_front(Wt);
    idx_t.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_idx_t);
    pack_tile(2, cb_idx_t);
    tile_regs_release();
    idx_t.push_back(1);
}
