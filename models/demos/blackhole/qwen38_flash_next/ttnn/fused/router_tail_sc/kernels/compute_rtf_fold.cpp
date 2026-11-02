// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// QWEN38_TOPK_PAR=1, fold core compute: NT tiles through compute_rtf_sort.cpp's RTF_DIRECT + RTF_INLINE + RTF_IDXF
// sequence (transpose the key tile, copy the index tile, +128 on faces 0 and 2, sort64 with pass_mask), the result
// tiles DST 0 (keys) and DST 2 (indices) packed as they are.
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

FORCE_INLINE void frt_load_tile(uint32_t cb_probs, uint32_t cb_index, uint32_t w, uint32_t slot) {
    transpose_init(cb_probs);
    transpose_tile(cb_probs, w, slot);
    copy_init(cb_index);
    copy_tile(cb_index, w, slot + 2);
}
FORCE_INLINE void frt_sort64(uint32_t pass_mask) {
    if (pass_mask == 0) {
        ckernel::topk_local_sort<false>(0, 0 /* largest */, 5 /* end_phase */);
    } else {
        topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask);
    }
}
constexpr uint32_t SORT_SHIFT_BITS = 0x43000000;  // 128.0f
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

void kernel_main() {
    const uint32_t pass_mask = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_keys = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_idx = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t cb_ok = get_named_compile_time_arg_val("cb_vals_t");
    constexpr uint32_t cb_oi = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t NT = get_named_compile_time_arg_val("NT");
    DataflowBuffer src(cb_keys);
    DataflowBuffer index(cb_idx);
    DataflowBuffer ok(cb_ok);
    DataflowBuffer oi(cb_oi);

    compute_kernel_hw_startup(cb_keys, cb_idx, cb_ok);
    binop_with_scalar_tile_init();
    ckernel::topk_tile_init();
    src.wait_front(NT);
    index.wait_front(NT);
    {
        FUSED_ZONE("k_fold_sort");
        tile_regs_acquire();
        for (uint32_t w = 0; w < NT; ++w) {
            frt_load_tile(cb_keys, cb_idx, w, (w == 0) ? 0 : 1);
            add_shift_faces02((w == 0) ? 0 : 1);
            if (w != 0) {
                frt_sort64(pass_mask);
            }
        }
        tile_regs_commit();
    }
    src.pop_front(NT);
    index.pop_front(NT);
    ok.reserve_back(1);
    oi.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_ok);
    pack_tile(0, cb_ok);
    pack_tile(2, cb_oi);
    tile_regs_release();
    ok.push_back(1);
    oi.push_back(1);
}
