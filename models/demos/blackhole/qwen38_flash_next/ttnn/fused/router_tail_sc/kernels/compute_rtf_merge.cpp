// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// QWEN38_TOPK_PAR=1, core B compute: instead of folding 16 tiles, merges the NL partial top-32 lists the fold cores
// sent (key tile in cb_mk, index tile in cb_mi, in DST layout): list 0 into DST 0 / 2, lists 1.. into DST 1 / 3, each
// followed by frt_sort64 (compute_rtf_sort.cpp's, verbatim).  DST 0 / 2 then hold what the old fold leaves, and the
// index tile (DST 2) is packed into cb_idx_t exactly as compute_rtf_sort.cpp does.
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

FORCE_INLINE void frt_sort64(uint32_t pass_mask) {
    if (pass_mask == 0) {
        ckernel::topk_local_sort<false>(0, 0 /* largest */, 5 /* end_phase */);
    } else {
        topk_local_sort_lanes<false>(0, 0 /* largest */, 5 /* end_phase */, pass_mask);
    }
}

void kernel_main() {
    const uint32_t pass_mask = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_mk = get_named_compile_time_arg_val("cb_mk");
    constexpr uint32_t cb_mi = get_named_compile_time_arg_val("cb_mi");
    constexpr uint32_t cb_vals_t = get_named_compile_time_arg_val("cb_vals_t");
    constexpr uint32_t cb_idx_t = get_named_compile_time_arg_val("cb_idx_t");
    constexpr uint32_t NL = get_named_compile_time_arg_val("NL");
    DataflowBuffer mk(cb_mk);
    DataflowBuffer mi(cb_mi);
    DataflowBuffer idx_t(cb_idx_t);

    compute_kernel_hw_startup(cb_mk, cb_mi, cb_vals_t);
    ckernel::topk_tile_init();
    mk.wait_front(NL);
    mi.wait_front(NL);
    {
        FUSED_ZONE("fz_rt_c_topk_merge");
        tile_regs_acquire();
        for (uint32_t j = 0; j < NL; ++j) {
            const uint32_t slot = (j == 0) ? 0 : 1;
            copy_init(cb_mk);
            copy_tile(cb_mk, j, slot);
            copy_init(cb_mi);
            copy_tile(cb_mi, j, slot + 2);
            if (j != 0) {
                frt_sort64(pass_mask);
            }
        }
        tile_regs_commit();
    }
    mk.pop_front(NL);
    mi.pop_front(NL);
    idx_t.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_idx_t);
    pack_tile(2, cb_idx_t);
    tile_regs_release();
    idx_t.push_back(1);
}
