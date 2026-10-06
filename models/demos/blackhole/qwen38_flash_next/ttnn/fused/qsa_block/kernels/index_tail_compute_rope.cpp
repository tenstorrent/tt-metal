// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, rope core compute (16-bit dest, the op's own): RoPE of the normalized query's first two tiles (the first
// pair), then per lane of this pair RoPE of the pooled key's first two tiles with the lane's block-start rows.
// Runtime args: 0 lane_count, 1 do_query.

#include "rope_mirror.h"
#include "index_tail_cbs.h"
#include "../../kernels/zones.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t lane_count = get_arg_val<uint32_t>(0);
    const uint32_t do_query = get_arg_val<uint32_t>(1);
    CircularBuffer(CB_SCALAR).wait_front(1);
    compute_kernel_hw_startup(CB_ROT_Q, CB_SCALAR, CB_ROT_NEG);
    if (do_query) {
        FUSED_ZONE("fz_qs_it_cr_query");
        rope64_rows<CB_IN_Q, CB_ROT_Q, CB_COS, CB_SIN, CB_SCALAR, CB_ROT_NEG, CB_XCOS, CB_RSIN, CB_OUT_Q>(1);
    }
    for (uint32_t i = 0; i < lane_count; ++i) {
        FUSED_ZONE("fz_qs_it_cr_lane");
        rope64_rows<CB_IN_K, CB_ROT_K, CB_BCOS, CB_BSIN, CB_SCALAR, CB_ROT_NEG, CB_XCOS, CB_RSIN, CB_OUT_K>(1);
    }
}
