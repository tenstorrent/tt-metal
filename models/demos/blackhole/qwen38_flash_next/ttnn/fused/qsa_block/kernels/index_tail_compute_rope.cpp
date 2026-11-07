// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, rope core compute (16-bit dest, the op's own): RoPE of the normalized query's first two tiles (first
// pair only), then per lane RoPE of the pooled key's first two tiles with the lane's block-start rows.
// Runtime args: 0 lane_count, 1 do_query.

#include "rope_mirror.h"
#include "index_tail_cbs.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t lane_count = get_arg_val<uint32_t>(0);
    const uint32_t do_query = get_arg_val<uint32_t>(1);
    CircularBuffer(cb_scalar).wait_front(1);
    compute_kernel_hw_startup(cb_rot_q, cb_scalar, cb_rot_neg);
    if (do_query) {
        rope64_rows<cb_in_q, cb_rot_q, cb_cos, cb_sin, cb_scalar, cb_rot_neg, cb_xcos, cb_rsin, cb_out_q>(1);
    }
    for (uint32_t i = 0; i < lane_count; ++i) {
        rope64_rows<cb_in_k, cb_rot_k, cb_bcos, cb_bsin, cb_scalar, cb_rot_neg, cb_xcos, cb_rsin, cb_out_k>(1);
    }
}
