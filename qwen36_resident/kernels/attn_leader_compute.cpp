// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention leader compute (see attn_common.hpp). Per attention layer and user:
//   q = rope(rmsnorm_rows(q) * w_q) into cb_q_mc (multicast);  gs = sigmoid(gate);
//   o = (the group heads' partial sums) / row sum * gs into cb_out.
//
// Runtime args: 0 children (group heads of the reduction tree).

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "attn_compute_common.hpp"

using namespace resident_attn;

namespace {

FORCE_INLINE void gate_sigmoid() {
    cb_wait_front(cb_gate, Dt);
    cb_reserve_back(cb_gs, Dt);
    reconfig_data_format(cb_gate, cb_gate);
    pack_reconfig_data_format(cb_gs);
    copy_init(cb_gate);
    sigmoid_tile_init();
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        copy_tile(cb_gate, d, 0);
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gs);
        tile_regs_release();
    }
    cb_push_back(cb_gs, Dt);
    cb_pop_front(cb_gate, Dt);
}

// cb_out = osum / rowsum * gs
FORCE_INLINE void normalize_gate() {
    cb_wait_front(cb_osum, kPart);
    cb_reserve_back(cb_rl, 1);
    reconfig_data_format(cb_osum, cb_osum);
    pack_reconfig_data_format(cb_rl);
    copy_init(cb_osum);
    tile_regs_acquire();
    copy_tile(cb_osum, Dt, 0);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rl);
    tile_regs_release();
    cb_push_back(cb_rl, 1);

    cb_wait_front(cb_rl, 1);
    cb_wait_front(cb_gs, Dt);
    cb_reserve_back(cb_out, Dt);
    reconfig_data_format(cb_osum, cb_rl);
    pack_reconfig_data_format(cb_out);
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        mul_bcast_cols_init(cb_osum, cb_rl);
        mul_tiles_bcast_cols(cb_osum, cb_rl, d, 0, 0);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gs);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gs, d, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, Dt);
    cb_pop_front(cb_osum, kPart);
    cb_pop_front(cb_rl, 1);
    cb_pop_front(cb_gs, Dt);
}

}  // namespace

void kernel_main() {
    const uint32_t children = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_qraw, cb_qraw, cb_sq);
    cb_wait_front(cb_mean, 1);
    cb_wait_front(cb_rope, 3 * batch);
    for (uint32_t l = 0; l < layers; l++) {
        // user u + 1's q and gate are prepared before user u's output, while the workers run user u
        cb_wait_front(cb_qw, Dt);
        auto prep = [&](uint32_t u) {
            norm_rope(cb_qraw, cb_qw, 0, cb_q_mc, cb_q_mc, 3 * u);
            gate_sigmoid();
            if (u + 1 == batch) {
                cb_pop_front(cb_qw, Dt);
            }
        };
        prep(0);
        for (uint32_t u = 0; u < batch; u++) {
            if (u + 1 < batch) {
                prep(u + 1);
            }
            sum_partials<false>(children);
            normalize_gate();
        }
    }
}
