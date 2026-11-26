// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention leader compute (see attn_common.hpp). Per attention layer and user:
//   q = rope(rmsnorm_rows(q) * w_q) into cb_q_mc (multicast);  gs = sigmoid(gate);
//   o = (the group heads' partial sums) / row sum * gs into cb_out.
//
// Runtime args: 0 children (group heads of the reduction tree).
// Rows mode (active_rows.hpp): only the active users run (the reader's mask in cb_lead_active).

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "attn_compute_common.hpp"
#include "active_rows.hpp"

using namespace resident_attn;

namespace {

FORCE_INLINE void gate_sigmoid(uint32_t out_cb) {
    cb_wait_front(cb_gate, Dt);
    cb_reserve_back(out_cb, Dt);
    reconfig_data_format(cb_gate, cb_gate);
    pack_reconfig_data_format(out_cb);
    copy_init(cb_gate);
    sigmoid_tile_init();
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        copy_tile(cb_gate, d, 0);
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_push_back(out_cb, Dt);
    cb_pop_front(cb_gate, Dt);
}

// cb_out = osum / rowsum * gs (S0-9 kVRot: osum / rowsum rotated back by HAD, then * gs)
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
    if constexpr (kVRotPass) {
        // o = osum / rowsum into cb_vr1, for the reader to place into its pass's rows (pass_out rotates and gates)
        cb_reserve_back(cb_vr1, Dt);
        reconfig_data_format(cb_osum, cb_rl);
        pack_reconfig_data_format(cb_vr1);
        mul_bcast_cols_init(cb_osum, cb_rl);
        for (uint32_t d = 0; d < Dt; d++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_osum, cb_rl, d, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_vr1);
            tile_regs_release();
        }
        cb_push_back(cb_vr1, Dt);
        cb_pop_front(cb_osum, kPart);
        cb_pop_front(cb_rl, 1);
        return;
    }
    cb_wait_front(cb_gs, Dt);
    if constexpr (kVRot) {
        // o = osum / rowsum into cb_vr1, rotated back (V is stored rotated), then * gs into cb_out
        cb_reserve_back(cb_vr1, Dt);
        reconfig_data_format(cb_osum, cb_rl);
        pack_reconfig_data_format(cb_vr1);
        mul_bcast_cols_init(cb_osum, cb_rl);
        for (uint32_t d = 0; d < Dt; d++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_osum, cb_rl, d, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_vr1);
            tile_regs_release();
        }
        cb_push_back(cb_vr1, Dt);
        cb_pop_front(cb_osum, kPart);
        cb_pop_front(cb_rl, 1);
        had_mm_stage(cb_vr1, cb_vr2);
        had_bfly_stage(cb_vr1, cb_vr1, cb_vr2, cb_vr1);
        cb_wait_front(cb_vr1, Dt);
        cb_reserve_back(cb_out, Dt);
        reconfig_data_format(cb_vr1, cb_gs);
        pack_reconfig_data_format(cb_out);
        mul_init(cb_vr1, cb_gs);
        for (uint32_t d = 0; d < Dt; d++) {
            tile_regs_acquire();
            mul_tiles(cb_vr1, cb_gs, d, d, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, Dt);
        cb_pop_front(cb_vr1, Dt);
        cb_pop_front(cb_gs, Dt);
        return;
    }
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

// S0-9 2d (kVRotPass): a pass's normalized outputs (cb_vp, placed by the reader) rotated back by HAD, then * the pass's
// sigmoid(gate) into cb_out
FORCE_INLINE void pass_out() {
    had_mm_stage(cb_vp, cb_vr2);
    had_bfly_stage(cb_rotx, cb_rotx, cb_vr2, cb_rot2);
    cb_wait_front(cb_rotx, Dt);
    cb_wait_front(cb_gs_pass, Dt);
    cb_reserve_back(cb_out, Dt);
    reconfig_data_format(cb_rotx, cb_gs_pass);
    pack_reconfig_data_format(cb_out);
    mul_init(cb_rotx, cb_gs_pass);
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        mul_tiles(cb_rotx, cb_gs_pass, d, d, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, Dt);
    cb_pop_front(cb_rotx, Dt);
    cb_pop_front(cb_gs_pass, Dt);
}

}  // namespace

void kernel_main() {
    const uint32_t children = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_qraw, cb_qraw, cb_sq);
    cb_wait_front(cb_mean, 1);
    if constexpr (kKRot) {
        cb_wait_front(cb_had, 1);
    }
    cb_wait_front(cb_rope, kRopeLead);
    uint32_t mask = ~0u;
    if constexpr (rows_mode) {
        constexpr uint32_t cb_lead_active = 7;
        cb_wait_front(cb_lead_active, 1);
        mask = read_tile_value(cb_lead_active, 0, kTokActive);
        cb_pop_front(cb_lead_active, 1);
    }
    // (row groups: global rows, rope pages 3 u of the token state's; verify rows: a bit per pass)
    const ActiveRows<kUsers> act(pass_mask(mask));
    const uint32_t n_act = act.n;
    for (uint32_t l = 0; l < layers && n_act > 0; l++) {
        // the next user's q and gate are prepared before this user's output, while the workers run this user
        cb_wait_front(cb_qw, Dt);
        [[maybe_unused]] auto prep = [&](uint32_t k) __attribute__((noinline)) {
            const uint32_t u = act.row[k];
            if constexpr (kLeadBatched) {
                // the pass of active users k .. k + kQPerPass - 1 at once (the writer takes each user's rows)
                if (k % kQPerPass == 0) {
                    norm_rope(cb_qraw, cb_qw, 0, cb_q_mc, cb_q_mc, kRopeQPass + 3 * (k / kQPerPass));
                }
            } else {
                norm_rope(cb_qraw, cb_qw, 0, cb_q_mc, cb_q_mc, packed ? kRopeQ + 3 * u : 3 * u);
            }
            if constexpr (kGatePass) {
                if (k % kQPerPass == 0) {
                    gate_sigmoid(cb_gs_pass);  // (the pass's users; the reader extracts each one's rows)
                }
            } else {
                gate_sigmoid(cb_gs);
            }
            if (k + 1 == n_act) {
                cb_pop_front(cb_qw, Dt);
            }
        };
        if constexpr (kLeadStaged) {
            // (O-2b, attn_common.hpp kLeadStaged: pass p + 1's prep in four stages before pass p's users 0 .. 3)
            const uint32_t n_pass = (n_act + kQPerPass - 1) / kQPerPass;
            auto stage = [&](uint32_t p, uint32_t s) __attribute__((noinline)) {
                if (s == 0) {
                    norm_stage(cb_qraw, cb_qw, 0);
                    rope_stage(cb_q_mc, cb_q_mc, kRopeQPass + 3 * p);
                    if (p + 1 == n_pass) {
                        cb_pop_front(cb_qw, Dt);
                    }
                } else if (s == 1) {
                    had_mm_stage();
                } else if (s == 2) {
                    had_bfly_stage(cb_q_mc, cb_q_mc);
                } else {
                    gate_sigmoid(cb_gs_pass);  // (the pass's users; the reader extracts each one's rows)
                }
            };
            for (uint32_t s = 0; s < 4; s++) {
                stage(0, s);
            }
            for (uint32_t k = 0; k < n_act; k++) {
                const uint32_t p = k / kQPerPass, j = k % kQPerPass;
                if (j < 4 && p + 1 < n_pass) {
                    stage(p + 1, j);
                }
                sum_partials<false>(children);
                if constexpr (kVRotPass) {
                    cb_reserve_back(cb_vdone, 1);  // (the partials are summed: their slots may be rewritten)
                    cb_push_back(cb_vdone, 1);
                }
                normalize_gate();
                if (kVRotPass && (j + 1 == kQPerPass || k + 1 == n_act)) {
                    pass_out();
                }
            }
        } else {
            // (O-2: q two users ahead, as the writer sends it)
            constexpr uint32_t ahead = kAttnPipe ? 2 : 1;
            for (uint32_t k = 0; k < ahead && k < n_act; k++) {
                prep(k);
            }
            for (uint32_t k = 0; k < n_act; k++) {
                if (k + ahead < n_act) {
                    prep(k + ahead);
                }
                sum_partials<false>(children);
                if constexpr (kVRotPass) {
                    cb_reserve_back(cb_vdone, 1);
                    cb_push_back(cb_vdone, 1);
                }
                normalize_gate();
                if (kVRotPass && (k % kQPerPass == kQPerPass - 1 || k + 1 == n_act)) {
                    pass_out();
                }
            }
        }
    }
}
