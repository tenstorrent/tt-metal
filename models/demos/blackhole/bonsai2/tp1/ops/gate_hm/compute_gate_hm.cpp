// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute for gate_hm v2: gated = o * rsqrt(mean(o^2) + eps) * w * silu(z), one head per core, fp32 DST, on ONE
// packed tile per head (row t = channels 32t..32t+31; reader_gate_hm.cpp). v1 ran every SFPU op on Dt tiles whose
// only live row was row 0 and recomputed rinv per tile: 18.8 us against ~13 us for the chain it replaces.
// sum(o^2): sum over t of o4_t . o4_t^T, whose [r, 0] is the head's sum of squares for each packed row r; rinv is
// computed on it, packed to cb_r, and applied per row from column 0 by the FPU's column-broadcast multiply. Not
// sfpu_mul_bcast_col: that builds its broadcast by masking the row group's other lanes with 0.0 and rotate-adding, so
// stale NaN/inf in the rinv tile's other columns poisons column 0 (A, 04:53-05:00: sum and rinv right, o*w*rinv NaN).
// The FPU path unpacks column 0 only. fp32 (o) and bf16 (w, z) CBs
// alternate, so srcA's format is reconfigured before every copy (copy_init does not).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_o4 = 0, cb_pz = 1, cb_pw = 2, cb_ss = 3, cb_out = 4, cb_po = 5, cb_r = 7;

namespace {
void copy_to(uint32_t cb, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, 0, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    // Debug stage (test only): 0 = o, 1 = o*w, 2 = o*rinv*w, 3 = the full gate (default), 4 = sum(o^2), 5 = rinv
    // (4/5: packed rows 0..Dt-1, columns 0..Dt-1 carry the value).
    constexpr uint32_t stage = get_compile_time_arg_val(1);
    const uint32_t inv_d_bits = get_arg_val<uint32_t>(0);  // 1/D as fp32 bits
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_o4, cb_o4, cb_ss);
    CircularBuffer c4(cb_o4), cpo(cb_po), cpz(cb_pz), cpw(cb_pw), css(cb_ss), cr(cb_r), cout(cb_out);
    c4.wait_front(Dt);
    cpo.wait_front(1);
    cpz.wait_front(1);
    cpw.wait_front(1);

    css.reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format(cb_o4, cb_o4);
    matmul_init(cb_o4, cb_o4, 1);
    for (uint32_t t = 0; t < Dt; t++) {
        matmul_tiles(cb_o4, cb_o4, t, t, 0);  // [r, 0] += o_t . o_t
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_ss);
    tile_regs_release();
    css.push_back(1);
    css.wait_front(1);

    // rinv (stage 4: the raw sum), column 0 of rows 0..Dt-1
    cr.reserve_back(1);
    tile_regs_acquire();
    copy_to(cb_ss, 0);
    if constexpr (stage != 4) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_d_bits);
        add_unary_tile(0, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r);
    tile_regs_release();
    cr.push_back(1);
    cr.wait_front(1);

    cout.reserve_back(1);
    tile_regs_acquire();
    if constexpr (stage >= 4) {
        copy_to(cb_r, 1);
    } else if constexpr (stage >= 2) {
        reconfig_data_format(cb_po, cb_r);  // o -> srcA, rinv -> srcB
        mul_bcast_cols_init(cb_po, cb_r);
        mul_tiles_bcast_cols(cb_po, cb_r, 0, 0, 1);  // o * rinv, per row from column 0
    } else {
        copy_to(cb_po, 1);
    }
    if constexpr (stage >= 1 && stage <= 3) {
        copy_to(cb_pw, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);  // * w
    }
    if constexpr (stage == 3) {
        copy_to(cb_pz, 2);
        silu_tile_init();
        silu_tile(2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);  // * silu(z)
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(1, cb_out);
    tile_regs_release();
    cout.push_back(1);
    c4.pop_front(Dt);
    cpo.pop_front(1);
    cpz.pop_front(1);
    cpw.pop_front(1);
    css.pop_front(1);
    cr.pop_front(1);
}
