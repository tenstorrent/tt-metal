// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill causal conv, compute (shift first, then multiply). Token r of a 32-token tile needs
// x[r - 3 + k] * w_k for k = 0..3. DEST 0 = x (I @ x), DEST s = S_s @ x + P_s @ prev (Q_s @ state for the first
// tile), s = 1..3: exact row shifts of the bf16 input (0/1 matrices, HiFi2 covers bf16). The SFPU then forms
// sum_s DEST s * w_(3-s) in fp32 (bf16 x bf16 products are exact) with the taps as lane vectors (the column tile's
// vector tile in DEST 7: tap k's vector v = 2 fc + par at SFPU row 4 k + v) and applies SiLU. New state = R @ last
// tile.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "../../ternary/kernels/core_span.hpp"

namespace {
constexpr uint32_t cb_x = 0, cb_s = 1, cb_w = 2, cb_c = 3, cb_p = 6, cb_l = 7, cb_out = 16, cb_ns = 17;
constexpr uint32_t I0 = 0, S1 = 1, P1 = 4, Q1 = 7, R = 10;
}  // namespace

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// rows q with 2 ((q / 8) & 1) + (q & 1) == V: DEST 0 row = sum_s DEST s row * w_(3 - s)
template <uint32_t V>
inline void conv_rows() {
    constexpr uint32_t T = 32, fc = V >> 1, par = V & 1;
    const sfpi::vFloat w0 = sfpi::dst_reg[7 * T + 0 + V];
    const sfpi::vFloat w1 = sfpi::dst_reg[7 * T + 4 + V];
    const sfpi::vFloat w2 = sfpi::dst_reg[7 * T + 8 + V];
    const sfpi::vFloat w3 = sfpi::dst_reg[7 * T + 12 + V];
#pragma GCC unroll 8
    for (uint32_t j = 0; j < 8; ++j) {
        // the 8 rows: faces f with f & 1 == fc (f = fc, fc + 2), rows of parity par
        const uint32_t f = fc + 2 * (j / 4), q = 8 * f + 2 * (j % 4) + par;
        const sfpi::vFloat a = sfpi::dst_reg[q];
        const sfpi::vFloat b = sfpi::dst_reg[T + q];
        const sfpi::vFloat c = sfpi::dst_reg[2 * T + q];
        const sfpi::vFloat d = sfpi::dst_reg[3 * T + q];
        sfpi::dst_reg[q] = a * w3 + b * w2 + c * w1 + d * w0;
    }
}
inline void conv_mac() {
    conv_rows<0>();
    conv_rows<1>();
    conv_rows<2>();
    conv_rows<3>();
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    // common args: Dt, ncores, grid_x, Tt. Work units (column ct, time tile tt), ct-major, split evenly over the cores:
    // a core's units form segments of consecutive time tiles of one column.
    const uint32_t Dt = get_common_arg_val<uint32_t>(0);
    const uint32_t Tt = get_common_arg_val<uint32_t>(3);
    uint32_t first, n;
    core_span(Dt * Tt, get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), 1, first, n);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_c, cb_x, cb_out);
    cb_wait_front(cb_c, 11);
    for (uint32_t u = first; u < first + n;) {
        const uint32_t tt0 = u % Tt, seg = (Tt - tt0) < (first + n - u) ? (Tt - tt0) : (first + n - u);
        cb_wait_front(cb_w, 1);
        if (tt0 == 0) {
            cb_wait_front(cb_s, 1);
        }
        for (uint32_t tt = tt0; tt < tt0 + seg; ++tt) {
            cb_wait_front(cb_x, 1);  // cur
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_c, cb_x);
            matmul_init(cb_c, cb_x);
            matmul_tiles(cb_c, cb_x, I0, 0, 0);
            for (uint32_t s = 1; s <= 3; ++s) {
                matmul_tiles(cb_c, cb_x, S1 + s - 1, 0, s);
            }
            // the 3 tokens before the tile: the conv state (first tile) or the previous tile (c_6, from the reader)
            const uint32_t pcb = tt == 0 ? cb_s : cb_p;
            if (tt != 0) {
                cb_wait_front(cb_p, 1);
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_c, pcb);
            matmul_init(cb_c, pcb);
            for (uint32_t s = 1; s <= 3; ++s) {
                matmul_tiles(cb_c, pcb, (tt == 0 ? Q1 : P1) + s - 1, 0, s);
            }
            if (tt == 0) {  // new state rows 0-2 = the column's last tile's rows 29-31 (c_7)
                cb_wait_front(cb_l, 1);
                reconfig_data_format<SrcOrder::Reverse>(cb_c, cb_l);
                matmul_init(cb_c, cb_l);
                matmul_tiles(cb_c, cb_l, R, 0, 4);
            }
            reconfig_data_format_srca(cb_w);
            copy_init(cb_w);
            copy_tile(cb_w, 0, 7);
            MATH(({
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
                TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                ckernel::sfpu::conv_mac();
                math::clear_dst_reg_addr();
            }));
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            if (tt == 0) {
                cb_reserve_back(cb_ns, 1);
                pack_tile(4, cb_ns);
                cb_push_back(cb_ns, 1);
            }
            tile_regs_release();
            cb_pop_front(cb_x, 1);
            if (tt != 0) {
                cb_pop_front(cb_p, 1);
            }
        }
        cb_pop_front(cb_w, 1);
        if (tt0 == 0) {
            cb_pop_front(cb_s, 1);
            cb_pop_front(cb_l, 1);
        }
        u += seg;
    }
}
