// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill gates (forward.py GateTiles), compute. Per token row tile r of the a/b projection
// ab [T, 2H] (H = 48: a in columns 0..47, b in 48..95; three tiles per row):
// g = A_neg * softplus(a + dt_bias) on the SFPU (fp32), transposed (FPU) into the [heads, T] tiles of g_bh;
// beta = sigmoid(b) (fp32), head h's column moved to column 0 of its own tile by a matmul with the 0/1 selection tile
// SEL_j (exact): the [T, 1] tiles of beta_bh. Every value stays fp32 (the ttnn chain rounded each step to bf16).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/softplus.h"

namespace {
constexpr uint32_t cb_ab = 0, cb_c = 1, cb_sel = 2, cb_t = 3, cb_bt = 4, cb_g = 16, cb_beta = 17;
constexpr uint32_t DTB0 = 0, DTB1 = 1, AN0 = 2, AN1 = 3;  // const tiles: dt_bias and A_neg per column of a-tiles 0, 1
ALWI void fmt(uint32_t a, uint32_t b) { reconfig_data_format(a, b); }
ALWI void put(uint32_t cb, uint32_t d) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    cb_reserve_back(cb, 1);
    pack_tile(d, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}
}  // namespace

#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void add01() {  // DEST 0 += DEST 1
#pragma GCC unroll 8
    for (uint32_t q = 0; q < 32; ++q) {
        const sfpi::vFloat a = sfpi::dst_reg[q];
        const sfpi::vFloat b = sfpi::dst_reg[32 + q];
        sfpi::dst_reg[q] = a + b;
    }
}
inline void mul02() {  // DEST 0 *= DEST 2
#pragma GCC unroll 8
    for (uint32_t q = 0; q < 32; ++q) {
        const sfpi::vFloat a = sfpi::dst_reg[q];
        const sfpi::vFloat b = sfpi::dst_reg[64 + q];
        sfpi::dst_reg[q] = a * b;
    }
}
}  // namespace ckernel::sfpu
#endif

#define SFPU_DO(fn)                                                                       \
    MATH(({                                                                               \
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0); \
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);                                \
        ckernel::sfpu::fn();                                                              \
        math::clear_dst_reg_addr();                                                       \
    }))

void kernel_main() {
    // common args: rows, softplus beta, 1 / beta, threshold (fp32 bits)
    const uint32_t rows = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_b = get_common_arg_val<uint32_t>(1), sp_rb = get_common_arg_val<uint32_t>(2);
    const uint32_t sp_th = get_common_arg_val<uint32_t>(3);
    const uint32_t me = get_arg_val<uint32_t>(0);
    if (me >= rows) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_sel, cb_bt, cb_beta);
    cb_wait_front(cb_c, 4);
    cb_wait_front(cb_sel, 32);
    cb_wait_front(cb_ab, 3);
    // g: a-tiles 0 (heads 0..31) and 1 (heads 32..47 in columns 0..15; the b columns meet zero constants)
    for (uint32_t i = 0; i < 2; ++i) {
        tile_regs_acquire();
        fmt(cb_ab, cb_ab);
        copy_init(cb_ab);
        copy_tile(cb_ab, i, 0);
        fmt(cb_c, cb_c);
        copy_init(cb_c);
        copy_tile(cb_c, i == 0 ? DTB0 : DTB1, 1);
        copy_tile(cb_c, i == 0 ? AN0 : AN1, 2);
        SFPU_DO(add01);
        softplus_tile_init();
        softplus_tile(0, sp_b, sp_rb, sp_th);
        SFPU_DO(mul02);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_t);
        cb_reserve_back(cb_t, 1);
        pack_tile(0, cb_t);
        cb_push_back(cb_t, 1);
        tile_regs_release();
        cb_wait_front(cb_t, 1);
        tile_regs_acquire();
        fmt(cb_t, cb_t);
        transpose_init(cb_t);
        transpose_tile(cb_t, 0, 0);
        put(cb_g, 0);
        cb_pop_front(cb_t, 1);
    }
    // beta: sigmoid of b-tiles (ab tiles 1 and 2), then one column per head
    for (uint32_t i = 1; i < 3; ++i) {
        tile_regs_acquire();
        fmt(cb_ab, cb_ab);
        copy_init(cb_ab);
        copy_tile(cb_ab, i, 0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(0);
        put(cb_bt, 0);
    }
    cb_wait_front(cb_bt, 2);
    for (uint32_t h = 0; h < 48; ++h) {
        const uint32_t src = h < 16 ? 0 : 1, j = h < 16 ? 16 + h : h - 16;
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_bt, cb_sel);
        matmul_init(cb_bt, cb_sel);
        matmul_tiles(cb_bt, cb_sel, src, j, 0);
        put(cb_beta, 0);
    }
    cb_pop_front(cb_bt, 2);
    cb_pop_front(cb_ab, 3);
    cb_pop_front(cb_sel, 32);
    cb_pop_front(cb_c, 4);
}
