// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill per-head RMSNorm + partial RoPE (qk_rope.py), compute. Per item (token row tile r, head h) of a head-major
// [1, H, T, D] tensor: R = sum_j (x_j * x_j) @ ONES over the head's Dt tiles, inv = rsqrt(R / D + eps); the
// pass-through tiles j >= 2 -> x_j * inv * W_j; the two rotary tiles (rotary dim 64: halves of one tile each) n_0, n_1
// likewise, then on the SFPU (fp32) out_0 = n_0 cos_0 - n_1 sin_0, out_1 = n_1 cos_1 + n_0 sin_1 (rotate_half).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "../../ternary/kernels/core_span.hpp"

namespace {
constexpr uint32_t cb_x = 0, cb_c = 1, cb_cs = 2, cb_sq = 3, cb_inv = 4, cb_out = 16;
constexpr uint32_t ONES = 0, W0 = 1;
constexpr auto TO_A = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
ALWI void fmt(uint32_t a, uint32_t b) { reconfig_data_format(a, b); }
ALWI void fmt_mm(uint32_t a, uint32_t b) { reconfig_data_format<SrcOrder::Reverse>(a, b); }
}  // namespace

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// DEST 0 n_0, 1 n_1, 2 cos_0, 3 cos_1, 4 sin_0, 5 sin_1 -> DEST 0, 1 the rotated tiles
inline void rope2() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 1
    for (uint32_t q = 0; q < T; ++q) {
        const sfpi::vFloat a = sfpi::dst_reg[0];
        const sfpi::vFloat b = sfpi::dst_reg[T];
        const sfpi::vFloat c0 = sfpi::dst_reg[2 * T], c1 = sfpi::dst_reg[3 * T];
        const sfpi::vFloat s0 = sfpi::dst_reg[4 * T], s1 = sfpi::dst_reg[5 * T];
        sfpi::dst_reg[0] = a * c0 - b * s0;
        sfpi::dst_reg[T] = b * c1 + a * s1;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

// DEST d = x_j * inv * W_j for j = j0 .. j0 + n - 1 (d = j - j0)
ALWI void normalise(uint32_t j0, uint32_t n) {
    fmt(cb_x, cb_inv);
    mul_init(cb_x, cb_inv);
    for (uint32_t i = 0; i < n; ++i) {
        mul_tiles(cb_x, cb_inv, j0 + i, 0, i);
    }
    fmt(cb_c, cb_c);
    mul_reuse_dest_init<TO_A>(cb_c);
    for (uint32_t i = 0; i < n; ++i) {
        mul_reuse_dest_tiles<TO_A>(cb_c, W0 + j0 + i, i);
    }
}

ALWI void put(uint32_t cb, uint32_t n, bool fp32_tiles) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    cb_reserve_back(cb, n);
    for (uint32_t d = 0; d < n; ++d) {
        if (fp32_tiles) {
            pack_tile<true>(d, cb, d);
        } else {
            pack_tile(d, cb);
        }
    }
    cb_push_back(cb, n);
    tile_regs_release();
}

void kernel_main() {
    // common args: items, ncores, grid_x, Dt, 1/D (fp32 bits), eps (fp32 bits)
    uint32_t first, n_items;
    core_span(
        get_common_arg_val<uint32_t>(0),
        get_common_arg_val<uint32_t>(1),
        get_common_arg_val<uint32_t>(2),
        1,
        first,
        n_items);
    const uint32_t Dt = get_common_arg_val<uint32_t>(3);
    const uint32_t inv_d = get_common_arg_val<uint32_t>(4), eps = get_common_arg_val<uint32_t>(5);
    if (n_items == 0) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_c, cb_x, cb_out);
    cb_wait_front(cb_c, 1 + Dt);
    for (uint32_t it = 0; it < n_items; ++it) {
        cb_wait_front(cb_x, Dt);
        // squares -> R -> inv
        tile_regs_acquire();
        fmt(cb_x, cb_x);
        mul_init(cb_x, cb_x);
        for (uint32_t j = 0; j < Dt; ++j) {
            mul_tiles(cb_x, cb_x, j, j, j);
        }
        put(cb_sq, Dt, true);
        cb_wait_front(cb_sq, Dt);
        tile_regs_acquire();
        fmt_mm(cb_sq, cb_c);
        matmul_init(cb_sq, cb_c);
        for (uint32_t j = 0; j < Dt; ++j) {
            matmul_tiles(cb_sq, cb_c, j, ONES, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_d);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        put(cb_inv, 1, true);
        cb_pop_front(cb_sq, Dt);
        cb_wait_front(cb_inv, 1);
        // pass-through tiles 2 .. Dt - 1
        tile_regs_acquire();
        normalise(2, Dt - 2);
        put(cb_out, Dt - 2, false);
        // rotary tiles 0, 1
        cb_wait_front(cb_cs, 4);
        tile_regs_acquire();
        normalise(0, 2);
        fmt(cb_cs, cb_cs);
        copy_init(cb_cs);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_cs, i, 2 + i);  // cos_0, cos_1, sin_0, sin_1
        }
        MATH(({
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
            TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
            ckernel::sfpu::rope2();
            math::clear_dst_reg_addr();
        }));
        put(cb_out, 2, false);
        cb_pop_front(cb_cs, 4);
        cb_pop_front(cb_inv, 1);
        cb_pop_front(cb_x, Dt);
#ifdef KV_CACHE
        // V's tiles, repacked (bfp8, cb_out's format) for the writer's cache write
        cb_wait_front(5, Dt);
        tile_regs_acquire();
        fmt(5, 5);
        copy_init(5);
        for (uint32_t j = 0; j < Dt; ++j) {
            copy_tile(5, j, j);
        }
        put(cb_out, Dt, false);
        cb_pop_front(5, Dt);
#endif
    }
    cb_pop_front(cb_c, 1 + Dt);
}
