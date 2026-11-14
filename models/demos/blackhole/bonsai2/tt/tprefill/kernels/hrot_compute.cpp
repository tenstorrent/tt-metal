// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Prefill rotation, compute. Output tile c of the block = sum_a H32[c, a] * (X_a @ diag(d_a) H32 / 32), H32[c, a] =
// (-1)^popcount(c & a). With c = 8 c1 + c0 and a = 8 a1 + a0 the sign splits: H32[c, a] = H4[c1, a1] H8[c0, a0]. Per
// c1: DEST tile a0 (8 tiles) = sum_a1 H4[c1, a1] X_a @ M_a (4 matmuls, the weight tile of the sign), then an 8-point
// Walsh-Hadamard transform over the 8 DEST tiles on the SFPU (fp32) leaves output tile 8 c1 + c0 in DEST tile c0.
// 128 matmuls per row tile instead of 1024. The weight entries are +-1/32 (one bit) and HiFi3 covers the whole bf16
// activation, so every product is exact; fp32 accumulation and fp32 sums. HROT_DENSE: the 1024-matmul version.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/compute_kernel_api.h"  // sigmoid_tile
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// In place over DEST tiles 0..7: tile c0 = sum_a0 (-1)^popcount(c0 & a0) tile a0, one SFPU row of every tile per step.
inline void fwht8() {
    constexpr uint32_t T = 32;
#pragma GCC unroll 1
    for (uint32_t q = 0; q < T; ++q) {
#pragma GCC unroll 2
        for (uint32_t h = 0; h < 8; h += 4) {  // 4-point transform of tiles h .. h + 3
            const sfpi::vFloat a = sfpi::dst_reg[(h + 0) * T];
            const sfpi::vFloat b = sfpi::dst_reg[(h + 1) * T];
            const sfpi::vFloat c = sfpi::dst_reg[(h + 2) * T];
            const sfpi::vFloat d = sfpi::dst_reg[(h + 3) * T];
            const sfpi::vFloat s0 = a + b, s1 = a - b, s2 = c + d, s3 = c - d;
            sfpi::dst_reg[(h + 0) * T] = s0 + s2;
            sfpi::dst_reg[(h + 1) * T] = s1 + s3;
            sfpi::dst_reg[(h + 2) * T] = s0 - s2;
            sfpi::dst_reg[(h + 3) * T] = s1 - s3;
        }
#pragma GCC unroll 4
        for (uint32_t i = 0; i < 4; ++i) {  // the high bit: tiles i and i + 4
            const sfpi::vFloat a = sfpi::dst_reg[i * T];
            const sfpi::vFloat b = sfpi::dst_reg[(i + 4) * T];
            sfpi::dst_reg[i * T] = a + b;
            sfpi::dst_reg[(i + 4) * T] = a - b;
        }
        sfpi::dst_reg++;
    }
}
inline void mul4() {  // DEST i *= DEST 4 + i, i < 4
#pragma GCC unroll 2
    for (uint32_t q = 0; q < 32; ++q) {
#pragma GCC unroll 4
        for (uint32_t i = 0; i < 4; ++i) {
            const sfpi::vFloat a = sfpi::dst_reg[32 * i];
            const sfpi::vFloat b = sfpi::dst_reg[32 * (4 + i)];
            sfpi::dst_reg[32 * i] = a * b;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    const uint32_t m_count = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_x = tt::CBIndex::c_0;
#ifdef GATE
    constexpr uint32_t cb_g = tt::CBIndex::c_3, cb_rot = tt::CBIndex::c_2;  // gated input, packed for the rotation
#else
    constexpr uint32_t cb_rot = cb_x;
#endif
    constexpr uint32_t cb_h = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    if (m_count == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_h, cb_rot, cb_out);  // SrcA = in1 (the weight tiles), SrcB = in0 (x)
    matmul_init(cb_rot, cb_h);
    cb_wait_front(cb_h, 64);
    for (uint32_t m = 0; m < m_count; ++m) {
#ifdef GATE
        // x_a = attn_a * sigmoid(gate_a) (fp32 on the SFPU, one bf16 rounding) for the block's 32 input tiles
        cb_wait_front(cb_x, 32);
        cb_wait_front(cb_g, 32);
        copy_init(cb_x);  // c_0 and c_3 share the format
        sigmoid_tile_init<false>();
        for (uint32_t a0 = 0; a0 < 32; a0 += 4) {  // 4 tiles per DEST pass: attn in DEST 0..3, gates in 4..7
            tile_regs_acquire();
            for (uint32_t i = 0; i < 4; ++i) {
                copy_tile(cb_x, a0 + i, i);
                copy_tile(cb_g, a0 + i, 4 + i);
            }
            for (uint32_t i = 4; i < 8; ++i) {
                sigmoid_tile<VectorMode::RC, false>(i);
            }
            MATH(({
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
                TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                ckernel::sfpu::mul4();
                math::clear_dst_reg_addr();
            }));
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_rot, 4);
            for (uint32_t i = 0; i < 4; ++i) {
                pack_tile(i, cb_rot);
            }
            cb_push_back(cb_rot, 4);
            tile_regs_release();
        }
        cb_pop_front(cb_x, 32);
        cb_pop_front(cb_g, 32);
#endif
        cb_wait_front(cb_rot, 32);
#ifdef HROT_DENSE
        for (uint32_t c = 0; c < 32; ++c) {
            tile_regs_acquire();
            for (uint32_t a = 0; a < 32; ++a) {
                matmul_tiles(cb_rot, cb_h, a, 2 * a + (__builtin_popcount(c & a) & 1), 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
#else
        for (uint32_t c1 = 0; c1 < 4; ++c1) {
            tile_regs_acquire();
            matmul_init(cb_rot, cb_h);
            for (uint32_t a0 = 0; a0 < 8; ++a0) {
                for (uint32_t a1 = 0; a1 < 4; ++a1) {
                    const uint32_t a = 8 * a1 + a0;
                    matmul_tiles(cb_rot, cb_h, a, 2 * a + (__builtin_popcount(c1 & a1) & 1), a0);
                }
            }
            addcmul_tile_init();
            MATH(({
                math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
                TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                ckernel::sfpu::fwht8();
                math::clear_dst_reg_addr();
            }));
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c0 = 0; c0 < 8; ++c0) {
                cb_reserve_back(cb_out, 1);
                pack_tile(c0, cb_out);
                cb_push_back(cb_out, 1);
            }
            tile_regs_release();
        }
#endif
        cb_pop_front(cb_rot, 32);
    }
}
