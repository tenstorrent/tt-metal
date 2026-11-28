// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// OpenVDN delta-rule solve, one 128x128 matrix pair at a time per core:
//
//     X = (I + A)^-1      Newton-Schulz in residual form from X0 = c I:  E = I - M X,  X <- X + X E
//     transition = Diag(alpha) X      (HAS_ALPHA)
//     injection  = B X
//
// Same arithmetic as `linear_ops_openvdn.spd_inverse` / `vdn_factor_apply`, fused: every fp32 product is split, each
// fp32 operand `x = hi + lo` with `hi = bf16(x)`, `lo = bf16(x - hi)`, and `a b ~= a_hi b_hi + a_hi b_lo + a_lo b_hi`
// is accumulated in fp32 DST. Sums, differences and the alpha scaling run on the SFPU in fp32; fp32 tiles enter DST
// unpacked to DST (exact). The whole matrix and its split parts stay in L1 between iterations.
//
// Compile-time args: iterations, has_alpha. Runtime args: number of matrices for this core.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t NT = 4;  // tiles per side
constexpr uint32_t TILES = NT * NT;

// fp32 inputs and work buffers (unpacked to DST)
constexpr uint32_t cb_a = 0;
constexpr uint32_t cb_b = 1;
constexpr uint32_t cb_alpha = 2;  // column-broadcast alpha, one tile per row of tiles
constexpr uint32_t cb_const = 3;  // tiles: 0 identity, 1 zeros, 2 c * identity
constexpr uint32_t cb_tmp = 6;  // fp32 staging for a split (16 tiles)
// X and its split parts, two sets (current and next iterate) used alternately: an index into a buffer is only valid
// up to its end, so the next iterate cannot sit behind the current one in the same buffer.
constexpr uint32_t cb_x32[2] = {7, 20};
constexpr uint32_t cb_x_hi[2] = {10, 21};
constexpr uint32_t cb_x_lo[2] = {11, 22};
// bf16 split parts
constexpr uint32_t cb_m_hi = 8;
constexpr uint32_t cb_m_lo = 9;
constexpr uint32_t cb_e_hi = 12;
constexpr uint32_t cb_e_lo = 13;
constexpr uint32_t cb_b_hi = 14;
constexpr uint32_t cb_b_lo = 15;
// outputs, fp32
constexpr uint32_t cb_tr = 16;
constexpr uint32_t cb_inj = 17;

constexpr uint32_t IDENTITY = 0, ZEROS = 1, START = 2;

// Exact copy of one tile into DST slot `dst`.
FORCE_INLINE void copy_to(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

// Pack DST slot `dst` as the next tile of `cb` (between tile_regs_wait and tile_regs_release).
FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}

// DST[dst] += (a b)[i, j] with a = a_hi + a_lo and b = b_hi + b_lo, the a_lo b_lo term dropped. Tile indices are
// relative to each buffer's front; the matrices are TILES tiles, row-major.
FORCE_INLINE void split_matmul_tile(
    uint32_t a_hi, uint32_t a_lo, uint32_t b_hi, uint32_t b_lo, uint32_t i, uint32_t j, uint32_t dst) {
    reconfig_data_format<SrcOrder::Reverse>(a_hi, b_hi);
    matmul_init(a_hi, b_hi);
    for (uint32_t k = 0; k < NT; ++k) {
        matmul_tiles(a_hi, b_hi, i * NT + k, k * NT + j, dst);
        matmul_tiles(a_hi, b_lo, i * NT + k, k * NT + j, dst);
        matmul_tiles(a_lo, b_hi, i * NT + k, k * NT + j, dst);
    }
}

// lo[t] = f32[base + t] - hi[base + t] for the TILES tiles of a matrix, pushed to `lo`. Both inputs must be waited for.
FORCE_INLINE void make_lo(uint32_t f32, uint32_t hi, uint32_t lo, uint32_t base) {
    for (uint32_t t = 0; t < TILES; ++t) {
        tile_regs_acquire();
        copy_to(f32, base + t, 0);
        copy_to(hi, base + t, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, lo);
        tile_regs_release();
    }
}

// Split an fp32 matrix streamed through `in` (plus the identity on the diagonal tiles when `add_identity`) into
// hi/lo. Leaves TILES tiles in each of hi and lo.
FORCE_INLINE void split_input(uint32_t in, bool add_identity, uint32_t hi, uint32_t lo) {
    for (uint32_t t = 0; t < TILES; ++t) {
        cb_wait_front(in, 1);
        tile_regs_acquire();
        copy_to(in, 0, 0);
        if (add_identity) {
            copy_to(cb_const, (t / NT == t % NT) ? IDENTITY : ZEROS, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_tmp);
        pack_to(0, hi);
        tile_regs_release();
        cb_pop_front(in, 1);
    }
    cb_wait_front(cb_tmp, TILES);
    cb_wait_front(hi, TILES);
    make_lo(cb_tmp, hi, lo, 0);
    cb_pop_front(cb_tmp, TILES);
    cb_wait_front(lo, TILES);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t iterations = get_compile_time_arg_val(0);
    constexpr bool has_alpha = get_compile_time_arg_val(1) != 0;
    const uint32_t num_matrices = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_m_hi, cb_x_hi[0], cb_x32[0]);
    cb_wait_front(cb_const, 3);

    for (uint32_t m = 0; m < num_matrices; ++m) {
        // M = A + I, split.
        split_input(cb_a, true, cb_m_hi, cb_m_lo);

        // X0 = c I (c is bf16-exact, so X0's lo part is zero).
        uint32_t cur = 0;
        for (uint32_t t = 0; t < TILES; ++t) {
            tile_regs_acquire();
            copy_to(cb_const, (t / NT == t % NT) ? START : ZEROS, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_x32[cur]);
            pack_to(0, cb_x_hi[cur]);
            tile_regs_release();
        }
        cb_wait_front(cb_x32[cur], TILES);
        cb_wait_front(cb_x_hi[cur], TILES);
        make_lo(cb_x32[cur], cb_x_hi[cur], cb_x_lo[cur], 0);
        cb_wait_front(cb_x_lo[cur], TILES);

        for (uint32_t it = 0; it < iterations; ++it) {
            const uint32_t x32 = cb_x32[cur], x_hi = cb_x_hi[cur], x_lo = cb_x_lo[cur];
            const uint32_t nx32 = cb_x32[1 - cur], nx_hi = cb_x_hi[1 - cur], nx_lo = cb_x_lo[1 - cur];
            // E = I - M X, split (fp32 staging in cb_tmp).
            for (uint32_t i = 0; i < NT; ++i) {
                for (uint32_t j = 0; j < NT; ++j) {
                    tile_regs_acquire();
                    split_matmul_tile(cb_m_hi, cb_m_lo, x_hi, x_lo, i, j, 0);
                    copy_to(cb_const, i == j ? IDENTITY : ZEROS, 1);
                    sub_binary_tile_init();
                    sub_binary_tile(1, 0, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_to(0, cb_tmp);
                    pack_to(0, cb_e_hi);
                    tile_regs_release();
                }
            }
            cb_wait_front(cb_tmp, TILES);
            cb_wait_front(cb_e_hi, TILES);
            make_lo(cb_tmp, cb_e_hi, cb_e_lo, 0);
            cb_pop_front(cb_tmp, TILES);
            cb_wait_front(cb_e_lo, TILES);

            // X_next = X + X E into the other set, then its lo part; the current X is dropped.
            for (uint32_t i = 0; i < NT; ++i) {
                for (uint32_t j = 0; j < NT; ++j) {
                    tile_regs_acquire();
                    split_matmul_tile(x_hi, x_lo, cb_e_hi, cb_e_lo, i, j, 0);
                    copy_to(x32, i * NT + j, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_to(0, nx32);
                    pack_to(0, nx_hi);
                    tile_regs_release();
                }
            }
            cb_pop_front(cb_e_hi, TILES);
            cb_pop_front(cb_e_lo, TILES);
            cb_pop_front(x32, TILES);
            cb_pop_front(x_hi, TILES);
            cb_pop_front(x_lo, TILES);
            cb_wait_front(nx32, TILES);
            cb_wait_front(nx_hi, TILES);
            make_lo(nx32, nx_hi, nx_lo, 0);
            cb_wait_front(nx_lo, TILES);
            cur = 1 - cur;
        }
        cb_pop_front(cb_m_hi, TILES);
        cb_pop_front(cb_m_lo, TILES);
        const uint32_t x32 = cb_x32[cur], x_hi = cb_x_hi[cur], x_lo = cb_x_lo[cur];

        // transition = Diag(alpha) (X + X^T) / 2 (symmetrised as the composite does; the transposed tile is transposed
        // in DST, exact): row-tile i scaled by alpha's column-broadcast tile i (fp32, exact product).
        if constexpr (has_alpha) {
            cb_wait_front(cb_alpha, NT);
            for (uint32_t i = 0; i < NT; ++i) {
                for (uint32_t j = 0; j < NT; ++j) {
                    tile_regs_acquire();
                    copy_to(x32, i * NT + j, 0);
                    copy_to(x32, j * NT + i, 1);
                    transpose_dest_init<true>();
                    transpose_dest<true>(1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                    fill_tile_init();
                    fill_tile(1, 0.5f);
                    mul_binary_tile_init();
                    mul_binary_tile(0, 1, 0);
                    copy_to(cb_alpha, i, 1);
                    mul_binary_tile_init();
                    mul_binary_tile(0, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_to(0, cb_tr);
                    tile_regs_release();
                }
            }
            cb_pop_front(cb_alpha, NT);
        }

        // injection = B X.
        split_input(cb_b, false, cb_b_hi, cb_b_lo);
        for (uint32_t i = 0; i < NT; ++i) {
            for (uint32_t j = 0; j < NT; ++j) {
                tile_regs_acquire();
                split_matmul_tile(cb_b_hi, cb_b_lo, x_hi, x_lo, i, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_to(0, cb_inj);
                tile_regs_release();
            }
        }
        cb_pop_front(cb_b_hi, TILES);
        cb_pop_front(cb_b_lo, TILES);
        cb_pop_front(x32, TILES);
        cb_pop_front(x_hi, TILES);
        cb_pop_front(x_lo, TILES);
    }
}
