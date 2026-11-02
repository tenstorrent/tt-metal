// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC residual update, exact fp32. Per token-tile r the reader supplies the n*n + n coefficient
// column tiles (coefficient in column 0); phase A broadcasts each across the 32 columns with the
// unpacker's 32-bit column-broadcast datacopy (unpack-to-dest: exact fp32) into a resident CB.
// Per position (r, c) and output stream j:
//     acc = sum_i bcast(comb[i, j]) * x[r, i, c] + bcast(post[j]) * y[r, c]
// with fp32 SFPU multiplies and adds in the fp32 dest register (exact). With ``collapse`` the n
// stream results are summed in dest and scaled by 1/n (the LM head's mean over the streams).
// Dest slots (fp32 mode, 4 per acquire): 0 = per-stream acc, 1 = coefficient / product, 2 = operand,
// 3 = collapse total.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t CB_X = get_compile_time_arg_val(0);
constexpr uint32_t CB_Y = get_compile_time_arg_val(1);
constexpr uint32_t CB_COL = get_compile_time_arg_val(2);
constexpr uint32_t CB_BCAST = get_compile_time_arg_val(3);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(4);
constexpr uint32_t N = get_compile_time_arg_val(5);
constexpr uint32_t CT = get_compile_time_arg_val(6);
constexpr bool COLLAPSE = get_compile_time_arg_val(7) != 0;
constexpr uint32_t INV_N_BITS = get_compile_time_arg_val(8);
constexpr uint32_t N_COEF = N * N + N;

constexpr uint32_t D_ACC = 0, D_COEF = 1, D_OPND = 2, D_TOTAL = 3;

// dest[D_ACC] <- sum over the n stream products + the post * y product for output stream j.
FORCE_INLINE void stream_j(uint32_t j, uint32_t acc) {
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_BCAST, i * N + j, D_COEF);  // bcast(comb[i, j])
        copy_tile(CB_X, i, D_OPND);              // x[r, i, c]
        mul_binary_tile_init();
        if (i == 0) {
            mul_binary_tile(D_COEF, D_OPND, acc);
        } else {
            mul_binary_tile(D_COEF, D_OPND, D_COEF);
            add_binary_tile_init();
            add_binary_tile(acc, D_COEF, acc);
        }
    }
    copy_tile(CB_BCAST, N * N + j, D_COEF);  // bcast(post[j])
    copy_tile(CB_Y, 0, D_OPND);              // y[r, c]
    mul_binary_tile_init();
    mul_binary_tile(D_COEF, D_OPND, D_COEF);
    add_binary_tile_init();
    add_binary_tile(acc, D_COEF, acc);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_OUT);
    const uint32_t start_pos = get_arg_val<uint32_t>(0);
    const uint32_t num_pos = get_arg_val<uint32_t>(1);

    CircularBuffer cbx(CB_X), cby(CB_Y), cbcol(CB_COL), cbb(CB_BCAST), cbo(CB_OUT);

    uint32_t cur_r = 0xFFFFFFFFu;
    for (uint32_t p = start_pos; p < start_pos + num_pos; ++p) {
        const uint32_t r = p / CT;
        if (r != cur_r) {
            if (cur_r != 0xFFFFFFFFu) {
                cbb.pop_front(N_COEF);  // the previous token-tile's broadcast set
            }
            // ---- phase A: column tiles -> broadcast tiles (32-bit column-broadcast datacopy, exact)
            cbcol.wait_front(N_COEF);
            unary_bcast_init<BroadcastType::COL>(CB_COL);
            for (uint32_t k = 0; k < N_COEF; ++k) {
                tile_regs_acquire();
                unary_bcast<BroadcastType::COL>(CB_COL, k, D_ACC);
                tile_regs_commit();
                tile_regs_wait();
                cbb.reserve_back(1);
                pack_tile(D_ACC, CB_BCAST);
                cbb.push_back(1);
                tile_regs_release();
            }
            unary_bcast_uninit<BroadcastType::COL>(CB_COL);
            cbcol.pop_front(N_COEF);  // lets the reader build the next token-tile's set
            cbb.wait_front(N_COEF);   // resident for every position of this token-tile
            copy_init(CB_X);          // plain fp32 copies (unpack to dest) for phase B
            cur_r = r;
        }

        // ---- phase B: the n output tiles (or their mean) of position (r, c)
        cbx.wait_front(N);
        cby.wait_front(1);
        if constexpr (COLLAPSE) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < N; ++j) {
                if (j == 0) {
                    stream_j(j, D_TOTAL);
                } else {
                    stream_j(j, D_ACC);
                    add_binary_tile_init();
                    add_binary_tile(D_TOTAL, D_ACC, D_TOTAL);
                }
            }
            binop_with_scalar_tile_init();
            mul_unary_tile(D_TOTAL, INV_N_BITS);
            tile_regs_commit();
            tile_regs_wait();
            cbo.reserve_back(1);
            pack_tile(D_TOTAL, CB_OUT);
            cbo.push_back(1);
            tile_regs_release();
        } else {
            for (uint32_t j = 0; j < N; ++j) {
                tile_regs_acquire();
                stream_j(j, D_ACC);
                tile_regs_commit();
                tile_regs_wait();
                cbo.reserve_back(1);
                pack_tile(D_ACC, CB_OUT);
                cbo.push_back(1);
                tile_regs_release();
            }
        }
        cbx.pop_front(N);
        cby.pop_front(1);
    }
    if (cur_r != 0xFFFFFFFFu) {
        cbb.pop_front(N_COEF);
    }
}
