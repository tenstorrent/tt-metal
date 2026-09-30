// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The verify rows' GDN recurrence as gdn_step's compute body in a row loop: for one (value head, state column
// block) item, conv + SiLU and both l2 norms once over the 32-row tile, then rows r = 0 .. ROWS-1 through the SAME
// per-row calls as gdn_step (S' = S * decay; v_read = k S'; delta_b = ((v - v_read) * M_r) * beta;
// S = S' + k^T delta_b; o_r = (q S) * 128^-0.5) with the gate scalars of every row computed in one pass before them
// (gates_all; the reader replicates them into per-row tiles), the state carried row to row in L1 and every prefix
// state handed to the writer, and the gated RMSNorm once on the assembled o rows.  Rounding points, LLK calls, DST
// accumulation orders and CB discipline are gdn_step's call for call: the row loop is bitwise ROWS sequential gdn_step
// calls. Compile-time args: 0 ROWS (real rows of the tile), 1 VBT (state column tiles per item: 4 = whole head, 1 = one
// of four blocks).  Runtime args: 0 items on this core.

#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#include "../../kernels/source_softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "../../kernels/source_reciprocal.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "../../kernels/zones.h"

using namespace ckernel;

namespace {
constexpr uint32_t CB_P = 0, CB_S = 1, CB_T = 2, CB_Z = 3, CB_AB = 4, CB_DTNA = 5, CB_W = 6, CB_STATE = 7;
constexpr uint32_t CB_MASK = 8, CB_SCALER = 9, CB_CONVSUM = 10, CB_QKV = 11, CB_SQ = 12, CB_GALL = 13, CB_UNIT = 14;
constexpr uint32_t CB_QROW = 15, CB_KROW = 16, CB_KCOL = 17, CB_BETA = 18, CB_DECAY = 19, CB_SDEC = 20, CB_SDECC = 21;
constexpr uint32_t CB_VREAD = 22, CB_DELTAB = 23, CB_SNEW = 24, CB_OUTS = 25, CB_OUTG = 26, CB_RS = 27;
constexpr uint32_t CB_DBG = 28, CB_OROWS = 29, CB_SNEWC = 30, CB_OBF = 31;
// phase-E reuse of phase-D buffers (same format and depth).  CB_OBF (each row's o, to the writer) is its own buffer:
// gdn_step aliases it with CB_CONVSUM, which only the compute consumes; here the writer pops it.
constexpr uint32_t CB_NRMW = CB_SQ, CB_NRM = CB_UNIT, CB_SIG = CB_VREAD, CB_SQO = CB_DELTAB;

constexpr uint32_t ROWS = get_compile_time_arg_val(0);
constexpr uint32_t VBT = get_compile_time_arg_val(1);
constexpr uint32_t HT = 4;                 // 128 / 32
constexpr uint32_t QK_TILES = 2 * HT;      // the q and k tiles of the item's key head
constexpr uint32_t NT = QK_TILES + VBT;    // conv tiles per item: q, k, the item's v tiles
constexpr uint32_t ST = HT * VBT;          // state tiles per item
constexpr uint32_t EPS_1E6 = 0x358637BD;   // 1e-6f: l2 norm eps and RMS_NORM_EPS
constexpr uint32_t QK_SCALE = 0x3DB504F3;  // 128^-0.5
constexpr uint32_t F_ONE = 0x3F800000, F_TWENTY = 0x41A00000;
static_assert(ROWS >= 1 && ROWS <= 32, "rows of one tile");
static_assert(VBT == 4 || VBT == 1, "a whole head or one of four column blocks");

#ifdef DEBUG_TAPS
// pack DST tile idst a second time into the debug CB; restore_cb is the CB the surrounding code packs into
ALWI void tap(uint32_t idst, uint32_t restore_cb) {
    cb_reserve_back(CB_DBG, 1);
    pack_reconfig_data_format(CB_DBG);
    pack_tile(idst, CB_DBG);
    cb_push_back(CB_DBG, 1);
    pack_reconfig_data_format(restore_cb);
}
#endif

// conv[t] = silu(bf16(sum_i slot_i[t] * tap_i[t])) for the item's NT tiles; taps broadcast row 0.  The reader
// pushes the inputs per tile (CB_S tile-major: tile t's slots at 3t + i; CB_T: its taps at 4t + i).
ALWI void conv_silu() {
    for (uint32_t t = 0; t < NT; ++t) {
        cb_wait_front(CB_P, t + 1);
        cb_wait_front(CB_S, 3 * (t + 1));
        cb_wait_front(CB_T, 4 * (t + 1));
        tile_regs_acquire();
        reconfig_data_format(CB_S, CB_T);
        mul_bcast_rows_init(CB_S, CB_T);
        for (uint32_t tap = 0; tap < 3; ++tap) {
            mul_tiles_bcast_rows(CB_S, CB_T, 3 * t + tap, 4 * t + tap, 0);
        }
        reconfig_data_format(CB_P, CB_T);
        mul_bcast_rows_init(CB_P, CB_T);
        mul_tiles_bcast_rows(CB_P, CB_T, t, 4 * t + 3, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_CONVSUM, 1);
        pack_reconfig_data_format(CB_CONVSUM);
        pack_tile(0, CB_CONVSUM);
        cb_push_back(CB_CONVSUM, 1);
        tile_regs_release();

        cb_wait_front(CB_CONVSUM, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_CONVSUM);
        copy_init(CB_CONVSUM);
        copy_tile(CB_CONVSUM, 0, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_QKV, 1);
        pack_reconfig_data_format(CB_QKV);
        pack_tile(0, CB_QKV);
        cb_push_back(CB_QKV, 1);
#ifdef DEBUG_TAPS
        if (t == 0 || t == QK_TILES) {
            tap(0, CB_QKV);
        }
#endif
        tile_regs_release();
        cb_pop_front(CB_CONVSUM, 1);
    }
}

// unit = x * rsqrt(sum(x*x) + eps) in the oracle's bf16 chain; the fp32 copy of unit -> out_cb.
ALWI void l2_norm(uint32_t base, uint32_t out_cb) {
    reconfig_data_format(CB_QKV, CB_QKV);
    mul_init(CB_QKV, CB_QKV);
    pack_reconfig_data_format(CB_SQ);
    cb_reserve_back(CB_SQ, HT);
    for (uint32_t t = 0; t < HT; ++t) {
        tile_regs_acquire();
        mul_tiles(CB_QKV, CB_QKV, base + t, base + t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SQ);
        tile_regs_release();
    }
    cb_push_back(CB_SQ, HT);

    cb_wait_front(CB_SQ, HT);
    reconfig_data_format(CB_SCALER, CB_SQ);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, CB_CONVSUM);
    pack_reconfig_data_format(CB_CONVSUM);
    tile_regs_acquire();
    for (uint32_t t = 0; t < HT; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, t, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_CONVSUM, 1);
    pack_tile(0, CB_CONVSUM);
    cb_push_back(CB_CONVSUM, 1);
    tile_regs_release();
    reduce_uninit();
    cb_pop_front(CB_SQ, HT);

    // + eps -> bf16, then rsqrt -> bf16
    for (uint32_t step = 0; step < 2; ++step) {
        cb_wait_front(CB_CONVSUM, 1);
        reconfig_data_format_srca(CB_CONVSUM);
        copy_init(CB_CONVSUM);
        if (step == 0) {
            binop_with_scalar_tile_init();
        } else {
            sqrt_tile_init();
            qwen38_recip_tile_init();
        }
        tile_regs_acquire();
        copy_tile(CB_CONVSUM, 0, 0);
        if (step == 0) {
            add_unary_tile(0, EPS_1E6);
        } else {
            sqrt_tile(0);
            qwen38_recip_tile(0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_CONVSUM, 1);
        pack_tile(0, CB_CONVSUM);
        cb_push_back(CB_CONVSUM, 1);
        tile_regs_release();
        cb_pop_front(CB_CONVSUM, 1);
    }

    cb_wait_front(CB_CONVSUM, 1);
    reconfig_data_format(CB_QKV, CB_CONVSUM);
    mul_bcast_cols_init(CB_QKV, CB_CONVSUM);
    pack_reconfig_data_format(CB_UNIT);
    cb_reserve_back(CB_UNIT, HT);
    for (uint32_t t = 0; t < HT; ++t) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(CB_QKV, CB_CONVSUM, base + t, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_UNIT);
        tile_regs_release();
    }
    cb_push_back(CB_UNIT, HT);
    cb_pop_front(CB_CONVSUM, 1);

    cb_wait_front(CB_UNIT, HT);
    reconfig_data_format_srca(CB_UNIT);
    copy_init(CB_UNIT);
    pack_reconfig_data_format(out_cb);
    cb_reserve_back(out_cb, HT);
    for (uint32_t t = 0; t < HT; ++t) {
        tile_regs_acquire();
        copy_tile(CB_UNIT, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
#ifdef DEBUG_TAPS
        tap(0, out_cb);
#endif
        tile_regs_release();
    }
    cb_push_back(out_cb, HT);
    cb_pop_front(CB_UNIT, HT);
}

// k_col = transpose(k) once per item (gdn_step transposes inside every step: the same data movement).
ALWI void k_columns() {
    cb_wait_front(CB_KROW, HT);
    reconfig_data_format_srca(CB_KROW);
    transpose_init(CB_KROW);
    pack_reconfig_data_format(CB_KCOL);
    cb_reserve_back(CB_KCOL, HT);
    for (uint32_t i = 0; i < HT; ++i) {
        tile_regs_acquire();
        transpose_tile(CB_KROW, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_KCOL);
#ifdef DEBUG_TAPS
        if (i == 0) {
            tap(0, CB_KCOL);
        }
#endif
        tile_regs_release();
    }
    cb_push_back(CB_KCOL, HT);
}

// beta_all = bf16(sigmoid(b)) and decay_all = exp(neg_exp_A * softplus(a + dt_bias)) for EVERY element of the
// projection's a / b tiles (CB_AB: tile 0 = a, tile 1 = b; element (row, head) is row `row`'s scalar of head `head`)
// in one SFPU pass each -- the same element-wise LLK calls gdn_step runs on its one scalar (the scalar-broadcast
// unpack there and the plain bf16 copy here both land the exact bf16 value in DST; sigmoid / softplus / exp / the
// binary ops are lane-wise), so element (row, head) holds bit for bit the value gdn_step's chain gives that row.  The
// two results go to the reader as fp32 tiles (CB_GALL), which replicates each row's element into the full fp32 tiles
// the recurrence multiplies (CB_BETA / CB_DECAY, exact 32-bit copies); beta keeps its bf16 pack round trip before
// that.  dt_bias / neg_exp_A stay full tiles (an fp32 broadcast through the source registers would truncate them).
ALWI void gates_all() {
    cb_wait_front(CB_AB, 2);
    cb_wait_front(CB_DTNA, 2);
    cb_reserve_back(CB_GALL, 2);
    // beta: sigmoid(b) -> bf16 (CB_CONVSUM, free after the conv) -> fp32 tile 0 of CB_GALL
    tile_regs_acquire();
    reconfig_data_format_srca(CB_AB);
    copy_init(CB_AB);
    copy_tile(CB_AB, 1, 0);
    sigmoid_tile_init<false>();
    sigmoid_tile<VectorMode::RC, false>(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_CONVSUM, 1);
    pack_reconfig_data_format(CB_CONVSUM);
    pack_tile(0, CB_CONVSUM);
    cb_push_back(CB_CONVSUM, 1);
    tile_regs_release();
    cb_wait_front(CB_CONVSUM, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(CB_CONVSUM);
    copy_init(CB_CONVSUM);
    copy_tile(CB_CONVSUM, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_GALL);
    pack_tile(0, CB_GALL);
    tile_regs_release();
    cb_pop_front(CB_CONVSUM, 1);
    // decay: exp(neg_exp_A * softplus(a + dt_bias)) -> fp32 tile 1 of CB_GALL
    tile_regs_acquire();
    reconfig_data_format_srca(CB_AB);
    copy_init(CB_AB);
    copy_tile(CB_AB, 0, 0);
    reconfig_data_format_srca(CB_DTNA);
    copy_init(CB_DTNA);
    copy_tile(CB_DTNA, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    qwen38_softplus_tile(0, F_ONE, F_ONE, F_TWENTY);
    copy_tile(CB_DTNA, 1, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_GALL);
    cb_push_back(CB_GALL, 2);
    tile_regs_release();
}

#ifdef DEBUG_TAPS
// the per-row gate taps (beta, decay) in the gates block's place: the reader's replicated tiles, once they exist
ALWI void tap_gate_tiles() {
    cb_wait_front(CB_BETA, ROWS);
    cb_wait_front(CB_DECAY, ROWS);
    for (uint32_t row = 0; row < ROWS; ++row) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_BETA);
        copy_init(CB_BETA);
        copy_tile(CB_BETA, row, 0);
        tile_regs_commit();
        tile_regs_wait();
        tap(0, CB_KCOL);
        tile_regs_release();
        tile_regs_acquire();
        reconfig_data_format_srca(CB_DECAY);
        copy_init(CB_DECAY);
        copy_tile(CB_DECAY, row, 0);
        tile_regs_commit();
        tile_regs_wait();
        tap(0, CB_KCOL);
        tile_regs_release();
    }
}
#endif

// One row: S' = S * decay; v_read = k S'; delta_b = ((v - v_read) * M_row) * beta; S_new = S' + k^T delta_b;
// o = (q S_new) * 128^-0.5.  The state comes from CB_STATE on the first row and from the previous row's CB_SNEWC
// afterwards; the consumed state is handed to the writer through the CB_OUTS alias (prefix state row).
ALWI void recurrence(uint32_t row, bool first) {
    const uint32_t src = first ? CB_STATE : CB_SNEWC;
    cb_wait_front(src, ST);

    reconfig_data_format_srca(src);
    copy_init(src);
    mul_binary_tile_init();
    pack_reconfig_data_format(CB_SDEC);
    cb_reserve_back(CB_SDEC, ST);
    cb_reserve_back(CB_SDECC, ST);
    for (uint32_t t = 0; t < ST; t += 2) {
        tile_regs_acquire();
        copy_tile(src, t, 0);
        copy_tile(CB_DECAY, row, 1);
        mul_binary_tile(0, 1, 0);
        copy_tile(src, t + 1, 2);
        copy_tile(CB_DECAY, row, 3);
        mul_binary_tile(2, 3, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SDEC);
        pack_tile(2, CB_SDEC);
        pack_tile(0, CB_SDECC);
        pack_tile(2, CB_SDECC);
#ifdef DEBUG_TAPS
        if (t == 0) {
            tap(0, CB_SDEC);
        }
#endif
        tile_regs_release();
    }
    cb_push_back(CB_SDEC, ST);
    cb_push_back(CB_SDECC, ST);
    cb_pop_front(src, ST);
    if (!first) {
        cb_push_back(CB_OUTS, ST);  // the state after `row` rows, to the writer (same bytes as CB_SNEWC's front)
    }
    cb_wait_front(CB_SDEC, ST);
    cb_wait_front(CB_SDECC, ST);

    reconfig_data_format<SrcOrder::Reverse>(CB_KROW, CB_SDEC);
    matmul_init(CB_KROW, CB_SDEC);
    pack_reconfig_data_format(CB_VREAD);
    tile_regs_acquire();
    for (uint32_t j = 0; j < VBT; ++j) {
        for (uint32_t i = 0; i < HT; ++i) {
            matmul_tiles(CB_KROW, CB_SDEC, i, i * VBT + j, j);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_VREAD, VBT);
    for (uint32_t j = 0; j < VBT; ++j) {
        pack_tile(j, CB_VREAD);
    }
    cb_push_back(CB_VREAD, VBT);
#ifdef DEBUG_TAPS
    tap(0, CB_VREAD);
#endif
    tile_regs_release();

    cb_wait_front(CB_VREAD, VBT);
    pack_reconfig_data_format(CB_DELTAB);
    cb_reserve_back(CB_DELTAB, VBT);
    for (uint32_t j = 0; j < VBT; ++j) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_QKV);
        copy_init(CB_QKV);
        copy_tile(CB_QKV, j, 0);
        reconfig_data_format_srca(CB_VREAD);
        copy_init(CB_VREAD);
        copy_tile(CB_VREAD, j, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        reconfig_data_format_srca(CB_MASK);
        copy_init(CB_MASK);
        copy_tile(CB_MASK, row, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(CB_BETA);
        copy_init(CB_BETA);
        copy_tile(CB_BETA, row, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_DELTAB);
#ifdef DEBUG_TAPS
        if (j == 0) {
            tap(0, CB_DELTAB);
        }
#endif
        tile_regs_release();
    }
    cb_push_back(CB_DELTAB, VBT);
    cb_pop_front(CB_VREAD, VBT);

    cb_wait_front(CB_DELTAB, VBT);
    cb_reserve_back(CB_OUTS, ST);  // the writer has drained the prefix state two rows back (depth 2 x ST)
    cb_reserve_back(CB_SNEW, ST);
    cb_reserve_back(CB_SNEWC, ST);
    pack_reconfig_data_format(CB_SNEW);
    for (uint32_t i = 0; i < HT; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_SDECC);
        copy_init(CB_SDECC);
        for (uint32_t j = 0; j < VBT; ++j) {
            copy_tile(CB_SDECC, i * VBT + j, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(CB_KCOL, CB_DELTAB);
        matmul_init(CB_KCOL, CB_DELTAB);
        for (uint32_t j = 0; j < VBT; ++j) {
            matmul_tiles(CB_KCOL, CB_DELTAB, i, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < VBT; ++j) {
            pack_tile(j, CB_SNEW);
        }
        for (uint32_t j = 0; j < VBT; ++j) {
            pack_tile(j, CB_SNEWC);
        }
#ifdef DEBUG_TAPS
        if (i == 0) {
            tap(0, CB_SNEW);
        }
#endif
        tile_regs_release();
    }
    cb_push_back(CB_SNEW, ST);
    cb_push_back(CB_SNEWC, ST);
    cb_pop_front(CB_SDEC, ST);
    cb_pop_front(CB_SDECC, ST);
    cb_pop_front(CB_DELTAB, VBT);

    cb_wait_front(CB_SNEW, ST);
    reconfig_data_format<SrcOrder::Reverse>(CB_QROW, CB_SNEW);
    matmul_init(CB_QROW, CB_SNEW);
    pack_reconfig_data_format(CB_OBF);
    tile_regs_acquire();
    for (uint32_t j = 0; j < VBT; ++j) {
        for (uint32_t i = 0; i < HT; ++i) {
            matmul_tiles(CB_QROW, CB_SNEW, i, i * VBT + j, j);
        }
    }
    binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < VBT; ++j) {
        mul_unary_tile(j, QK_SCALE);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_OBF, VBT);
    for (uint32_t j = 0; j < VBT; ++j) {
        pack_tile(j, CB_OBF);
    }
    cb_push_back(CB_OBF, VBT);
#ifdef DEBUG_TAPS
    tap(0, CB_OBF);
#endif
    tile_regs_release();
    cb_pop_front(CB_SNEW, ST);
}

// gated = bf16(bf16(w * bf16(o * rsqrt(mean(o^2) + eps))) * sigmoid(z)) on the head's four assembled o-row tiles.
ALWI void gated_norm() {
    cb_wait_front(CB_OROWS, HT);
    cb_wait_front(CB_W, HT);
    cb_wait_front(CB_Z, HT);

    reconfig_data_format(CB_OROWS, CB_OROWS);
    mul_init(CB_OROWS, CB_OROWS);
    pack_reconfig_data_format(CB_SQO);
    cb_reserve_back(CB_SQO, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        tile_regs_acquire();
        mul_tiles(CB_OROWS, CB_OROWS, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SQO);
        tile_regs_release();
    }
    cb_push_back(CB_SQO, HT);

    cb_wait_front(CB_SQO, HT);
    reconfig_data_format(CB_SCALER, CB_SQO);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQO, CB_SCALER, CB_RS);
    pack_reconfig_data_format(CB_RS);
    binop_with_scalar_tile_init();
    sqrt_tile_init();
    qwen38_recip_tile_init();
    tile_regs_acquire();
    for (uint32_t j = 0; j < HT; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQO, CB_SCALER, j, 1, 0);
    }
    add_unary_tile(0, EPS_1E6);
    sqrt_tile(0);
    qwen38_recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_RS, 1);
    pack_tile(0, CB_RS);
    cb_push_back(CB_RS, 1);
    tile_regs_release();
    reduce_uninit();
    cb_pop_front(CB_SQO, HT);

    cb_wait_front(CB_RS, 1);
    reconfig_data_format(CB_OROWS, CB_RS);
    mul_bcast_cols_init(CB_OROWS, CB_RS);
    pack_reconfig_data_format(CB_NRM);
    cb_reserve_back(CB_NRM, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(CB_OROWS, CB_RS, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_NRM);
        tile_regs_release();
    }
    cb_push_back(CB_NRM, HT);
    cb_pop_front(CB_RS, 1);
    cb_pop_front(CB_OROWS, HT);

    cb_wait_front(CB_NRM, HT);
    reconfig_data_format(CB_NRM, CB_W);
    mul_bcast_rows_init(CB_NRM, CB_W);
    pack_reconfig_data_format(CB_NRMW);
    cb_reserve_back(CB_NRMW, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(CB_NRM, CB_W, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_NRMW);
        tile_regs_release();
    }
    cb_push_back(CB_NRMW, HT);
    cb_pop_front(CB_NRM, HT);

    reconfig_data_format_srca(CB_Z);
    copy_init(CB_Z);
    sigmoid_tile_init<false>();
    pack_reconfig_data_format(CB_SIG);
    cb_reserve_back(CB_SIG, HT);
    for (uint32_t j = 0; j < HT; ++j) {
        tile_regs_acquire();
        copy_tile(CB_Z, j, 0);
        sigmoid_tile<VectorMode::RC, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SIG);
        tile_regs_release();
    }
    cb_push_back(CB_SIG, HT);
    cb_pop_front(CB_Z, HT);

    cb_wait_front(CB_NRMW, HT);
    cb_wait_front(CB_SIG, HT);
    pack_reconfig_data_format(CB_OUTG);
    cb_reserve_back(CB_OUTG, HT);
    mul_binary_tile_init();
    for (uint32_t j = 0; j < HT; ++j) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_NRMW);
        copy_init(CB_NRMW);
        copy_tile(CB_NRMW, j, 0);
        reconfig_data_format_srca(CB_SIG);
        copy_init(CB_SIG);
        copy_tile(CB_SIG, j, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUTG);
        tile_regs_release();
    }
    cb_push_back(CB_OUTG, HT);
    cb_pop_front(CB_NRMW, HT);
    cb_pop_front(CB_SIG, HT);
}
}  // namespace

void kernel_main() {
    const uint32_t items = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_S, CB_T, CB_CONVSUM);
    cb_wait_front(CB_SCALER, 2);  // reader-built: tile 0 = 1.0 (sums), tile 1 = 1/128 (means)

    for (uint32_t item = 0; item < items; ++item) {
        {
            FUSED_ZONE("fz_gsc_c_wait");
            cb_wait_front(CB_P, 1);  // the first tile's group; the conv waits for the rest tile by tile
            cb_wait_front(CB_S, 3);
            cb_wait_front(CB_T, 4);
        }
        {
            FUSED_ZONE("fz_gsc_c_conv_silu");
            conv_silu();
        }
        cb_pop_front(CB_P, NT);
        cb_pop_front(CB_S, 3 * NT);
        cb_pop_front(CB_T, 4 * NT);

        cb_wait_front(CB_QKV, QK_TILES);
        {
            FUSED_ZONE("fz_gsc_c_l2_norms");
            l2_norm(0, CB_QROW);
            l2_norm(HT, CB_KROW);
        }
        cb_pop_front(CB_QKV, QK_TILES);  // the item's v tiles are the front for every row
        {
            FUSED_ZONE("fz_gsc_c_gates");
            gates_all();
        }
        cb_pop_front(CB_AB, 2);
        cb_pop_front(CB_DTNA, 2);
#ifdef DEBUG_TAPS
        tap_gate_tiles();
#endif
        cb_wait_front(CB_QKV, VBT);
        cb_wait_front(CB_QROW, HT);
        {
            FUSED_ZONE("fz_gsc_c_kcol");
            k_columns();
        }
        cb_wait_front(CB_KCOL, HT);
        cb_wait_front(CB_MASK, ROWS);

        for (uint32_t row = 0; row < ROWS; ++row) {
            cb_wait_front(CB_BETA, row + 1);  // the reader replicates the gate tiles row by row
            cb_wait_front(CB_DECAY, row + 1);
            {
                FUSED_ZONE("fz_gsc_c_recurrence");
                recurrence(row, row == 0);
            }
        }
        // the state after every row: to the writer (prefix state ROWS-1)
        cb_wait_front(CB_SNEWC, ST);
        cb_pop_front(CB_SNEWC, ST);
        cb_push_back(CB_OUTS, ST);
        {
            FUSED_ZONE("fz_gsc_c_gated_norm");
            gated_norm();
        }
        cb_pop_front(CB_QKV, VBT);
        cb_pop_front(CB_QROW, HT);
        cb_pop_front(CB_KROW, HT);
        cb_pop_front(CB_KCOL, HT);
        cb_pop_front(CB_MASK, ROWS);
        cb_pop_front(CB_BETA, ROWS);
        cb_pop_front(CB_DECAY, ROWS);
    }
}
