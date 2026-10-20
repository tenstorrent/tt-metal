// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC pre-block for one token-tile (decode, T <= 32), one program on the whole grid, exact fp32:
//   mixes = (x @ fn_T) * rsqrt(sum x^2 + eps*nC)      [sqrt(nC) folded into fn]
//   pre, post, comb = Sinkhorn(mixes)                  (the mhc_split_sinkhorn kernel's tile sequence)
//   h = sum_i pre_i x_i ;  hn = h * rsqrt(mean_c h^2 + eps) * w   -> bf16
// Core k < Ct owns column tile c = k of the C-wide streams (the 4 stream tiles x[:, i, c]); the projection
// contraction over n*C = 4 * 128 tiles is split by column: each core multiplies its 4 x tiles by the matching 4
// fn tiles (x split into 3 bf16 parts so every product is exact, fp32 accumulation) and adds its rows' sum of
// squares into column 24 of the same partial tile; the partials are summed exactly on the SFPU up a two-level
// tree (column leaders (x, 0), then the coordinator (0, 0)); the coordinator finishes mixes, runs the Sinkhorn,
// expands pre / post / comb into lane-layout tiles (post / comb go to DRAM for the post-block kernel) and
// multicasts the pre tile; every core mixes h = sum_i pre_i x_i, sends its rows' sum of squares up the same
// tree, receives rsqrt(mean + eps) by multicast and writes hn * w as bf16. Dataflow kernels own the NoC sync.
//
// Dest slots (SyncFull fp32): 0..3 x_i (or scratch), 4..7 scratch. See the README for the CB plan.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "mhc_sfpu_x8.h"

namespace {

// ---- compile-time args: CB ids
constexpr uint32_t CB_X = get_compile_time_arg_val(0);        // 4 fp32 stream tiles of this column
constexpr uint32_t CB_FN = get_compile_time_arg_val(1);       // 4 bf16 projection tiles
constexpr uint32_t CB_HI = get_compile_time_arg_val(2);       // bf16 parts of x (hi)
constexpr uint32_t CB_LO = get_compile_time_arg_val(3);       //   (lo)
constexpr uint32_t CB_LO2 = get_compile_time_arg_val(4);      //   (lo2)
constexpr uint32_t CB_ZERO = get_compile_time_arg_val(5);     // one zero tile (never popped)
constexpr uint32_t CB_PART = get_compile_time_arg_val(6);     // partial / column-sum tiles out (to the writer)
constexpr uint32_t CB_GATHER1 = get_compile_time_arg_val(7);  // leaders: the column's partials (pushed by the reader)
constexpr uint32_t CB_GATHER2 = get_compile_time_arg_val(8);  // coordinator: the leaders' sums
constexpr uint32_t CB_PRE = get_compile_time_arg_val(9);      // multicast: pre lane tile
constexpr uint32_t CB_RS2 = get_compile_time_arg_val(10);     // multicast: rsqrt(mean h^2 + eps) tile
constexpr uint32_t CB_W = get_compile_time_arg_val(11);       // bf16 norm weight rows of this column
constexpr uint32_t CB_CONSTS = get_compile_time_arg_val(12);  // Sinkhorn constants (8 tiles)
constexpr uint32_t CB_MIXES = get_compile_time_arg_val(13);   // coordinator: mixes tile
constexpr uint32_t CB_SPRE = get_compile_time_arg_val(14);    // Sinkhorn raw outputs
constexpr uint32_t CB_SPOST = get_compile_time_arg_val(15);
constexpr uint32_t CB_SCOMB = get_compile_time_arg_val(16);
constexpr uint32_t CB_MA = get_compile_time_arg_val(17);  // Sinkhorn scratch
constexpr uint32_t CB_MB = get_compile_time_arg_val(18);
constexpr uint32_t CB_RS = get_compile_time_arg_val(19);
constexpr uint32_t CB_RECIP = get_compile_time_arg_val(20);
constexpr uint32_t CB_TMP = get_compile_time_arg_val(21);
constexpr uint32_t CB_COEF_OUT = get_compile_time_arg_val(22);  // coordinator: pre, post, comb_0..n-1 lane tiles
constexpr uint32_t CB_H = get_compile_time_arg_val(23);         // this core's h tile (fp32)
constexpr uint32_t CB_HSQ = get_compile_time_arg_val(24);       // this core's row sums of h^2 (all columns)
constexpr uint32_t CB_RS2OUT = get_compile_time_arg_val(25);    // coordinator: the rsqrt tile to multicast
constexpr uint32_t CB_OUT = get_compile_time_arg_val(26);       // bf16 hn tile
constexpr uint32_t CB_PARTS = get_compile_time_arg_val(35);     // per-column partials (summed before CB_PART)
constexpr uint32_t CB_DBG = get_compile_time_arg_val(36);       // DEBUG: the coordinator's raw sum and mixes tiles
constexpr bool DEBUG = get_compile_time_arg_val(37) != 0;       // DEBUG: comb tiles 2, 3 <- raw sum, mixes
constexpr bool EXACT_PROJ = get_compile_time_arg_val(38) != 0;  // SFPU projection (row-replicated fn tiles)
constexpr uint32_t CB_MASK = get_compile_time_arg_val(39);      // EXACT_PROJ: the lane-mask tile
constexpr uint32_t FREP = 3;                                    // row-replicated fn tiles per stream tile
// ---- compile-time args: parameters
constexpr uint32_t N = get_compile_time_arg_val(27);
constexpr uint32_t GY = get_compile_time_arg_val(28);             // cores per column (partials per leader)
constexpr uint32_t GX = get_compile_time_arg_val(29);             // columns (leader sums at the coordinator)
constexpr uint32_t ITERS = get_compile_time_arg_val(30);          // Sinkhorn iterations
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(31);       // Sinkhorn eps
constexpr uint32_t EPS_NC_BITS = get_compile_time_arg_val(32);    // norm_eps * nC (the projection rsqrt)
constexpr uint32_t INV_C_BITS = get_compile_time_arg_val(33);     // 1 / C
constexpr uint32_t NORM_EPS_BITS = get_compile_time_arg_val(34);  // the sublayer norm eps
constexpr uint32_t MAX_COLS = 2;  // column tiles per core (128 column tiles over 120 cores)
static_assert(N == 4, "the lane layout and the dest slot plan are written for n = 4");

// Sinkhorn constants: tile indices within CB_CONSTS (build_consts order)
constexpr uint32_t SEL_PRE = 0, SEL_POST = 1, SEL_COMB = 2, BASE_PRE = 3, BASE_POST = 4, BASE_COMB = 5, RB = 6, CBC = 7;
constexpr uint32_t TWO_BITS = 0x40000000u;
constexpr uint32_t SS_GROUP = 2, SS_LANE = 4;  // column 24 of the partial tile carries the sum of squares

FORCE_INLINE void release_keep_dest() { PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>())); }

// the packer's output format follows the CB it packs into (bf16 and fp32 CBs alternate in this kernel)
FORCE_INLINE void pack_to(uint32_t cb) {
    static uint32_t last = 0xFFFFFFFFu;
    if (cb != last) {
        pack_reconfig_data_format(cb);
        last = cb;
    }
}

// pack dest tile `slot` into `cb` (one tile, pushed)
FORCE_INLINE void pack_one(uint32_t slot, uint32_t cb) {
    CircularBuffer c(cb);
    pack_to(cb);
    c.reserve_back(1);
    pack_tile(slot, cb);
    c.push_back(1);
}

// ------------------------------------------------------------------ the Sinkhorn (mhc_split_sinkhorn's sequence)
FORCE_INLINE void mm(uint32_t a, uint32_t b_tile, uint32_t out) {
    CircularBuffer cb_a(a);
    cb_a.wait_front(1);
    reconfig_data_format(a, CB_CONSTS);
    matmul_init(a, CB_CONSTS);
    tile_regs_acquire();
    matmul_tiles(a, CB_CONSTS, 0, b_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_one(0, out);
    tile_regs_release();
}

FORCE_INLINE void add_bias(uint32_t a, uint32_t b_tile, uint32_t out) {
    CircularBuffer cb_a(a);
    cb_a.wait_front(1);
    reconfig_data_format(a, CB_CONSTS);
    add_init(a, CB_CONSTS);
    tile_regs_acquire();
    add_tiles(a, CB_CONSTS, 0, b_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_one(0, out);
    tile_regs_release();
    cb_a.pop_front(1);
}

FORCE_INLINE void mul(uint32_t a, uint32_t b, uint32_t out) {
    CircularBuffer cb_a(a), cb_b(b);
    cb_a.wait_front(1);
    cb_b.wait_front(1);
    reconfig_data_format(a, b);
    mul_init(a, b);
    tile_regs_acquire();
    mul_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_one(0, out);
    tile_regs_release();
    cb_a.pop_front(1);
    cb_b.pop_front(1);
}

constexpr uint32_t OP_EXP = 0, OP_SIGMOID = 1, OP_RECIP = 2, OP_ADD_EPS = 3, OP_MUL2 = 4, OP_RECIP_EPS = 5, OP_COPY = 6;

FORCE_INLINE void unary(uint32_t a, uint32_t op, uint32_t scalar_bits, uint32_t out) {
    CircularBuffer cb_a(a);
    cb_a.wait_front(1);
    reconfig_data_format_srca(a);
    copy_init(a);
    tile_regs_acquire();
    copy_tile(a, 0, 0);
    if (op == OP_EXP) {
        unary_min_tile_init();
        unary_min_tile(0, 0x42a00000u);  // min(x, 80.0f): no inf out of exp
        exp_tile_init();
        exp_tile(0);
    } else if (op == OP_SIGMOID) {
        sigmoid_tile_init();
        sigmoid_tile(0);
    } else if (op == OP_RECIP) {
        recip_tile_init();
        recip_tile(0);
    } else if (op == OP_RECIP_EPS) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, scalar_bits);
        recip_tile_init();
        recip_tile(0);
    } else if (op == OP_ADD_EPS) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, scalar_bits);
    } else if (op == OP_MUL2) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scalar_bits);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_one(0, out);
    tile_regs_release();
    cb_a.pop_front(1);
}

FORCE_INLINE void normalize(uint32_t m_in, uint32_t k_tile, bool use_eps, uint32_t m_out) {
    mm(m_in, k_tile, CB_RS);
    unary(CB_RS, use_eps ? OP_RECIP_EPS : OP_RECIP, EPS_BITS, CB_RECIP);
    mul(m_in, CB_RECIP, m_out);
}

// CB_MIXES (1 tile, kept) -> CB_SPRE, CB_SPOST, CB_SCOMB (raw coefficient tiles)
FORCE_INLINE void sinkhorn() {
    CircularBuffer cb_mixes(CB_MIXES);
    cb_mixes.wait_front(1);
    mm(CB_MIXES, SEL_COMB, CB_TMP);
    add_bias(CB_TMP, BASE_COMB, CB_MA);
    unary(CB_MA, OP_EXP, 0, CB_MB);
    normalize(CB_MB, RB, false, CB_MA);
    unary(CB_MA, OP_ADD_EPS, EPS_BITS, CB_MB);
    normalize(CB_MB, CBC, true, CB_MA);
    for (uint32_t i = 1; i < ITERS; ++i) {
        normalize(CB_MA, RB, true, CB_MB);
        normalize(CB_MB, CBC, true, CB_MA);
    }
    unary(CB_MA, OP_COPY, 0, CB_SCOMB);
    mm(CB_MIXES, SEL_PRE, CB_TMP);
    add_bias(CB_TMP, BASE_PRE, CB_MA);
    unary(CB_MA, OP_SIGMOID, 0, CB_MB);
    unary(CB_MB, OP_ADD_EPS, EPS_BITS, CB_SPRE);
    mm(CB_MIXES, SEL_POST, CB_TMP);
    add_bias(CB_TMP, BASE_POST, CB_MA);
    unary(CB_MA, OP_SIGMOID, 0, CB_MB);
    unary(CB_MB, OP_MUL2, TWO_BITS, CB_SPOST);
    cb_mixes.pop_front(1);
}

// ------------------------------------------------------------------ phases
// x (fp32) -> bf16 parts hi, lo, lo2 with x == hi + lo + lo2 exactly (each pack rounds to bf16, the residuals
// are exact fp32 subtractions). CB_X is never popped: x stays for the mix.
FORCE_INLINE void split_x(uint32_t col) {
    CircularBuffer cbx(CB_X), cbhi(CB_HI), cblo(CB_LO), cblo2(CB_LO2);
    const uint32_t x0 = col * N;  // this column's tiles within CB_X
    // hi = bf16(x)
    tile_regs_acquire();
    copy_init(CB_X);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_X, x0 + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(CB_HI);
    cbhi.reserve_back(N);
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile(i, CB_HI, i);
    }
    cbhi.push_back(N);
    tile_regs_release();
    // lo = bf16(x - hi)
    cbhi.wait_front(N);
    tile_regs_acquire();
    copy_init(CB_X);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_X, x0 + i, i);
    }
    reconfig_data_format_srca(CB_X, CB_HI);
    copy_init(CB_HI);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_HI, i, N + i);
    }
    sub_binary_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        sub_binary_tile(i, N + i, N + i);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(CB_LO);
    cblo.reserve_back(N);
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile(N + i, CB_LO, i);
    }
    cblo.push_back(N);
    tile_regs_release();
    // lo2 = bf16(x - hi - lo)
    cblo.wait_front(N);
    tile_regs_acquire();
    reconfig_data_format_srca(CB_HI, CB_X);
    copy_init(CB_X);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_X, x0 + i, i);
    }
    reconfig_data_format_srca(CB_X, CB_HI);
    copy_init(CB_HI);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_HI, i, N + i);
    }
    sub_binary_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        sub_binary_tile(i, N + i, N + i);  // r1 = x - hi
    }
    copy_init(CB_LO);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_LO, i, i);
    }
    sub_binary_tile_init();
    for (uint32_t i = 0; i < N; ++i) {
        sub_binary_tile(N + i, i, N + i);  // r2 = r1 - lo
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(CB_LO2);
    cblo2.reserve_back(N);
    for (uint32_t i = 0; i < N; ++i) {
        pack_tile(N + i, CB_LO2, i);
    }
    cblo2.push_back(N);
    tile_regs_release();
    cblo2.wait_front(N);
    reconfig_data_format_srca(CB_HI, CB_X);
}

// partial tile = sum over the 3 parts and the 4 streams of part_i @ fn_i (fp32 accumulation of exact products)
// + the rows' sum of x^2 in column 24 -> CB_PART
FORCE_INLINE void project_partial(uint32_t col) {
    CircularBuffer cbfn(CB_FN), cbzero(CB_ZERO), cbhi(CB_HI), cblo(CB_LO), cblo2(CB_LO2);
    const uint32_t f0 = col * N;
    cbzero.wait_front(1);
    tile_regs_acquire();
    copy_init(CB_ZERO);
    copy_tile(CB_ZERO, 0, 4);  // the accumulator starts at zero
    for (uint32_t part = 0; part < 3; ++part) {
        const uint32_t cb = part == 0 ? CB_HI : (part == 1 ? CB_LO : CB_LO2);
        reconfig_data_format(cb, CB_FN);
        matmul_init(cb, CB_FN);
        for (uint32_t i = 0; i < N; ++i) {
            matmul_tiles(cb, CB_FN, i, f0 + i, 4);
        }
    }
    // the rows' sum of squares of the fp32 x into column 24
    reconfig_data_format_srca(CB_HI, CB_X);
    copy_init(CB_X);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_X, f0 + i, i);
    }
    sfpu_bcast_col_init();
    MATH((_llk_math_eltwise_binary_sfpu_params_(
        mhc_sfpu::rowsum_sq_into, 4, 0, 4, VectorMode::None, N, SS_GROUP, SS_LANE)));
    tile_regs_commit();
    tile_regs_wait();
    pack_one(4, CB_PARTS);
    tile_regs_release();
    cbhi.pop_front(N);
    cblo.pop_front(N);
    cblo2.pop_front(N);
}

// EXACT_PROJ: the partial of column `col` with the projection on the SFPU: for every stream tile, x_i (slot 0)
// against its three row-replicated fn tiles (slots 1-3) into the accumulator slot 4; then the sum of squares
FORCE_INLINE void project_partial_exact(uint32_t col) {
    CircularBuffer cbfn(CB_FN), cbzero(CB_ZERO), cbmask(CB_MASK);
    cbzero.wait_front(1);
    cbmask.wait_front(1);
    tile_regs_acquire();
    copy_init(CB_ZERO);
    copy_tile(CB_ZERO, 0, 4);  // the accumulator starts at zero
    copy_init(CB_MASK);
    copy_tile(CB_MASK, 0, 5);
    sfpu_bcast_col_init();
    for (uint32_t i = 0; i < N; ++i) {
        reconfig_data_format_srca(CB_FN, CB_X);
        copy_init(CB_X);
        copy_tile(CB_X, col * N + i, 0);
        reconfig_data_format_srca(CB_X, CB_FN);
        copy_init(CB_FN);
        for (uint32_t q = 0; q < FREP; ++q) {
            copy_tile(CB_FN, (col * N + i) * FREP + q, 1 + q);
        }
        MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::proj_exact_tile, 4, 0, 4, VectorMode::None, 1, 5)));
    }
    // the rows' sum of squares of x into column 24
    reconfig_data_format_srca(CB_FN, CB_X);
    copy_init(CB_X);
    for (uint32_t i = 0; i < N; ++i) {
        copy_tile(CB_X, col * N + i, i);
    }
    MATH((_llk_math_eltwise_binary_sfpu_params_(
        mhc_sfpu::rowsum_sq_into, 4, 0, 4, VectorMode::None, N, SS_GROUP, SS_LANE)));
    tile_regs_commit();
    tile_regs_wait();
    pack_one(4, CB_PARTS);
    tile_regs_release();
}

// sum of `count` tiles of `cb` (exact SFPU adds) -> `out_cb`
FORCE_INLINE void sum_tiles(uint32_t cb, uint32_t count, uint32_t out_cb, uint32_t also_cb = 0xFFFFFFFFu) {
    CircularBuffer c(cb);
    c.wait_front(count);
    tile_regs_acquire();
    copy_init(cb);
    copy_tile(cb, 0, 0);
    add_binary_tile_init();
    for (uint32_t k = 1; k < count; ++k) {
        copy_tile(cb, k, 1);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_one(0, out_cb);
    if (also_cb != 0xFFFFFFFFu) {
        pack_one(0, also_cb);
    }
    tile_regs_release();
    c.pop_front(count);
}

// coordinator: CB_GATHER2 (GX leader sums) -> mixes (CB_MIXES) -> Sinkhorn -> lane tiles (CB_COEF_OUT)
FORCE_INLINE void coordinator_coefficients() {
    CircularBuffer g2(CB_GATHER2);
    g2.wait_front(GX);
    tile_regs_acquire();
    copy_init(CB_GATHER2);
    copy_tile(CB_GATHER2, 0, 0);
    add_binary_tile_init();
    for (uint32_t k = 1; k < GX; ++k) {
        copy_tile(CB_GATHER2, k, 1);
        add_binary_tile(0, 1, 0);
    }
    if constexpr (DEBUG) {  // the raw sum (mixes_un columns 0-23, sum of squares in column 24)
        tile_regs_commit();
        tile_regs_wait();
        pack_one(0, CB_DBG);
        release_keep_dest();
        tile_regs_acquire();
    }
    // rs = rsqrt(ss + eps*nC) from column 24 (on a copy), then every row of the sum scaled by it
    copy_dest_values<DataFormat::Float32>(0, 1);
    binop_with_scalar_tile_init();
    add_unary_tile(1, EPS_NC_BITS);
    rsqrt_tile_init();
    rsqrt_tile<RsqrtMode::Default>(1);
    sfpu_bcast_col_init();
    MATH((_llk_math_eltwise_binary_sfpu_params_(
        mhc_sfpu::scale_rows_from_lane, 0, 1, 0, VectorMode::None, SS_GROUP, SS_LANE)));
    tile_regs_commit();
    tile_regs_wait();
    pack_one(0, CB_MIXES);
    if constexpr (DEBUG) {
        pack_one(0, CB_DBG);
    }
    tile_regs_release();
    g2.pop_front(GX);

    sinkhorn();

    // lane-layout tiles: pre, post, comb_0, comb_1 (one acquire), comb_2, comb_3 (another)
    CircularBuffer spre(CB_SPRE), spost(CB_SPOST), scomb(CB_SCOMB), cout(CB_COEF_OUT);
    spre.wait_front(1);
    spost.wait_front(1);
    scomb.wait_front(1);
    tile_regs_acquire();
    copy_init(CB_SPRE);
    copy_tile(CB_SPRE, 0, 0);
    copy_init(CB_SPOST);
    copy_tile(CB_SPOST, 0, 1);
    copy_init(CB_SCOMB);
    copy_tile(CB_SCOMB, 0, 2);
    sfpu_bcast_col_init();
    MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_post, 0, 3, 3, VectorMode::None, N)));
    MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_post, 1, 4, 4, VectorMode::None, N)));
    MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_comb, 2, 5, 5, VectorMode::None, N, 0)));
    MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_comb, 2, 6, 6, VectorMode::None, N, 1)));
    tile_regs_commit();
    tile_regs_wait();
    pack_to(CB_COEF_OUT);
    cout.reserve_back(N + 2);
    for (uint32_t t = 0; t < 4; ++t) {
        pack_tile(3 + t, CB_COEF_OUT, t);
    }
    tile_regs_release();
    tile_regs_acquire();
    if constexpr (DEBUG) {
        CircularBuffer dbg(CB_DBG);
        dbg.wait_front(3);
        copy_init(CB_DBG);
        copy_tile(CB_DBG, 0, 1);  // comb tile 2 <- the coordinator's own partial
        copy_tile(CB_DBG, 2, 2);  // comb tile 3 <- mixes
        dbg.pop_front(3);
    } else {
        copy_init(CB_SCOMB);
        copy_tile(CB_SCOMB, 0, 0);
        sfpu_bcast_col_init();
        MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_comb, 0, 1, 1, VectorMode::None, N, 2)));
        MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::expand_comb, 0, 2, 2, VectorMode::None, N, 3)));
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(CB_COEF_OUT);
    pack_tile(1, CB_COEF_OUT, 4);
    pack_tile(2, CB_COEF_OUT, 5);
    cout.push_back(N + 2);
    tile_regs_release();
    spre.pop_front(1);
    spost.pop_front(1);
    scomb.pop_front(1);
}

// worker: h_j = sum_i pre_i x_i per column tile (CB_H, `count` tiles) and the rows' sum of h^2 over them (CB_HSQ)
FORCE_INLINE void mix_h(uint32_t count) {
    CircularBuffer cbpre(CB_PRE);
    cbpre.wait_front(1);
    tile_regs_acquire();
    copy_init(CB_PRE);
    copy_tile(CB_PRE, 0, 5);
    sfpu_bcast_col_init();
    for (uint32_t col = 0; col < count; ++col) {
        copy_init(CB_X);
        for (uint32_t i = 0; i < N; ++i) {
            copy_tile(CB_X, col * N + i, i);
        }
        MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::pre_mix, 6 + col, 5, 6 + col, VectorMode::None, N)));
    }
    MATH((_llk_math_eltwise_binary_sfpu_params_(mhc_sfpu::rowsum_sq_fill, 5, 6, 5, VectorMode::None, count)));
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t col = 0; col < count; ++col) {
        pack_one(6 + col, CB_H);
    }
    pack_one(5, CB_HSQ);
    tile_regs_release();
    cbpre.pop_front(1);
}

// coordinator: CB_GATHER2 (GX sums of h^2, every column the row's total) -> rsqrt(total / C + eps) -> CB_RS2OUT
FORCE_INLINE void coordinator_rs2() {
    CircularBuffer g2(CB_GATHER2);
    g2.wait_front(GX);
    tile_regs_acquire();
    copy_init(CB_GATHER2);
    copy_tile(CB_GATHER2, 0, 0);
    add_binary_tile_init();
    for (uint32_t k = 1; k < GX; ++k) {
        copy_tile(CB_GATHER2, k, 1);
        add_binary_tile(0, 1, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_C_BITS);
    add_unary_tile(0, NORM_EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile<RsqrtMode::Default>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_one(0, CB_RS2OUT);
    tile_regs_release();
    g2.pop_front(GX);
}

// worker: hn_j = h_j * rs2 * w_j -> bf16 CB_OUT, per column tile
FORCE_INLINE void finish(uint32_t count) {
    CircularBuffer cbh(CB_H), cbrs(CB_RS2), cbw(CB_W), cbx(CB_X), cbfn(CB_FN);
    cbh.wait_front(count);
    cbrs.wait_front(1);
    cbw.wait_front(count);
    for (uint32_t col = 0; col < count; ++col) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_W, CB_H);
        copy_init(CB_H);
        copy_tile(CB_H, col, 0);
        copy_init(CB_RS2);
        copy_tile(CB_RS2, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(CB_RS2, CB_W);
        copy_init(CB_W);
        copy_tile(CB_W, col, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_one(0, CB_OUT);
        tile_regs_release();
    }
    cbh.pop_front(count);
    cbrs.pop_front(1);
    cbw.pop_front(count);
    cbx.pop_front(N * count);
    cbfn.pop_front(N * count * (EXACT_PROJ ? FREP : 1));
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_OUT);
    const uint32_t count = get_arg_val<uint32_t>(0);  // column tiles of this core (0: idle)
    const bool leader = get_arg_val<uint32_t>(1) != 0;
    const bool coordinator = get_arg_val<uint32_t>(2) != 0;
    const bool worker = count != 0;

    if (worker) {
        CircularBuffer cbx(CB_X), cbfn(CB_FN);
        cbx.wait_front(N * count);
        cbfn.wait_front(N * count * (EXACT_PROJ ? FREP : 1));
        for (uint32_t col = 0; col < count; ++col) {
            if constexpr (EXACT_PROJ) {
                project_partial_exact(col);
            } else {
                split_x(col);
                project_partial(col);
            }
        }
        sum_tiles(CB_PARTS, count, CB_PART, (DEBUG && coordinator) ? CB_DBG : 0xFFFFFFFFu);  // DEBUG: own partial
    }
    if (leader) {
        sum_tiles(CB_GATHER1, GY, CB_PART);
    }
    if (coordinator) {
        coordinator_coefficients();
    }
    if (worker) {
        mix_h(count);
    }
    if (leader) {
        sum_tiles(CB_GATHER1, GY, CB_PART);
    }
    if (coordinator) {
        coordinator_rs2();
    }
    if (worker) {
        finish(count);
    }
}
