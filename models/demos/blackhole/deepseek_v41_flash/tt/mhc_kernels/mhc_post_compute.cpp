// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mHC post kernel: sum of the K-chunk partials, RMS normalisation of the mixes (column NCOL holds the sum of squares),
// then the deepseek_prefill mhc_split_sinkhorn parametrisation (copied from mhc_split_sinkhorn_compute.cpp). One core.
// Original description: Fused mHC parametrization on one core. Per token-tile (up to 32 tokens along tile rows):
//   comb = Sinkhorn( exp( mixes @ SEL_comb + base_comb ) )   via row/col-sum matmuls RB/CB
//   pre  = sigmoid( mixes @ SEL_pre  + base_pre  ) + eps
//   post = 2 * sigmoid( mixes @ SEL_post + base_post )
// SEL_* extract+left-align each group from the 24-wide mixes and fold in the scalar a; the
// Sinkhorn normalizations are same-shape `m / (m @ K)` divides (K = RB row-sum / CB col-sum),
// so the whole op is matmul + SFPU tiles. Validated bit-for-bit against the torch reference.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t CB_RAW = 0;     // c_0  (summed raw mixes)
constexpr uint32_t CB_P = 5;       // c_5  (K-chunk partial tiles)
constexpr uint32_t CB_MIXES = 6;   // c_6  (normalised mixes: what the body below reads)
constexpr uint32_t CB_CONSTS = 1;  // c_1  (8 resident tiles)
constexpr uint32_t CB_PRE = 2;     // c_2  out
constexpr uint32_t CB_POST = 3;    // c_3  out
constexpr uint32_t CB_COMB = 4;    // c_4  out
constexpr uint32_t CB_MA = 16;     // m ping
constexpr uint32_t CB_MB = 17;     // m pong
constexpr uint32_t CB_RS = 18;     // row/col-sum broadcast
constexpr uint32_t CB_RECIP = 24;  // reciprocal
constexpr uint32_t CB_TMP = 25;    // projection scratch

// const-tile indices within CB_CONSTS
constexpr uint32_t SEL_PRE = 0, SEL_POST = 1, SEL_COMB = 2;
constexpr uint32_t BASE_PRE = 3, BASE_POST = 4, BASE_COMB = 5;
constexpr uint32_t RB = 6, CB_COL = 7;
constexpr uint32_t SEL_SS = 8, IDENT = 9;  // extra const tiles: column NCOL replicated to all columns / identity

constexpr uint32_t TWO_BITS = 0x40000000u;  // 2.0f

// CB flow-control (wait/reserve/push/pop) goes through Device 2.0 CircularBuffer objects; the
// compute LLK ops (matmul_tiles/pack_tile/add_tiles/...) still take the raw CB index, so both
// the index and the object are kept in scope. b_tile is a tile index within CB_CONSTS, not a CB.
// The helpers below wrap the acquire/commit/pack/release LLK sequence every tile op repeats;
// there is no shared eltwise-helper layer to call into yet, so they are local.

// out <- a @ consts[b_tile].  Reads a[0] and consts[b_tile]; pops neither.
FORCE_INLINE void mm(uint32_t a, uint32_t b_tile, uint32_t out) {
    CircularBuffer cb_a(a), cb_out(out);
    cb_a.wait_front(1);
    reconfig_data_format(a, CB_CONSTS);
    matmul_init(a, CB_CONSTS);
    tile_regs_acquire();
    matmul_tiles(a, CB_CONSTS, 0, b_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_out.reserve_back(1);
    pack_tile(0, out);
    cb_out.push_back(1);
    tile_regs_release();
}

// out <- a + consts[b_tile].  Pops a.
FORCE_INLINE void add_bias(uint32_t a, uint32_t b_tile, uint32_t out) {
    CircularBuffer cb_a(a), cb_out(out);
    cb_a.wait_front(1);
    reconfig_data_format(a, CB_CONSTS);
    add_tiles_init(a, CB_CONSTS);
    tile_regs_acquire();
    add_tiles(a, CB_CONSTS, 0, b_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_out.reserve_back(1);
    pack_tile(0, out);
    cb_out.push_back(1);
    tile_regs_release();
    cb_a.pop_front(1);
}

// out <- a * b (elementwise).  Pops a and b.
FORCE_INLINE void mul(uint32_t a, uint32_t b, uint32_t out) {
    CircularBuffer cb_a(a), cb_b(b), cb_out(out);
    cb_a.wait_front(1);
    cb_b.wait_front(1);
    reconfig_data_format(a, b);
    mul_tiles_init(a, b);
    tile_regs_acquire();
    mul_tiles(a, b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_out.reserve_back(1);
    pack_tile(0, out);
    cb_out.push_back(1);
    tile_regs_release();
    cb_a.pop_front(1);
    cb_b.pop_front(1);
}

// SFPU op ids for unary()
constexpr uint32_t OP_EXP = 0, OP_SIGMOID = 1, OP_RECIP = 2, OP_ADD_EPS = 3, OP_MUL2 = 4, OP_RECIP_EPS = 5, OP_COPY = 6,
                   OP_RSQRT_EPS = 7;

// out <- op(a).  Pops a.  scalar_bits used by the *_EPS variants.
FORCE_INLINE void unary(uint32_t a, uint32_t op, uint32_t scalar_bits, uint32_t out) {
    CircularBuffer cb_a(a), cb_out(out);
    cb_a.wait_front(1);
    reconfig_data_format_srca(a);
    copy_tile_to_dst_init_short(a);
    tile_regs_acquire();
    copy_tile(a, 0, 0);
    if (op == OP_EXP) {
        // Cap logits before exp so an out-of-range learned logit cannot overflow to inf
        // (which becomes inf/inf = NaN in the softmax divide). The softmax ratio is unchanged
        // for any realistic logit (a_res init ~ 0.01 => |logit| << 1) -- equivalent to the
        // canonical row-max subtraction within range, with no cross-lane reduce.
        unary_min_tile_init();
        unary_min_tile(0, 0x42a00000u);  // min(x, 80.0f)
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
    } else if (op == OP_RSQRT_EPS) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, scalar_bits);
        rsqrt_tile_init();
        rsqrt_tile(0);
    } else if (op == OP_ADD_EPS) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, scalar_bits);
    } else if (op == OP_MUL2) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scalar_bits);
    }  // OP_COPY: copy_tile only, no SFPU

    tile_regs_commit();
    tile_regs_wait();
    cb_out.reserve_back(1);
    pack_tile(0, out);
    cb_out.push_back(1);
    tile_regs_release();
    cb_a.pop_front(1);
}

// m_out <- m_in / (m_in @ K [+ eps]).  m_in is popped (consumed by the final mul).
// The row/col sum, the (+eps) and the reciprocal run in ONE DST acquire (no CB round trip for RS).
FORCE_INLINE void normalize(uint32_t m_in, uint32_t k_tile, bool use_eps, uint32_t eps_bits, uint32_t m_out) {
    CircularBuffer cb_a(m_in);
    cb_a.wait_front(1);
    reconfig_data_format(m_in, CB_CONSTS);
    matmul_init(m_in, CB_CONSTS);
    tile_regs_acquire();
    matmul_tiles(m_in, CB_CONSTS, 0, k_tile, 0);
    if (use_eps) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);
    }
    recip_tile_init();
    recip_tile(0);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(m_in);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(m_in, 0, 0);  // dst = (1/rs) * m_in
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer cb_o(m_out);
    cb_o.reserve_back(1);
    pack_tile(0, m_out);
    cb_o.push_back(1);
    tile_regs_release();
    cb_a.pop_front(1);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_RAW, CB_CONSTS, CB_COMB);  // must precede any other compute work
    constexpr uint32_t iters = get_compile_time_arg_val(0);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(1);
    constexpr uint32_t nparts = get_compile_time_arg_val(2);
    constexpr uint32_t ss_eps_bits = get_compile_time_arg_val(3);  // N * norm_eps

    CircularBuffer cb_consts(CB_CONSTS), cb_mixes(CB_MIXES), cb_p(CB_P), cb_raw(CB_RAW);
    cb_consts.wait_front(10);  // constants resident
    cb_p.wait_front(nparts);

    // ---- raw = sum_s P_s  (identity matmuls accumulate in DST) ----
    reconfig_data_format(CB_CONSTS, CB_P);
    matmul_init(CB_CONSTS, CB_P);
    tile_regs_acquire();
    for (uint32_t s = 0; s < nparts; ++s) {
        matmul_tiles(CB_CONSTS, CB_P, IDENT, s, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_raw.reserve_back(1);
    pack_tile(0, CB_RAW);
    cb_raw.push_back(1);
    tile_regs_release();

    // ---- mixes = raw * rsqrt(ss + N*eps)   (sqrt(N) is folded into the SEL tiles) ----
    mm(CB_RAW, SEL_SS, CB_TMP);  // every column = ss
    unary(CB_TMP, OP_RSQRT_EPS, ss_eps_bits, CB_RECIP);
    mul(CB_RAW, CB_RECIP, CB_MIXES);  // pops RAW and RECIP

    cb_mixes.wait_front(1);  // reused by comb/pre/post; popped at the end

    // ---- comb = Sinkhorn(exp(mixes @ SEL_comb + base_comb)) ----
    mm(CB_MIXES, SEL_COMB, CB_TMP);
    add_bias(CB_TMP, BASE_COMB, CB_MA);
    unary(CB_MA, OP_EXP, 0, CB_MB);                   // m = exp(logits)  -> MB
    normalize(CB_MB, RB, false, 0, CB_MA);            // row softmax (no eps) -> MA
    unary(CB_MA, OP_ADD_EPS, eps_bits, CB_MB);        // m = softmax + eps -> MB
    normalize(CB_MB, CB_COL, true, eps_bits, CB_MA);  // first column norm -> MA
    for (uint32_t i = 1; i < iters; ++i) {
        normalize(CB_MA, RB, true, eps_bits, CB_MB);      // row -> MB
        normalize(CB_MB, CB_COL, true, eps_bits, CB_MA);  // col -> MA
    }
    unary(CB_MA, OP_COPY, 0, CB_COMB);  // comb out = m

    // ---- pre = sigmoid(mixes @ SEL_pre + base_pre) + eps ----
    mm(CB_MIXES, SEL_PRE, CB_TMP);
    add_bias(CB_TMP, BASE_PRE, CB_MA);
    unary(CB_MA, OP_SIGMOID, 0, CB_MB);
    unary(CB_MB, OP_ADD_EPS, eps_bits, CB_PRE);

    // ---- post = 2 * sigmoid(mixes @ SEL_post + base_post) ----
    mm(CB_MIXES, SEL_POST, CB_TMP);
    add_bias(CB_TMP, BASE_POST, CB_MA);
    unary(CB_MA, OP_SIGMOID, 0, CB_MB);
    unary(CB_MB, OP_MUL2, TWO_BITS, CB_POST);

    cb_mixes.pop_front(1);
}
