// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident attention compute, see attn_common.hpp. Two builds: the leaders' (kLeader, HiFi4) and the other
// cores' (HiFi2); DEST fp32, full sync (8 tiles). Per layer:
//   prep (L_g sliding, L_0 full; as tt/decoder.py decode_forward): q = rope(rmsnorm(q) w_q) -> cb_q;
//     sliding k = rope(rmsnorm(k) w_k), v = rmsnorm(v); full v = rmsnorm(kv), k = rope(v w_k); K / V rounded to bf16,
//     then packed BF8 (the cache's format) for the reader's splice. RoPE: rotate-half with the decode tables' rows.
//   attention over n position tiles (tail: the spliced tiles, masked; worker: its DRAM tiles, the first two window
//     tiles masked when needed): S = Q K^T (scale 1), m = rowmax S, P = exp(S - m), l = rowsum P, O = P V -> the
//     partial (O, m, l) in cb_own. Rows of different heads / positions never mix.
//   merge (roots, mergers with children): M = max m_x, s_x = exp(m_x - M), L = sum s_x l_x (SFPU, fp32), then
//     O = sum diag(w_x) O_x as one matmul (w = s / L at the root -> the final O; w = s at a merger -> (O, M, L)).
//
// Runtime args: 0 a, 1 g, 2 j, 3 full worker index (~0: none), 4 parts pages, 5 .. 15 the full-layer tree (Tree)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "attn_common.hpp"

using namespace gatt;

namespace {

constexpr bool kLeader = get_compile_time_arg_val(16) != 0;
constexpr uint32_t kDst = 8;
// every CB is pushed and popped in whole capacities (or single pages), so no reservation straddles its end
constexpr uint32_t kStat = 2;  // cb_m / cb_l pages
constexpr uint32_t PTmax = PTs > PTf ? PTs : PTf;

// M3_2 (F5v): the merges' SFPU vector mode. The merges read only column 0 of the m / l / s / w tiles (bcast_cols, and
// m, l as column vectors), which faces 0 and 2 hold: those two faces only (bitwise outputs; R = 1 -0.35 ms, R = 8
// -0.43 ms at ctx 1K, bench/resident/m2b/f5). A root's output rows past its real rows (heads x R) are never sent, so a
// root whose real rows fit in face 0 (<= 16: sliding layers, full layers at R <= 2) runs face 0 only; mergers keep
// faces 0 and 2, so no stale value reaches a parent's real rows.
FORCE_INLINE VectorMode merge_vm(bool root, uint32_t Dt) {
    const uint32_t rows = (Dt == kDtf ? 8u : 2u) * R;
    return (root && rows <= 16) ? VectorMode::None : VectorMode::C;
}
// the compute API's SFPU binaries with a vector mode
ALWI void v_sub(uint32_t a, uint32_t b, uint32_t o, VectorMode vm) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
                           (APPROX, ckernel::BinaryOp::SUB, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default), a, b,
                           o, vm)));
}
ALWI void v_add(uint32_t a, uint32_t b, uint32_t o, VectorMode vm) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
                           (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default), a, b,
                           o, vm)));
}
ALWI void v_mul(uint32_t a, uint32_t b, uint32_t o, VectorMode vm) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
                           (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE), a, b, o, vm)));
}

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb, uint32_t idx) {
    pack_reconfig_data_format(cb);
    pack_tile<true>(dst, cb, idx);
}

// ---------------------------------------------------------------- prep (leaders)

// cb_rinv = rsqrt(mean of the squares of rows over Dt tiles of cb_in from off + eps)
void rinv_rows(uint32_t cb_in, uint32_t off, uint32_t Dt, uint32_t inv_d) {
    cb_reserve_back(cb_sq, 16);
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_sq);
    mul_init(cb_in, cb_in);
    for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < kDst && d0 + d < Dt; d++) {
            mul_tiles(cb_in, cb_in, off + d0 + d, off + d0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < kDst && d0 + d < Dt; d++) {
            pack_tile<true>(d, cb_sq, d0 + d);
        }
        tile_regs_release();
    }
    cb_push_back(cb_sq, 16);
    cb_wait_front(cb_sq, 16);
    cb_reserve_back(cb_rinv, 1);
    reconfig_data_format(cb_one, cb_sq);  // REDUCE_ROW SUM: scaler -> SrcA, data -> SrcB
    pack_reconfig_data_format(cb_rinv);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, cb_rinv);
    tile_regs_acquire();
    for (uint32_t d = 0; d < Dt; d++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, d, 0, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_d);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rinv);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(cb_rinv, 1);
    cb_pop_front(cb_sq, 16);
}

// cb_out (16 pages, tiles [0, Dt)) = cb_in[off + d] * rinv (row broadcast) [* cb_w[w_off + d]]
void norm_apply(uint32_t cb_in, uint32_t off, uint32_t Dt, bool weighted, uint32_t w_off, uint32_t cb_o) {
    cb_wait_front(cb_rinv, 1);
    cb_reserve_back(cb_o, 16);
    for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
        const uint32_t W = Dt - d0 < kDst ? Dt - d0 : kDst;
        tile_regs_acquire();
        reconfig_data_format(cb_in, cb_rinv);
        mul_bcast_cols_init(cb_in, cb_rinv);
        for (uint32_t d = 0; d < W; d++) {
            mul_tiles_bcast_cols(cb_in, cb_rinv, off + d0 + d, 0, d);
        }
        if (weighted) {
            reconfig_data_format(cb_w, cb_w);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w);
            for (uint32_t d = 0; d < W; d++) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w, w_off + d0 + d, d);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_o);
        for (uint32_t d = 0; d < W; d++) {
            pack_tile<true>(d, cb_o, d0 + d);
        }
        tile_regs_release();
    }
    cb_push_back(cb_o, 16);
    cb_pop_front(cb_rinv, 1);
}

// cb_o[base + j] = x[j] cos[j] - x[j + H] sin[j] (j < H), x[j] cos[j] + x[j - H] sin[j] (j >= H), H = Dt / 2;
// x = cb_x tiles [0, Dt), cos / sin = cb_cs tiles from c0 / s0 (rows p + i % R)
void rope(uint32_t cb_x, uint32_t Dt, uint32_t c0, uint32_t s0, uint32_t cb_o, uint32_t base) {
    const uint32_t H = Dt / 2;
    cb_wait_front(cb_x, 16);
    for (uint32_t j0 = 0; j0 < Dt; j0 += kDst / 2) {
        tile_regs_acquire();
        reconfig_data_format(cb_x, cb_cs);
        mul_init(cb_x, cb_cs);
        for (uint32_t b = 0; b < kDst / 2; b++) {
            const uint32_t j = j0 + b, partner = j < H ? j + H : j - H;
            mul_tiles(cb_x, cb_cs, j, c0 + j, 2 * b);
            mul_tiles(cb_x, cb_cs, partner, s0 + j, 2 * b + 1);
        }
        if (j0 < H) {
            sub_binary_tile_init();
            for (uint32_t b = 0; b < kDst / 2; b++) {
                sub_binary_tile(2 * b, 2 * b + 1, 2 * b);
            }
        } else {
            add_binary_tile_init();
            for (uint32_t b = 0; b < kDst / 2; b++) {
                add_binary_tile(2 * b, 2 * b + 1, 2 * b);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_o);
        for (uint32_t b = 0; b < kDst / 2; b++) {
            pack_tile<true>(2 * b, cb_o, base + j0 + b);
        }
        tile_regs_release();
    }
}

// cb_o (16 pages) = tiles [0, n) of cb_in in cb_o's format
void convert(uint32_t cb_in, uint32_t n, uint32_t cb_o) {
    cb_wait_front(cb_in, 16);
    cb_reserve_back(cb_o, 16);
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_o);
    copy_init(cb_in);
    for (uint32_t d0 = 0; d0 < n; d0 += kDst) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < kDst && d0 + d < n; d++) {
            copy_tile(cb_in, d0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < kDst && d0 + d < n; d++) {
            pack_tile<true>(d, cb_o, d0 + d);
        }
        tile_regs_release();
    }
    cb_push_back(cb_o, 16);
}

void prep(bool full) {
    const uint32_t Dt = full ? kDtf : kDts, QT = full ? QTf : QTs;
    const uint32_t c0 = full ? 2 * kDts : 0, s0 = full ? 2 * kDts + kDtf : kDts;
    const uint32_t inv_d = full ? inv_df_bits : inv_ds_bits;
    cb_wait_front(cb_q32, kQ32);
    cb_wait_front(cb_kv32, 16);
    cb_wait_front(cb_w, 32);
    // Q
    cb_reserve_back(cb_q, CQ);
    for (uint32_t qt = 0; qt < QT; qt++) {
        rinv_rows(cb_q32, qt * Dt, Dt, inv_d);
        norm_apply(cb_q32, qt * Dt, Dt, true, 0, cb_xn);
        rope(cb_xn, Dt, c0, s0, cb_q, qt * Dt);
        cb_pop_front(cb_xn, 16);
    }
    cb_push_back(cb_q, CQ);
    cb_reserve_back(cb_qready, 1);
    cb_push_back(cb_qready, 1);
    // K, V
    cb_reserve_back(cb_kr, 16);
    if (!full) {
        rinv_rows(cb_kv32, 0, Dt, inv_d);
        norm_apply(cb_kv32, 0, Dt, true, 16, cb_xn);
        rinv_rows(cb_kv32, Dt, Dt, inv_d);
        norm_apply(cb_kv32, Dt, Dt, false, 0, cb_vn);
    } else {
        rinv_rows(cb_kv32, 0, Dt, inv_d);
        norm_apply(cb_kv32, 0, Dt, false, 0, cb_vn);
        // k = v * w_k (rounded to bf16 as the reference's multiply)
        cb_wait_front(cb_vn, 16);
        cb_reserve_back(cb_xn, 16);
        for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
            tile_regs_acquire();
            reconfig_data_format(cb_vn, cb_w);
            mul_init(cb_vn, cb_w);
            for (uint32_t d = 0; d < kDst; d++) {
                mul_tiles(cb_vn, cb_w, d0 + d, 16 + d0 + d, d);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_xn);
            for (uint32_t d = 0; d < kDst; d++) {
                pack_tile<true>(d, cb_xn, d0 + d);
            }
            tile_regs_release();
        }
        cb_push_back(cb_xn, 16);
    }
    rope(cb_xn, Dt, c0, s0, cb_kr, 0);
    cb_push_back(cb_kr, 16);
    cb_pop_front(cb_xn, 16);
    convert(cb_kr, Dt, cb_k8);
    convert(cb_vn, Dt, cb_v8);
    cb_pop_front(cb_kr, 16);
    cb_pop_front(cb_vn, 16);
    cb_pop_front(cb_q32, kQ32);
    cb_pop_front(cb_kv32, 16);
    cb_pop_front(cb_w, 32);
}

// ---------------------------------------------------------------- attention over n position tiles

// K / V sources: the tail (all tiles in cb_tk / cb_tv at c * Dt + d, masked by cb_tmask[c]) or a worker's stream
// (cb_k: kKW tiles per position tile; cb_v: NV tiles per d; masks cb_wmask[t - lo] where needed)
struct Src {
    bool tail;
    uint32_t n, Dt, QT, NV;
    uint32_t t0, lo;  // worker: first position tile, the window's first tile
    const Plan* pl;
    bool sliding;
    bool kw;  // M1L: the writer streams this layer's K (wide full layers; else the reader, into cb_k2)
    uint32_t kcb() const { return tail ? cb_tk : (kw ? cb_k : cb_k2); }
    uint32_t vcb() const { return tail ? cb_tv : cb_v; }
    // mask tile (cb, index) of local position tile t, or cb = ~0
    void mask(uint32_t t, uint32_t& cb, uint32_t& idx) const {
        cb = ~0u;
        if (tail) {
            cb = cb_tmask;
            idx = t;
        } else if (sliding) {
            const uint32_t gt = t0 + t;
            if (gt < lo + 2 && needs_wmask(*pl, gt)) {
                cb = cb_wmask;
                idx = gt - lo;
            }
        }
    }
};

void attend(const Src& s, uint32_t out_cb = cb_own) {
    const uint32_t n = s.n, Dt = s.Dt, QT = s.QT;
    const uint32_t kcb = s.kcb(), vcb = s.vcb();
    // S = Q K^T (+ mask)
    cb_reserve_back(cb_s, CS);
    pack_reconfig_data_format(cb_s);
    for (uint32_t t = 0; t < n; t++) {
        if (!s.tail) {
            cb_wait_front(kcb, kKW);
        }
        const uint32_t kb = s.tail ? t * Dt : 0;
        reconfig_data_format(kcb, cb_q);
        matmul_init(cb_q, kcb, 1);
        tile_regs_acquire();
        for (uint32_t qt = 0; qt < QT; qt++) {
            for (uint32_t d = 0; d < Dt; d++) {
                matmul_tiles(cb_q, kcb, qt * Dt + d, kb + d, qt);
            }
        }
        uint32_t mcb, midx;
        s.mask(t, mcb, midx);
        if (mcb != ~0u) {
            reconfig_data_format(mcb, mcb);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(mcb);
            for (uint32_t qt = 0; qt < QT; qt++) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(mcb, midx, qt);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t qt = 0; qt < QT; qt++) {
            pack_tile<true>(qt, cb_s, qt * n + t);
        }
        tile_regs_release();
        if (!s.tail) {
            cb_pop_front(kcb, kKW);
        }
    }
    cb_push_back(cb_s, CS);
    cb_wait_front(cb_s, CS);
    // m = rowmax S
    cb_reserve_back(cb_m, kStat);
    reconfig_data_format(cb_s, cb_one);
    pack_reconfig_data_format(cb_m);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, cb_m);
    for (uint32_t qt = 0; qt < QT; qt++) {
        tile_regs_acquire();
        for (uint32_t t = 0; t < n; t++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, qt * n + t, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_m, qt);
        tile_regs_release();
    }
    reduce_uninit();
    cb_push_back(cb_m, kStat);
    cb_wait_front(cb_m, kStat);
    // P = exp(S - m)
    cb_reserve_back(cb_p, CS);
    for (uint32_t qt = 0; qt < QT; qt++) {
        for (uint32_t t0 = 0; t0 < n; t0 += kDst) {
            const uint32_t W = n - t0 < kDst ? n - t0 : kDst;
            tile_regs_acquire();
            reconfig_data_format(cb_s, cb_m);
            sub_bcast_cols_init(cb_s, cb_m);
            for (uint32_t i = 0; i < W; i++) {
                sub_tiles_bcast_cols(cb_s, cb_m, qt * n + t0 + i, qt, i);
            }
            exp_tile_init<false>();
            for (uint32_t i = 0; i < W; i++) {
                exp_tile<false>(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_p);
            for (uint32_t i = 0; i < W; i++) {
                pack_tile<true>(i, cb_p, qt * n + t0 + i);
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_p, CS);
    cb_pop_front(cb_s, CS);
    cb_wait_front(cb_p, CS);
    // l = rowsum P
    cb_reserve_back(cb_l, kStat);
    reconfig_data_format(cb_one, cb_p);
    pack_reconfig_data_format(cb_l);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, cb_l);
    for (uint32_t qt = 0; qt < QT; qt++) {
        tile_regs_acquire();
        for (uint32_t t = 0; t < n; t++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, qt * n + t, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_l, qt);
        tile_regs_release();
    }
    reduce_uninit();
    cb_push_back(cb_l, kStat);
    cb_wait_front(cb_l, kStat);
    // O = P V -> the partial (O tiles qt * Dt + d, then m, l)
    cb_reserve_back(out_cb, kOwn);
    pack_reconfig_data_format(out_cb);
    if (s.tail) {
        cb_wait_front(cb_tv, kTail);
    }
    reconfig_data_format(vcb, cb_p);
    matmul_init(cb_p, vcb, 0);
    for (uint32_t d = 0; d < Dt; d++) {
        if (!s.tail) {
            cb_wait_front(cb_v, s.NV);
        }
        tile_regs_acquire();
        for (uint32_t qt = 0; qt < QT; qt++) {
            for (uint32_t t = 0; t < n; t++) {
                matmul_tiles(cb_p, vcb, qt * n + t, s.tail ? t * Dt + d : t, qt);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t qt = 0; qt < QT; qt++) {
            pack_tile<true>(qt, out_cb, qt * Dt + d);
        }
        tile_regs_release();
        if (!s.tail) {
            cb_pop_front(cb_v, s.NV);
        }
    }
    cb_pop_front(cb_p, CS);
    reconfig_data_format_srca(cb_m);
    for (uint32_t qt = 0; qt < QT; qt++) {
        tile_regs_acquire();
        copy_init(cb_m);
        copy_tile(cb_m, qt, 0);
        copy_init(cb_l);
        copy_tile(cb_l, qt, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb, QT * Dt + qt);
        pack_tile<true>(1, out_cb, QT * Dt + QT + qt);
        tile_regs_release();
    }
    cb_pop_front(cb_m, kStat);
    cb_pop_front(cb_l, kStat);
    cb_push_back(out_cb, kOwn);
}

// ---------------------------------------------------------------- merge

// cb_pout = cb_own (PT tiles): a merger without children forwards its own partial
void forward_own(uint32_t PT) {
    cb_wait_front(cb_own, kOwn);
    cb_reserve_back(cb_pout, PTf);
    reconfig_data_format_srca(cb_own);
    pack_reconfig_data_format(cb_pout);
    copy_init(cb_own);
    for (uint32_t i0 = 0; i0 < PT; i0 += kDst) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < kDst && i0 + i < PT; i++) {
            copy_tile(cb_own, i0 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < kDst && i0 + i < PT; i++) {
            pack_tile<true>(i, cb_pout, i0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_pout, PTf);
    cb_pop_front(cb_own, kOwn);
}

// partial x (0: own in cb_own, 1 + i: slot i of cb_parts) tile idx -> (cb, index)
FORCE_INLINE uint32_t part_cb(uint32_t x) { return x == 0 ? cb_own : cb_parts; }
FORCE_INLINE uint32_t part_idx(uint32_t x, uint32_t PT, uint32_t idx) { return x == 0 ? idx : (x - 1) * PT + idx; }

// own + k children -> root: cb_out = O / L; merger: cb_pout = (O, M, L)
void merge(uint32_t k, bool root, uint32_t Dt, uint32_t QT, uint32_t PT, uint32_t parts_pages) {
    cb_wait_front(cb_own, kOwn);
    if (k > 0) {
        cb_wait_front(cb_parts, parts_pages);
    }
    const uint32_t out = root ? cb_out : cb_pout;
    cb_reserve_back(out, root ? kQ32 : PTf);
    const VectorMode vm = merge_vm(root, Dt);
    const uint32_t mi = QT * Dt, li = QT * Dt + QT;
    for (uint32_t qt = 0; qt < QT; qt++) {
        // M = max m_x
        cb_reserve_back(cb_m, kStat);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_own);
        copy_init(cb_own);
        copy_tile(cb_own, mi + qt, 0);
        for (uint32_t x = 1; x <= k; x++) {
            copy_init(cb_parts);
            copy_tile(cb_parts, part_idx(x, PT, mi + qt), 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0, vm);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_m, 0);
        tile_regs_release();
        cb_push_back(cb_m, kStat);
        cb_wait_front(cb_m, kStat);
        // s_x = exp(m_x - M)
        cb_reserve_back(cb_sc, 8);
        for (uint32_t x = 0; x <= k; x++) {
            tile_regs_acquire();
            copy_init(part_cb(x));
            copy_tile(part_cb(x), part_idx(x, PT, mi + qt), 0);
            copy_init(cb_m);
            copy_tile(cb_m, 0, 1);
            sub_binary_tile_init();
            v_sub(0, 1, 0, vm);
            exp_tile_init<false>();
            exp_tile<false>(0, vm);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_sc, x);
            tile_regs_release();
        }
        cb_push_back(cb_sc, 8);
        cb_wait_front(cb_sc, 8);
        // L = sum s_x l_x (root: 1 / L)
        cb_reserve_back(cb_l, kStat);
        tile_regs_acquire();
        for (uint32_t x = 0; x <= k; x++) {
            const uint32_t acc = x == 0 ? 0 : 1;
            copy_init(cb_sc);
            copy_tile(cb_sc, x, acc);
            copy_init(part_cb(x));
            copy_tile(part_cb(x), part_idx(x, PT, li + qt), 2);
            mul_binary_tile_init();
            v_mul(acc, 2, acc, vm);
            if (x > 0) {
                add_binary_tile_init();
                v_add(0, 1, 0, vm);
            }
        }
        if (root) {
            recip_tile_init();
            recip_tile(0, vm);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_l, 0);
        tile_regs_release();
        cb_push_back(cb_l, kStat);
        cb_wait_front(cb_l, kStat);
        // diag(w_x) = I * w_x (row broadcast on the FPU: exact at the leaders' HiFi4; the mergers' HiFi2 keeps 7
        // bits of w, as the matmul below does anyway), w_x = s_x [/ L at the root: fp32 SFPU product first]
        uint32_t wcb = cb_sc;
        if (root) {
            wcb = cb_sc2;
            cb_reserve_back(cb_sc2, 8);
            for (uint32_t x = 0; x <= k; x++) {
                tile_regs_acquire();
                copy_init(cb_sc);
                copy_tile(cb_sc, x, 0);
                copy_init(cb_l);
                copy_tile(cb_l, 0, 1);
                mul_binary_tile_init();
                v_mul(0, 1, 0, vm);
                tile_regs_commit();
                tile_regs_wait();
                pack_to(0, cb_sc2, x);
                tile_regs_release();
            }
            cb_push_back(cb_sc2, 8);
            cb_wait_front(cb_sc2, 8);
        }
        cb_reserve_back(cb_diag, 8);
        reconfig_data_format(cb_one, wcb);
        pack_reconfig_data_format(cb_diag);
        for (uint32_t x = 0; x <= k; x++) {
            tile_regs_acquire();
            mul_bcast_cols_init(cb_one, wcb);
            mul_tiles_bcast_cols(cb_one, wcb, 1, x, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_diag, x);
            tile_regs_release();
        }
        cb_push_back(cb_diag, 8);
        cb_wait_front(cb_diag, 8);
        if (root) {
            cb_pop_front(cb_sc2, 8);
        }
        // O = sum diag(w_x) O_x
        pack_reconfig_data_format(out);
        for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
            reconfig_data_format(cb_own, cb_diag);
            matmul_init(cb_diag, cb_own, 0);
            tile_regs_acquire();
            for (uint32_t d = 0; d < kDst; d++) {
                for (uint32_t x = 0; x <= k; x++) {
                    matmul_tiles(cb_diag, part_cb(x), x, part_idx(x, PT, qt * Dt + d0 + d), d);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < kDst; d++) {
                pack_tile<true>(d, out, qt * Dt + d0 + d);
            }
            tile_regs_release();
        }
        if (!root) {
            reconfig_data_format_srca(cb_m);
            copy_init(cb_m);
            tile_regs_acquire();
            copy_tile(cb_m, 0, 0);
            copy_init(cb_l);
            copy_tile(cb_l, 0, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out, mi + qt);
            pack_tile<true>(1, out, li + qt);
            tile_regs_release();
        }
        cb_pop_front(cb_m, kStat);
        cb_pop_front(cb_l, kStat);
        cb_pop_front(cb_sc, 8);
        cb_pop_front(cb_diag, 8);
    }
    cb_push_back(out, root ? kQ32 : PTf);
    cb_pop_front(cb_own, kOwn);
    if (k > 0) {
        cb_pop_front(cb_parts, parts_pages);
    }
}

// M1L: a full worker's running partial: c = merge(a, b) as (O, M, L) (merge() with k = 1, non-root, any CBs)
void merge2(uint32_t ca, uint32_t cbb, uint32_t cc, uint32_t Dt, uint32_t QT) {
    cb_wait_front(ca, kOwn);
    cb_wait_front(cbb, kOwn);
    cb_reserve_back(cc, kOwn);
    const VectorMode vm = merge_vm(false, Dt);
    const uint32_t mi = QT * Dt, li = QT * Dt + QT;
    const uint32_t src[2] = {ca, cbb};
    for (uint32_t qt = 0; qt < QT; qt++) {
        // M = max(m_a, m_b)
        cb_reserve_back(cb_m, kStat);
        tile_regs_acquire();
        reconfig_data_format_srca(ca);
        copy_init(ca);
        copy_tile(ca, mi + qt, 0);
        copy_init(cbb);
        copy_tile(cbb, mi + qt, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0, vm);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_m, 0);
        tile_regs_release();
        cb_push_back(cb_m, kStat);
        cb_wait_front(cb_m, kStat);
        // s_x = exp(m_x - M)
        cb_reserve_back(cb_sc, 8);
        for (uint32_t x = 0; x < 2; x++) {
            tile_regs_acquire();
            copy_init(src[x]);
            copy_tile(src[x], mi + qt, 0);
            copy_init(cb_m);
            copy_tile(cb_m, 0, 1);
            sub_binary_tile_init();
            v_sub(0, 1, 0, vm);
            exp_tile_init<false>();
            exp_tile<false>(0, vm);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_sc, x);
            tile_regs_release();
        }
        cb_push_back(cb_sc, 8);
        cb_wait_front(cb_sc, 8);
        // L = s_a l_a + s_b l_b
        cb_reserve_back(cb_l, kStat);
        tile_regs_acquire();
        for (uint32_t x = 0; x < 2; x++) {
            copy_init(cb_sc);
            copy_tile(cb_sc, x, x);
            copy_init(src[x]);
            copy_tile(src[x], li + qt, 2);
            mul_binary_tile_init();
            v_mul(x, 2, x, vm);
        }
        add_binary_tile_init();
        v_add(0, 1, 0, vm);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_l, 0);
        tile_regs_release();
        cb_push_back(cb_l, kStat);
        cb_wait_front(cb_l, kStat);
        // diag(s_x)
        cb_reserve_back(cb_diag, 8);
        reconfig_data_format(cb_one, cb_sc);
        pack_reconfig_data_format(cb_diag);
        for (uint32_t x = 0; x < 2; x++) {
            tile_regs_acquire();
            mul_bcast_cols_init(cb_one, cb_sc);
            mul_tiles_bcast_cols(cb_one, cb_sc, 1, x, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_diag, x);
            tile_regs_release();
        }
        cb_push_back(cb_diag, 8);
        cb_wait_front(cb_diag, 8);
        // O = diag(s_a) O_a + diag(s_b) O_b
        pack_reconfig_data_format(cc);
        for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
            reconfig_data_format(ca, cb_diag);
            matmul_init(cb_diag, ca, 0);
            tile_regs_acquire();
            for (uint32_t d = 0; d < kDst; d++) {
                for (uint32_t x = 0; x < 2; x++) {
                    matmul_tiles(cb_diag, src[x], x, qt * Dt + d0 + d, d);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < kDst; d++) {
                pack_tile<true>(d, cc, qt * Dt + d0 + d);
            }
            tile_regs_release();
        }
        reconfig_data_format_srca(cb_m);
        copy_init(cb_m);
        tile_regs_acquire();
        copy_tile(cb_m, 0, 0);
        copy_init(cb_l);
        copy_tile(cb_l, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cc, mi + qt);
        pack_tile<true>(1, cc, li + qt);
        tile_regs_release();
        cb_pop_front(cb_m, kStat);
        cb_pop_front(cb_l, kStat);
        cb_pop_front(cb_sc, 8);
        cb_pop_front(cb_diag, 8);
    }
    cb_push_back(cc, kOwn);
    cb_pop_front(ca, kOwn);
    cb_pop_front(cbb, kOwn);
}

}  // namespace

void kernel_main() {
    const uint32_t a = get_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(1);
    const uint32_t j = get_arg_val<uint32_t>(2);
    const uint32_t wf = get_arg_val<uint32_t>(3);
    const uint32_t parts_pages = get_arg_val<uint32_t>(4);
    compute_kernel_hw_startup(cb_q, cb_one, cb_s);
    cb_wait_front(cb_pos, 1);
    const Plan pl = make_plan(read_tile_value(cb_pos, 0, 0));
    cb_wait_front(cb_one, 2);
    cb_wait_front(cb_wmask, 2);
    if constexpr (kLeader) {
        cb_wait_front(cb_tmask, 2);
        cb_wait_front(cb_cs, kCS);
    }
    const Tree tree{5};
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        const uint32_t Dt = full ? kDtf : kDts, QT = full ? QTf : QTs, PT = full ? PTf : PTs;
        if (g >= kGroups && !(full && pl.wide)) {
            continue;  // M1L full-only cores: wide full layers only
        }
        if constexpr (kLeader) {
            if (full ? a != 0 : false) {
                continue;  // L_1..3: idle in full layers
            }
            prep(full);
            cb_wait_front(cb_tk, kTail);
            attend(Src{true, pl.ct, Dt, QT, 0, 0, 0, &pl, !full, false});
            cb_pop_front(cb_tk, kTail);
            cb_pop_front(cb_tv, kTail);
            const uint32_t k = full ? tree.active_children(pl) : pl.s_a;
            // the writer pushes the parts every root layer (k may be 0)
            cb_wait_front(cb_parts, parts_pages);
            merge(k, true, Dt, QT, PT, parts_pages);
            if (k == 0) {
                cb_pop_front(cb_parts, parts_pages);
            }
            cb_pop_front(cb_q, CQ);
        } else {
            cb_wait_front(cb_q, CQ);
            bool active;
            uint32_t t0 = 0, n = 0;
            if (full) {
                active = wf < pl.f_a;
                if (active) {
                    tree.split_full(pl, wf, t0, n);
                }
            } else {
                active = j - 1 < pl.s_a;
                if (active) {
                    split(pl.s_n, pl.s_a, j - 1, t0, n);
                    t0 += pl.s_lo;
                }
            }
            if (active) {
                const uint32_t NV = full ? NVf : NVs;
                // chunks of NV position tiles (sliding: one chunk); chunk j's partial merges into a running one
                const uint32_t nch = (n + NV - 1) / NV;
                for (uint32_t j = 0; j < nch; j++) {
                    const uint32_t cj = n - j * NV < NV ? n - j * NV : NV;
                    const uint32_t tgt = nch == 1 ? cb_own : (j == 0 ? cb_ra : cb_cp);
                    attend(Src{false, cj, Dt, QT, NV, t0 + j * NV, pl.s_lo, &pl, !full, full && pl.wide}, tgt);
                    const uint32_t pad = (CV - (Dt * NV) % CV) % CV;
                    if (pad) {
                        cb_wait_front(cb_v, pad);
                        cb_pop_front(cb_v, pad);
                    }
                    if (j > 0) {
                        const uint32_t prev = (j & 1) ? cb_ra : cb_rb;
                        const uint32_t next = j == nch - 1 ? cb_own : ((j & 1) ? cb_rb : cb_ra);
                        merge2(prev, cb_cp, next, Dt, QT);
                    }
                }
                if (full ? tree.merger() : j == 1) {
                    // a merger's partial always leaves through cb_pout (merged, or its own copied): a CB with two
                    // consumer RISCs breaks, as each keeps its own acked count and stores it to L1 on pop
                    const uint32_t k = full ? tree.active_children(pl) : 0;
                    if (k > 0) {
                        merge(k, false, Dt, QT, PT, parts_pages);
                    } else {
                        forward_own(PT);
                    }
                }
            }
            cb_pop_front(cb_q, CQ);
        }
    }
}
