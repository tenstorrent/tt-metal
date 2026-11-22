// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K2a unit-core compute (see k2a_common.hpp). The engine's attention math (attn_compute.cpp attend, merge2 and the root
// merge with F5v's vector mode, as CP0 ran it), one packed 32-row Q tile per unit (its user's rows plus other users' of
// the same kv head, computed and dropped; every op is per row). Per layer, after cb_q:
//   units of the layer's kind (records from cb_sched): per sub-chunk S = Q K^T (+ the mask tile from cb_mask where the
//     tile is a window-start or tail tile), m = rowmax, P = exp(S - m), l = rowsum, O = P V; sub-chunks fold into a
//     running partial (merge2); the unit's partial into cb_own;
//   roots: the k partials (cb_parts, in absolute chunk order) merged and finalized into cb_out (16 pages per root).
// K and V come as rings: cb_k Dt pages per position tile, cb_v NV pages per (sub-chunk, d); a push never straddles
// the ring's end (both sides pad to the push size when the layer kind changes).
// d_k2a copy (M6b): cb_sc / cb_diag hold kSC pages (drafter: a root merges up to kMaxKS = 9 partials).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "d_k2a_common.hpp"

using namespace k2a;

namespace {

constexpr uint32_t kDst = 8;
constexpr uint32_t kStat = 2;
// packed Q tiles hold several users' rows: the merges keep faces 0 and 2 (column 0 of all 32 rows)
constexpr VectorMode kVm = VectorMode::C;

// the compute API's SFPU binaries with a vector mode (CP0 / the engine's F5v merges)
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

uint32_t kpos = 0, vpos = 0;  // ring positions (pages consumed, mod the ring)

// K ring: pad a 16-page window to a 16-page boundary (the producer pads alike)
FORCE_INLINE void k_align(uint32_t Dt) {
    if (kpos % Dt != 0) {
        cb_wait_front(cb_k, kpos % Dt);
        cb_pop_front(cb_k, kpos % Dt);
        kpos = (kpos + kpos % Dt) % KR;
    }
}
FORCE_INLINE void k_done(uint32_t Dt) {
    cb_pop_front(cb_k, Dt);
    kpos = (kpos + Dt) % KR;
}
FORCE_INLINE void v_align(uint32_t NV) {
    const uint32_t pad = (NV - vpos % NV) % NV;
    if (pad) {
        cb_wait_front(cb_v, pad);
        cb_pop_front(cb_v, pad);
        vpos = (vpos + pad) % VR;
    }
}

// one sub-chunk of n position tiles (local tiles lo .. of a unit with nW / nT masked tiles of nu) against Q tiles
// [qb, qb + Dt): the partial (O tiles d, then m, l) into out_cb
void attend(uint32_t n, uint32_t qb, uint32_t Dt, uint32_t NV, uint32_t lo, uint32_t nW, uint32_t nT, uint32_t nu,
            uint32_t out_cb) {
    cb_reserve_back(cb_s, kCS);
    pack_reconfig_data_format(cb_s);
    for (uint32_t t = 0; t < n; t++) {
        k_align(Dt);
        cb_wait_front(cb_k, Dt);
        reconfig_data_format(cb_k, cb_q);
        matmul_init(cb_q, cb_k, 1);
        tile_regs_acquire();
        for (uint32_t d = 0; d < Dt; d++) {
            matmul_tiles(cb_q, cb_k, qb + d, d, 0);
        }
        const bool mask = masked(nW, nT, nu, lo + t);
        if (mask) {
            cb_wait_front(cb_mask, 1);
            reconfig_data_format(cb_mask, cb_mask);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_s, t);
        tile_regs_release();
        if (mask) {
            cb_pop_front(cb_mask, 1);
        }
        k_done(Dt);
    }
    cb_push_back(cb_s, kCS);
    cb_wait_front(cb_s, kCS);
    // m = rowmax S
    cb_reserve_back(cb_m, kStat);
    reconfig_data_format(cb_s, cb_one);
    pack_reconfig_data_format(cb_m);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, cb_m);
    tile_regs_acquire();
    for (uint32_t t = 0; t < n; t++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, t, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_m, 0);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(cb_m, kStat);
    cb_wait_front(cb_m, kStat);
    // P = exp(S - m)
    cb_reserve_back(cb_p, kCS);
    for (uint32_t t0 = 0; t0 < n; t0 += kDst) {
        const uint32_t W = n - t0 < kDst ? n - t0 : kDst;
        tile_regs_acquire();
        reconfig_data_format(cb_s, cb_m);
        sub_bcast_cols_init(cb_s, cb_m);
        for (uint32_t i = 0; i < W; i++) {
            sub_tiles_bcast_cols(cb_s, cb_m, t0 + i, 0, i);
        }
        exp_tile_init<false>();
        for (uint32_t i = 0; i < W; i++) {
            exp_tile<false>(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_p);
        for (uint32_t i = 0; i < W; i++) {
            pack_tile<true>(i, cb_p, t0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_p, kCS);
    cb_pop_front(cb_s, kCS);
    cb_wait_front(cb_p, kCS);
    // l = rowsum P
    cb_reserve_back(cb_l, kStat);
    reconfig_data_format(cb_one, cb_p);
    pack_reconfig_data_format(cb_l);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, cb_l);
    tile_regs_acquire();
    for (uint32_t t = 0; t < n; t++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, t, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_l, 0);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(cb_l, kStat);
    cb_wait_front(cb_l, kStat);
    // O = P V
    cb_reserve_back(out_cb, kPT);
    pack_reconfig_data_format(out_cb);
    reconfig_data_format(cb_v, cb_p);
    matmul_init(cb_p, cb_v, 0);
    for (uint32_t d = 0; d < Dt; d++) {
        v_align(NV);
        cb_wait_front(cb_v, NV);
        tile_regs_acquire();
        for (uint32_t t = 0; t < n; t++) {
            matmul_tiles(cb_p, cb_v, t, t, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb, d);
        tile_regs_release();
        cb_pop_front(cb_v, NV);
        vpos = (vpos + NV) % VR;
    }
    cb_pop_front(cb_p, kCS);
    reconfig_data_format_srca(cb_m);
    tile_regs_acquire();
    copy_init(cb_m);
    copy_tile(cb_m, 0, 0);
    copy_init(cb_l);
    copy_tile(cb_l, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, out_cb, Dt);
    pack_tile<true>(1, out_cb, Dt + 1);
    tile_regs_release();
    cb_pop_front(cb_m, kStat);
    cb_pop_front(cb_l, kStat);
    cb_push_back(out_cb, kPT);
}

// c = merge(a, b) as (O, M, L) (the engine's merge2, one Q tile)
void merge2(uint32_t ca, uint32_t cbb, uint32_t cc, uint32_t Dt) {
    cb_wait_front(ca, kPT);
    cb_wait_front(cbb, kPT);
    cb_reserve_back(cc, kPT);
    const uint32_t src[2] = {ca, cbb};
    cb_reserve_back(cb_m, kStat);
    tile_regs_acquire();
    reconfig_data_format_srca(ca);
    copy_init(ca);
    copy_tile(ca, Dt, 0);
    copy_init(cbb);
    copy_tile(cbb, Dt, 1);
    binary_max_tile_init();
    binary_max_tile(0, 1, 0, kVm);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, cb_m, 0);
    tile_regs_release();
    cb_push_back(cb_m, kStat);
    cb_wait_front(cb_m, kStat);
    cb_reserve_back(cb_sc, kSC);
    for (uint32_t x = 0; x < 2; x++) {
        tile_regs_acquire();
        copy_init(src[x]);
        copy_tile(src[x], Dt, 0);
        copy_init(cb_m);
        copy_tile(cb_m, 0, 1);
        sub_binary_tile_init();
        v_sub(0, 1, 0, kVm);
        exp_tile_init<false>();
        exp_tile<false>(0, kVm);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_sc, x);
        tile_regs_release();
    }
    cb_push_back(cb_sc, kSC);
    cb_wait_front(cb_sc, kSC);
    cb_reserve_back(cb_l, kStat);
    tile_regs_acquire();
    for (uint32_t x = 0; x < 2; x++) {
        copy_init(cb_sc);
        copy_tile(cb_sc, x, x);
        copy_init(src[x]);
        copy_tile(src[x], Dt + 1, 2);
        mul_binary_tile_init();
        v_mul(x, 2, x, kVm);
    }
    add_binary_tile_init();
    v_add(0, 1, 0, kVm);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, cb_l, 0);
    tile_regs_release();
    cb_push_back(cb_l, kStat);
    cb_wait_front(cb_l, kStat);
    cb_reserve_back(cb_diag, kSC);
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
    cb_push_back(cb_diag, kSC);
    cb_wait_front(cb_diag, kSC);
    pack_reconfig_data_format(cc);
    for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
        reconfig_data_format(ca, cb_diag);
        matmul_init(cb_diag, ca, 0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < kDst; d++) {
            for (uint32_t x = 0; x < 2; x++) {
                matmul_tiles(cb_diag, src[x], x, d0 + d, d);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < kDst; d++) {
            pack_tile<true>(d, cc, d0 + d);
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
    pack_tile<true>(0, cc, Dt);
    pack_tile<true>(1, cc, Dt + 1);
    tile_regs_release();
    cb_pop_front(cb_m, kStat);
    cb_pop_front(cb_l, kStat);
    cb_pop_front(cb_sc, kSC);
    cb_pop_front(cb_diag, kSC);
    cb_push_back(cc, kPT);
    cb_pop_front(ca, kPT);
    cb_pop_front(cbb, kPT);
}

// a root: its k partials at the front of cb_parts (slot x at x PT) -> cb_out = sum_x diag(exp(m_x - M) / L) O_x
void merge_root(uint32_t k, uint32_t Dt, uint32_t PT) {
    cb_wait_front(cb_parts, k * PT);
    cb_reserve_back(cb_out, 16);
    // M = max m_x
    cb_reserve_back(cb_m, kStat);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_parts);
    copy_init(cb_parts);
    copy_tile(cb_parts, Dt, 0);
    for (uint32_t x = 1; x < k; x++) {
        copy_init(cb_parts);
        copy_tile(cb_parts, x * PT + Dt, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0, kVm);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, cb_m, 0);
    tile_regs_release();
    cb_push_back(cb_m, kStat);
    cb_wait_front(cb_m, kStat);
    // s_x = exp(m_x - M)
    cb_reserve_back(cb_sc, kSC);
    for (uint32_t x = 0; x < k; x++) {
        tile_regs_acquire();
        copy_init(cb_parts);
        copy_tile(cb_parts, x * PT + Dt, 0);
        copy_init(cb_m);
        copy_tile(cb_m, 0, 1);
        sub_binary_tile_init();
        v_sub(0, 1, 0, kVm);
        exp_tile_init<false>();
        exp_tile<false>(0, kVm);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_sc, x);
        tile_regs_release();
    }
    cb_push_back(cb_sc, kSC);
    cb_wait_front(cb_sc, kSC);
    // 1 / L, L = sum s_x l_x
    cb_reserve_back(cb_l, kStat);
    tile_regs_acquire();
    for (uint32_t x = 0; x < k; x++) {
        const uint32_t acc = x == 0 ? 0 : 1;
        copy_init(cb_sc);
        copy_tile(cb_sc, x, acc);
        copy_init(cb_parts);
        copy_tile(cb_parts, x * PT + Dt + 1, 2);
        mul_binary_tile_init();
        v_mul(acc, 2, acc, kVm);
        if (x > 0) {
            add_binary_tile_init();
            v_add(0, 1, 0, kVm);
        }
    }
    recip_tile_init();
    recip_tile(0, kVm);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, cb_l, 0);
    tile_regs_release();
    cb_push_back(cb_l, kStat);
    cb_wait_front(cb_l, kStat);
    // diag(w_x), w_x = s_x / L (cb_sc / cb_diag hold kSC pages, k <= kMaxKS <= kSC)
    cb_reserve_back(cb_diag, kSC);
    for (uint32_t x = 0; x < k; x++) {
        tile_regs_acquire();
        copy_init(cb_sc);
        copy_tile(cb_sc, x, 0);
        copy_init(cb_l);
        copy_tile(cb_l, 0, 1);
        mul_binary_tile_init();
        v_mul(0, 1, 0, kVm);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_diag, x);
        tile_regs_release();
    }
    cb_push_back(cb_diag, kSC);
    cb_wait_front(cb_diag, kSC);
    cb_pop_front(cb_sc, kSC);
    cb_reserve_back(cb_sc, kSC);
    reconfig_data_format(cb_one, cb_diag);
    pack_reconfig_data_format(cb_sc);
    for (uint32_t x = 0; x < k; x++) {
        tile_regs_acquire();
        mul_bcast_cols_init(cb_one, cb_diag);
        mul_tiles_bcast_cols(cb_one, cb_diag, 1, x, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_sc, x);
        tile_regs_release();
    }
    cb_push_back(cb_sc, kSC);
    cb_wait_front(cb_sc, kSC);
    cb_pop_front(cb_diag, kSC);
    // O = sum diag(w_x) O_x
    pack_reconfig_data_format(cb_out);
    for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
        reconfig_data_format(cb_parts, cb_sc);
        matmul_init(cb_sc, cb_parts, 0);
        tile_regs_acquire();
        for (uint32_t d = 0; d < kDst; d++) {
            for (uint32_t x = 0; x < k; x++) {
                matmul_tiles(cb_sc, cb_parts, x, x * PT + d0 + d, d);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < kDst; d++) {
            pack_tile<true>(d, cb_out, d0 + d);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_m, kStat);
    cb_pop_front(cb_l, kStat);
    cb_pop_front(cb_sc, kSC);
    cb_push_back(cb_out, 16);
    cb_pop_front(cb_parts, k * PT);
}

FORCE_INLINE uint32_t rec(uint32_t i) { return read_tile_value(cb_sched, 0, i); }

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_q, cb_one, cb_s);
    cb_wait_front(cb_sched, 1);
    const uint32_t nu = rec(0), nr = rec(1), nus = rec(2), nrs = rec(4);
    cb_wait_front(cb_one, 2);
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        const uint32_t Dt = full ? kDtf : kDts, NV = full ? NVf : NVs, PT = full ? kPTf : kPTs;
        cb_wait_front(cb_q, CQ);
        for (uint32_t i = full ? nus : 0; i < (full ? nu : nus); i++) {
            const uint32_t w0 = rec(kUnitRec + kUW * i), w1 = rec(kUnitRec + kUW * i + 1);
            const uint32_t t0 = w1 & 0xffff, n = w1 >> 16;
            const uint32_t nW = (w0 >> 16) & 0xf, nT = (w0 >> 20) & 0xf, qb = (w0 >> 24) * Dt;
            const uint32_t nsub = subchunks(full, t0, n);
            for (uint32_t j = 0; j < nsub; j++) {
                uint32_t s0, sn;
                subchunk(full, t0, n, j, s0, sn);
                const uint32_t tgt = nsub == 1 ? cb_own : (j == 0 ? cb_ra : cb_cp);
                attend(sn, qb, Dt, NV, s0 - t0, nW, nT, n, tgt);
                if (j > 0) {
                    const uint32_t prev = (j & 1) ? cb_ra : cb_rb;
                    const uint32_t next = j == nsub - 1 ? cb_own : ((j & 1) ? cb_rb : cb_ra);
                    merge2(prev, cb_cp, next, Dt);
                }
            }
        }
        uint32_t used = 0;
        for (uint32_t r = full ? nrs : 0; r < (full ? nr : nrs); r++) {
            const uint32_t k = (rec(kRootRec + kRW * r) >> 16) & 0xff;
            merge_root(k, Dt, PT);
            used += k * PT;
        }
        // the writer pads cb_parts to NP every layer
        if (NP > used) {
            cb_wait_front(cb_parts, NP - used);
            cb_pop_front(cb_parts, NP - used);
        }
        cb_pop_front(cb_q, CQ);
    }
}
