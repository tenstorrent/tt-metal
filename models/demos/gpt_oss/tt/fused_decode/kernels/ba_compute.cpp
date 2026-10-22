// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Block attention, compute (TRISC) (dflash/block_attention.py): one of kv head g's packed Q tiles (tile t = block rows
// 4 t .. 4 t + 3 x query heads 8 g .. 8 g + 7, every row real; one tile per core, the other tile group's core shares
// the K / V reads) over this slice's chunks. Online softmax over groups of G chunks, with the running state (m, l, O)
// in cb_m, cb_l, cb_o (FP32; O unnormalised; the merge / output loops over the core's Tt = 1 Q tiles index them
// cb_m + t ..; the pipelined group loop is written for Tt = 1):
//   S_j = Qs . K_j^T (+ causal mask of the block's chunks); on slice 0's first group also the sink "chunk" (column
//   0 = each head's sink logit: in the max and the sum, no V);
//   m' = max(m, rowmax S); a = exp(m - m'); P_j = exp(S_j - m') (BF16-accurate exp, P in BF16);
//   l' = a l + rowsum P; O' = a O + sum_j P_j . V_j.
// The children's partials (cb_recv: [m, l, O_0, O_1], tree rounds) merge with
//   m' = max(m, m_c); a = exp(m - m'), b = exp(m_c - m'); l' = a l + b l_c; O' = a O + b O_c.
// The root writes out = O / l (BF16); other slices send their state (cb_send).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_PACK  // the running-state SFPU ops run on the pack thread
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_binary_max_min.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace {
constexpr uint32_t cb_q = get_compile_time_arg_val(0);
constexpr uint32_t cb_k = get_compile_time_arg_val(1);
constexpr uint32_t cb_v = get_compile_time_arg_val(2);
constexpr uint32_t cb_aux = get_compile_time_arg_val(3);
constexpr uint32_t cb_qs = get_compile_time_arg_val(4);
constexpr uint32_t cb_s = get_compile_time_arg_val(5);
constexpr uint32_t cb_p = get_compile_time_arg_val(6);
constexpr uint32_t cb_m = get_compile_time_arg_val(7);     // cb_m + t: Q tile t's running max (FP32)
constexpr uint32_t cb_l = get_compile_time_arg_val(8);     // cb_l + t: running sum
constexpr uint32_t cb_o = get_compile_time_arg_val(9);     // cb_o + t: running output (2 tiles)
constexpr uint32_t cb_ab = get_compile_time_arg_val(10);   // merge factors a, b / 1 / l (FP32)
constexpr uint32_t cb_out = get_compile_time_arg_val(11);  // BF16 output tiles (root)
constexpr uint32_t cb_send = get_compile_time_arg_val(12);
constexpr uint32_t cb_recv = get_compile_time_arg_val(13);
constexpr uint32_t cb_lacc = get_compile_time_arg_val(14);  // the group's P tiles summed (FP32)
constexpr uint32_t scale_bits = get_compile_time_arg_val(15);
constexpr uint32_t M = get_compile_time_arg_val(16);
constexpr uint32_t G = get_compile_time_arg_val(17);
constexpr uint32_t Tt = 1;  // this core's packed Q tiles (of M / 4): one
static_assert(M / 4 == 2, "2 packed Q tiles per kv head, one per core (the 2 cores share the K / V reads)");
constexpr uint32_t NM = 2;
constexpr uint32_t mask_sink = NM * Tt + 1, tile_sink = NM * Tt + 2, tile_zero = NM * Tt + 3;  // cb_aux tiles

// formats for the next op: SrcA, SrcB operands and the pack output (compute_kernel_hw_startup runs once, at the start)
inline void cfg(uint32_t a, uint32_t b, uint32_t out) {
    reconfig_data_format(a, b);
    pack_reconfig_data_format(out);
}

// FP32-accurate exp of DST tiles (the merge factors: they compound over the groups)
inline void exp_accurate_init() { exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative, true>(); }
// (the state tiles hold their values in column 0: faces 0 and 2 only, VectorMode::C)
inline void exp_accurate(uint32_t i) {
    exp_tile<false, false, InputClamping::ClampToNegative, 8, true>(i, VectorMode::C);
}

// the running state's SFPU ops on the pack thread (the same calls as binary_max_tile / sub_binary_tile /
// add_binary_tile / exp_accurate, issued by PACK)
inline void pk_max_c(uint32_t a, uint32_t b, uint32_t o) {
    PACK((SFPU_BINARY_INIT_FN(max, sfpu::binary_max_min_init, (true))));
    PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binary_max_min, (true), a, b, o, VectorMode::C)));
}
inline void pk_sub(uint32_t a, uint32_t b, uint32_t o) {
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (false, ckernel::BinaryOp::SUB))));
    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (false, ckernel::BinaryOp::SUB, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::RC)));
}
inline void pk_add(uint32_t a, uint32_t b, uint32_t o) {
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (false, ckernel::BinaryOp::ADD))));
    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (false, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::RC)));
}
inline void pk_exp_accurate(uint32_t i) {
    exp_packthread_tile_init<false, 0x3F800000, InputClamping::ClampToNegative, true>();
    exp_packthread_tile<false, false, InputClamping::ClampToNegative, 8, true>(i, VectorMode::C);
}

// sub / add of the state tiles on the math thread over faces 0 and 2 only (their values sit in column 0)
inline void mt_sub_c(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (false, ckernel::BinaryOp::SUB, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::C)));
}
inline void mt_add_c(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (false, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::C)));
}

// Q tile t's state merged with a partial [m, l, O_0, O_1] at tiles i .. i + 3 of cb (the children's partials)
void merge(uint32_t t, uint32_t cb, uint32_t i) {
    const uint32_t cm = cb_m + t, cl = cb_l + t, co = cb_o + t;
    cb_wait_front(cm, 1);
    cb_wait_front(cl, 1);
    cb_wait_front(co, 2);
    cfg(cm, cb, cm);
    tile_regs_acquire();
    copy_init(cm);
    copy_tile(cm, 0, 0);
    copy_tile(cb, i, 1);
    binary_max_tile_init();
    binary_max_tile(0, 1, 2, VectorMode::C);
    sub_binary_tile_init();
    mt_sub_c(0, 2, 0);
    mt_sub_c(1, 2, 1);
    exp_accurate_init();
    exp_accurate(0);
    exp_accurate(1);
    tile_regs_commit();
    cb_pop_front(cm, 1);
    cb_reserve_back(cm, 1);
    cb_reserve_back(cb_ab, 2);
    tile_regs_wait();
    pack_tile(2, cm);
    pack_tile(0, cb_ab);
    pack_tile(1, cb_ab);
    tile_regs_release();
    cb_push_back(cm, 1);
    cb_push_back(cb_ab, 2);

    cb_wait_front(cb_ab, 2);
    cfg(cl, cb_ab, cl);
    tile_regs_acquire();
    mul_init(cl, cb_ab);
    mul_tiles(cl, cb_ab, 0, 0, 0);
    mul_tiles(cb, cb_ab, i + 1, 1, 1);
    add_binary_tile_init();
    mt_add_c(0, 1, 0);
    tile_regs_commit();
    cb_pop_front(cl, 1);
    cb_reserve_back(cl, 1);
    tile_regs_wait();
    pack_tile(0, cl);
    tile_regs_release();
    cb_push_back(cl, 1);

    tile_regs_acquire();
    mul_bcast_cols_init(co, cb_ab);
    mul_tiles_bcast_cols(co, cb_ab, 0, 0, 0);
    mul_tiles_bcast_cols(co, cb_ab, 1, 0, 1);
    mul_tiles_bcast_cols(cb, cb_ab, i + 2, 1, 2);
    mul_tiles_bcast_cols(cb, cb_ab, i + 3, 1, 3);
    add_binary_tile_init();
    add_binary_tile(0, 2, 0);
    add_binary_tile(1, 3, 1);
    tile_regs_commit();
    cb_pop_front(co, 2);
    cb_reserve_back(co, 2);
    tile_regs_wait();
    pack_tile(0, co);
    pack_tile(1, co);
    tile_regs_release();
    cb_push_back(co, 2);
    cb_pop_front(cb_ab, 2);
}
}  // namespace

void kernel_main() {
    uint32_t info = 0;
    UNPACK(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    const uint32_t nc = info & 0x1FFF;
    if (nc == 0) {
        return;
    }
#ifdef BA_UNIFORM
    const bool sink = false, win_first = (info >> 13) & 1;  // chunk 0 takes mask e = 1 (a window's cut start)
#else
    const bool sink = (info >> 13) & 1, win_first = false;
#endif
    const bool root = (info >> 14) & 1;
    const uint32_t rounds = (info >> 15) & 7;
    const int32_t off = static_cast<int32_t>(info >> 18) - 1;  // chunk j takes causal mask e = j - off (0 <= e <= 1)
    const uint32_t n_groups = (nc + G - 1) / G;

    cb_wait_front(cb_q, 2 * Tt);
    // Qs = Q * scale
    compute_kernel_hw_startup(cb_q, cb_q, cb_qs);
    cb_reserve_back(cb_qs, 2 * Tt);
    for (uint32_t t = 0; t < 2 * Tt; t += 2) {
        tile_regs_acquire();
        copy_init(cb_q);
        copy_tile(cb_q, t, 0);
        copy_tile(cb_q, t + 1, 1);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);
        mul_unary_tile(1, scale_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_qs);
        pack_tile(1, cb_qs);
        tile_regs_release();
    }
    cb_push_back(cb_qs, 2 * Tt);
    cb_pop_front(cb_q, 2 * Tt);
    cb_wait_front(cb_qs, 2 * Tt);
    cb_wait_front(cb_aux, NM * Tt + 4);

    // Pipelined groups: every SFPU op of the loop runs on the pack thread (P's exp, the running-state updates) and
    // the math thread issues only FPU ops, so it fills the other DST half while the pack thread runs the SFPU (one
    // SFPU, two instruction streams: the exp no longer serialises the FPU work). Order per group g: P(g)'s first DST
    // block, the scores of g + 1 (cb_s holds 2 groups), P(g)'s other blocks, the row max of g + 1, then l(g) and
    // O(g). The same ops compute every value as in a group-by-group loop (same results). The math thread may issue
    // SFPU ops only before the loop (Qs, the sink tile) and after it (merge, 1 / l).
    static_assert(Tt == 1, "the pipelined loop holds one Q tile's state (cb_s sized for one tile's 2 groups)");
    constexpr uint32_t t = 0;
    const uint32_t cm = cb_m, cl = cb_l, co = cb_o;
    auto qk_block = [&](uint32_t gi, uint32_t jb, uint32_t n) {
        const uint32_t j0 = gi * G;
        const bool extra = sink && gi == 0;  // the sink tile: group 0 only, computed before the loop (prelude)
        if (jb == 0) {
            cb_wait_front(cb_k, 2 * G);
            cb_reserve_back(cb_s, G + 1);
        }
        cfg(cb_k, cb_qs, cb_s);
        const uint32_t nb = jb + 4 < n ? 4 : n - jb;
        tile_regs_acquire();
        matmul_init(cb_qs, cb_k, 1);
        for (uint32_t b = 0; b < nb; ++b) {
            matmul_tiles(cb_qs, cb_k, 2 * t, 2 * (jb + b), b);
            matmul_tiles(cb_qs, cb_k, 2 * t + 1, 2 * (jb + b) + 1, b);
        }
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_aux);
        for (uint32_t b = 0; b < nb; ++b) {
            const int32_t e = static_cast<int32_t>(j0 + jb + b) - off;
            const uint32_t jg = j0 + jb + b;
            const uint32_t mk = (e == 0 || e == 1)       ? 1 + NM * t + e
                                : (win_first && jg == 0) ? 1 + NM * t + 1
                                                         : tile_zero;
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_aux, mk, b);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < nb; ++b) {
            pack_tile(b, cb_s);
        }
        tile_regs_release();
        if (jb + 4 >= n) {
            if (extra) {
                cfg(cb_aux, cb_aux, cb_s);
                tile_regs_acquire();
                copy_init(cb_aux);
                copy_tile(cb_aux, tile_sink, 0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, scale_bits);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_aux);
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_aux, mask_sink, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_s);
                tile_regs_release();
            }
            cb_push_back(cb_s, G + 1);
            cb_pop_front(cb_k, 2 * G);
        }
    };
    auto qk_all = [&](uint32_t gi, uint32_t n) {
        for (uint32_t jb = 0; jb < n; jb += 4) {
            qk_block(gi, jb, n);
        }
    };
    // m' = max(m, rowmax S) (math: the reduce + m's copy; pack: max, a = exp(m - m'))
    auto mx = [&](uint32_t ns, bool first) {
        cb_wait_front(cb_s, G + 1);
        cfg(cb_s, cb_aux, cm);
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_aux, cm);
        for (uint32_t j = 0; j < ns; ++j) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_aux, j, 0, 0);
        }
        reduce_uninit(cb_s);
        if (!first) {
            cb_wait_front(cm, 1);
            reconfig_data_format_srca(cm);
            copy_init(cm);
            copy_tile(cm, 0, 1);
        }
        tile_regs_commit();
        if (!first) {
            cb_pop_front(cm, 1);
            cb_reserve_back(cb_ab, 1);
        }
        cb_reserve_back(cm, 1);
        tile_regs_wait();
        if (!first) {
            pk_max_c(0, 1, 0);
            pk_sub(1, 0, 1);
            pk_exp_accurate(1);
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        }
        pack_tile(0, cm);
        if (!first) {
            pack_tile(1, cb_ab);
        }
        tile_regs_release();
        cb_push_back(cm, 1);
        if (!first) {
            cb_push_back(cb_ab, 1);
        }
    };
    // P = exp(S - m') block jb (math: the subtraction; pack: the exp, P and the L1-accumulated sum)
    auto p_block = [&](uint32_t jb, uint32_t ns) {
        if (jb == 0) {
            cb_wait_front(cm, 1);
            cb_reserve_back(cb_p, G + 1);
            cb_reserve_back(cb_lacc, 1);
        }
        cfg(cb_s, cm, cb_p);
        sub_bcast_cols_init(cb_s, cm);
        if (jb == 0) {
            exp_packthread_tile_init<false, 0x3F800000, InputClamping::ClampToNegative, false>();
        }
        const uint32_t nb = jb + 4 < ns ? 4 : ns - jb;
        tile_regs_acquire();
        for (uint32_t b = 0; b < nb; ++b) {
            sub_tiles_bcast_cols(cb_s, cm, jb + b, 0, b);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t b = 0; b < nb; ++b) {
            exp_packthread_tile<false, false, InputClamping::ClampToNegative, 8, false>(b);
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        // (L1 accumulation is off at each block's start and end and switched at most once in between: every switch
        // waits for the packer to drain)
        pack_reconfig_data_format(cb_p);
        for (uint32_t b = 0; b < nb; ++b) {
            pack_tile(b, cb_p);
        }
        pack_reconfig_data_format(cb_lacc);
        uint32_t b0 = 0;
        if (jb == 0) {
            pack_tile<true>(0, cb_lacc, 0);  // the group's first tile overwrites, the others accumulate
            b0 = 1;
        }
        if (b0 < nb) {
            pack_reconfig_l1_acc(1);
            for (uint32_t b = b0; b < nb; ++b) {
                pack_tile<true>(b, cb_lacc, 0);
            }
            pack_reconfig_l1_acc(0);
        }
        tile_regs_release();
        if (jb + 4 >= ns) {
            cb_push_back(cb_p, G + 1);
            cb_push_back(cb_lacc, 1);
            cb_pop_front(cb_s, G + 1);
        }
    };
    // l' = a l + rowsum P (math: the reduce, a l; pack: the add)
    auto l_upd = [&](bool first) {
        cb_wait_front(cb_lacc, 1);
        cfg(cb_aux, cb_lacc, cl);
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_lacc, cb_aux, cl);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_lacc, cb_aux, 0, 0, 0);
        reduce_uninit(cb_lacc);
        if (!first) {
            cb_wait_front(cl, 1);
            cb_wait_front(cb_ab, 1);
            reconfig_data_format(cl, cb_ab);
            mul_init(cl, cb_ab);
            mul_tiles(cl, cb_ab, 0, 0, 1);
        }
        tile_regs_commit();
        if (!first) {
            cb_pop_front(cl, 1);
        }
        cb_reserve_back(cl, 1);
        tile_regs_wait();
        if (!first) {
            pk_add(0, 1, 0);
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        }
        pack_tile(0, cl);
        tile_regs_release();
        cb_push_back(cl, 1);
        cb_pop_front(cb_lacc, 1);
    };
    // O' = a O + sum_j P_j . V_j (FPU only)
    auto pv = [&](uint32_t n, bool first) {
        cb_wait_front(cb_p, G + 1);
        cb_wait_front(cb_v, 2 * G);
        tile_regs_acquire();
        if (!first) {
            cb_wait_front(co, 2);
            cfg(co, cb_ab, co);
            mul_bcast_cols_init(co, cb_ab);
            mul_tiles_bcast_cols(co, cb_ab, 0, 0, 0);
            mul_tiles_bcast_cols(co, cb_ab, 1, 0, 1);
        }
        cfg(cb_v, cb_p, co);
        matmul_init(cb_p, cb_v, 0);
        for (uint32_t j = 0; j < n; ++j) {
            matmul_tiles(cb_p, cb_v, j, 2 * j, 0);
            matmul_tiles(cb_p, cb_v, j, 2 * j + 1, 1);
        }
        tile_regs_commit();
        if (!first) {
            cb_pop_front(co, 2);
            cb_pop_front(cb_ab, 1);
        }
        cb_reserve_back(co, 2);
        tile_regs_wait();
        pack_tile(0, co);
        pack_tile(1, co);
        tile_regs_release();
        cb_push_back(co, 2);
        cb_pop_front(cb_p, G + 1);
        cb_pop_front(cb_v, 2 * G);
    };
    auto group_n = [&](uint32_t gi) { return gi * G + G < nc ? G : nc - gi * G; };
    {
        const uint32_t n0 = group_n(0);
        qk_all(0, n0);
        mx(n0 + (sink ? 1 : 0), true);
    }
    for (uint32_t gi = 0; gi < n_groups; ++gi) {
        const uint32_t n = group_n(gi);
        const uint32_t ns = n + ((sink && gi == 0) ? 1 : 0);
        const bool nxt = gi + 1 < n_groups;
        p_block(0, ns);
        if (nxt) {
            qk_all(gi + 1, group_n(gi + 1));
        }
        for (uint32_t jb = 4; jb < ns; jb += 4) {
            p_block(jb, ns);
        }
        if (nxt) {
            mx(group_n(gi + 1), false);
        }
        l_upd(gi == 0);
        pv(n, gi == 0);
    }

    // the children's partials: [m, l, O_0, O_1] per Q tile (merge's and the root's SFPU ops run on the math thread:
    // the pack thread issues no SFPU op after the loop, and its last one precedes the release of the DST half that
    // merge's first acquire takes)
    for (uint32_t r = 0; r < rounds; ++r) {
        cb_wait_front(cb_recv, 4 * Tt);
        for (uint32_t t = 0; t < Tt; ++t) {
            merge(t, cb_recv, 4 * t);
        }
        cb_pop_front(cb_recv, 4 * Tt);
    }

    if (root) {
        // out = O / l
        cb_reserve_back(cb_out, 2 * Tt);
        for (uint32_t t = 0; t < Tt; ++t) {
            const uint32_t cm = cb_m + t, cl = cb_l + t, co = cb_o + t;
            cb_wait_front(cl, 1);
            cfg(cl, cl, cb_ab);
            tile_regs_acquire();
            copy_init(cl);
            copy_tile(cl, 0, 0);
            recip_tile_init();
            recip_tile(0);
            tile_regs_commit();
            cb_reserve_back(cb_ab, 1);
            tile_regs_wait();
            pack_tile(0, cb_ab);
            tile_regs_release();
            cb_push_back(cb_ab, 1);
            cb_wait_front(cb_ab, 1);
            cb_wait_front(co, 2);
            cfg(co, cb_ab, cb_out);
            tile_regs_acquire();
            mul_bcast_cols_init(co, cb_ab);
            mul_tiles_bcast_cols(co, cb_ab, 0, 0, 0);
            mul_tiles_bcast_cols(co, cb_ab, 1, 0, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            pack_tile(1, cb_out);
            tile_regs_release();
            cb_pop_front(cb_ab, 1);
            cb_pop_front(co, 2);
            cb_pop_front(cl, 1);
            cb_wait_front(cm, 1);
            cb_pop_front(cm, 1);
        }
        cb_push_back(cb_out, 2 * Tt);
    } else {
        // send the state: [m, l, O_0, O_1] per Q tile
        cb_reserve_back(cb_send, 4 * Tt);
        for (uint32_t t = 0; t < Tt; ++t) {
            const uint32_t cm = cb_m + t, cl = cb_l + t, co = cb_o + t;
            cb_wait_front(cm, 1);
            cb_wait_front(cl, 1);
            cb_wait_front(co, 2);
            cfg(cm, cm, cb_send);
            tile_regs_acquire();
            copy_init(cm);
            copy_tile(cm, 0, 0);
            copy_tile(cl, 0, 1);
            copy_tile(co, 0, 2);
            copy_tile(co, 1, 3);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_send);
            pack_tile(1, cb_send);
            pack_tile(2, cb_send);
            pack_tile(3, cb_send);
            tile_regs_release();
            cb_pop_front(cm, 1);
            cb_pop_front(cl, 1);
            cb_pop_front(co, 2);
        }
        cb_push_back(cb_send, 4 * Tt);
    }
}
