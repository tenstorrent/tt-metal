// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel attention phase (kernels/mla/att_common.hpp) — compute of owner core (g, c), per token:
//   S_i = Q K_i^T (FPU, K tiles transposed in the unpacker, fp32 DST) * scale, the tile holding p masked (an additive
//   mask tile: columns > p % 32 get -1e30; SFPU ops take compile-time scalars only); m = rowmax_i S_i; E_i = exp(S_i -
//   m) (fp32 -> l = rowsum, bf16 -> the PV matmul); O = sum_i E_i V_i.
// A partial is two CBs: the stats (m, l; fp32, unpacked straight to DST: the SFPU math stays exact fp32) and O (16 fp32
// tiles, a matmul operand). Every rescale of O is an FPU matmul with a diagonal coefficient tile (diag(a) = I *
// bcast_col(a), one SFPU op per coefficient): combine (M = max(m_a, m_b), a = exp(m_a - M), b = exp(m_b - M), l = a
// l_a + b l_b; O = diag(a) O_a + diag(b) O_b, both matmuls accumulated in DST), the root's o = diag(1 / l) O (bf16).
// The last step before a core's send writes straight into the send CBs (no copy). Stats are pushed before O and O tile
// by tile, so the NCRISC streams a partial to the parent in groups (att_dm1.cpp) while it is being computed.
// With MLA_DSA (kernels/mla/dsa_common.hpp): sparse tokens (CB_AINFO word 2 = 0x5C) first score this core's pooled-key
// tiles (I4: relu(q_idx K_pool^T) -> wsc @ . -> CB_ASC, fp32), then fold the selected pools' K batches into a running
// partial (online softmax: m' = max(m, rowmax S), a = exp(m - m'), l' = a l + rowsum E, O' = diag(a) O + E V).
// Compile-time args: CB_AQ, CB_AK, CB_AMASK, CB_AINFO, CB_AS, CB_AM, CB_AP, CB_AE, CB_APA, CB_APB, CB_ARX, CB_ACOEF,
//   CB_ASEND, CB_AOUT, C, LV, SCALE_BITS, NEG_BITS (unused), CB_APAO, CB_APBO, CB_ARXO, CB_ASENDO, CB_ADIAG, CB_AID
//   [, MLA_DSA: CB_AQI, CB_AWS, CB_APK, CB_ASR, CB_ASC].
// Runtime args: c, T.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"

namespace {

constexpr uint32_t CB_AQ = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_AK = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_AMASK = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_AINFO = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t CB_AS = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t CB_AM = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t CB_AP = get_compile_time_arg_val(PH_CT0 + 6);
constexpr uint32_t CB_AE = get_compile_time_arg_val(PH_CT0 + 7);
constexpr uint32_t CB_APA = get_compile_time_arg_val(PH_CT0 + 8);
constexpr uint32_t CB_APB = get_compile_time_arg_val(PH_CT0 + 9);
constexpr uint32_t CB_ARX = get_compile_time_arg_val(PH_CT0 + 10);
constexpr uint32_t CB_ACOEF = get_compile_time_arg_val(PH_CT0 + 11);
constexpr uint32_t CB_ASEND = get_compile_time_arg_val(PH_CT0 + 12);
constexpr uint32_t CB_AOUT = get_compile_time_arg_val(PH_CT0 + 13);
constexpr uint32_t C = get_compile_time_arg_val(PH_CT0 + 14);
constexpr uint32_t LV = get_compile_time_arg_val(PH_CT0 + 15);
constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(PH_CT0 + 16);
constexpr uint32_t CB_APAO = get_compile_time_arg_val(PH_CT0 + 18);
constexpr uint32_t CB_APBO = get_compile_time_arg_val(PH_CT0 + 19);
constexpr uint32_t CB_ARXO = get_compile_time_arg_val(PH_CT0 + 20);
constexpr uint32_t CB_ASENDO = get_compile_time_arg_val(PH_CT0 + 21);
constexpr uint32_t CB_ADIAG = get_compile_time_arg_val(PH_CT0 + 22);
constexpr uint32_t CB_AID = get_compile_time_arg_val(PH_CT0 + 23);
#if defined(MLA_DSA)
constexpr uint32_t CB_AQI = get_compile_time_arg_val(PH_CT0 + 24);
constexpr uint32_t CB_AWS = get_compile_time_arg_val(PH_CT0 + 25);
constexpr uint32_t CB_APK = get_compile_time_arg_val(PH_CT0 + 26);
constexpr uint32_t CB_ASR = get_compile_time_arg_val(PH_CT0 + 27);
constexpr uint32_t CB_ASC = get_compile_time_arg_val(PH_CT0 + 28);
#endif
constexpr uint32_t DT = mlaatt::DT, NKMAX = mlaatt::NKMAX, ST = mlaatt::STAT_TILES;
// O tiles per DST acquire (fp32 DST, half sync: 4 tiles): one commit / pack / release per group; the groups match the
// NCRISC's streamed groups (att_dm1.cpp OG)
#if defined(MLA_ATT_OB)
constexpr uint32_t OB = MLA_ATT_OB;
#else
constexpr uint32_t OB = 4;
#endif
static_assert(OB >= 1 && OB <= 4 && DT % OB == 0, "O groups: at most 4 fp32 tiles per DST acquire (half sync)");

// MLA_ATT_FID_QK (rev 31, tt/mk_mla.py GLM53_MLA_ATT_FID_QK): the fidelity of the two bf16 x bf16 matmuls, S = Q K^T
// and O += E V; the diag(coefficient) rescales of the fp32 O partials keep the kernel's MATH_FIDELITY (tf32 operands: a
// lower fidelity there drops mantissa bits of O, the 2026-10-20 all-HiFi2 failure)
#if defined(MLA_ATT_FID_QK)
constexpr ckernel::MathFidelity QK_FID = static_cast<ckernel::MathFidelity>(MLA_ATT_FID_QK);
FORCE_INLINE void mm_init_qk(uint32_t in0, uint32_t in1, uint32_t transpose) {
    state_configure(in1, in0, __builtin_LINE());
    MATH((llk_math_matmul_init<QK_FID, MM_THROTTLE>(in0, in1, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose)));
}
FORCE_INLINE void mm_tile_qk(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1)));
    MATH((llk_math_matmul<QK_FID, MM_THROTTLE>(idst)));
}
#else
FORCE_INLINE void mm_init_qk(uint32_t in0, uint32_t in1, uint32_t transpose) { matmul_init(in0, in1, transpose); }
FORCE_INLINE void mm_tile_qk(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst) {
    matmul_tiles(in0, in1, i0, i1, idst);
}
#endif

// a partial = (stats CB: m, l) + (O CB: DT tiles)
struct Part {
    uint32_t s, o;
};
constexpr Part NONE{0xFFu, 0xFFu};  // no running partial

FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, i, d);
}
FORCE_INLINE void pack_to(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}
// n unwritten pages pushed: the CB's next user starts at its base again (att_common.hpp NKMAX)
FORCE_INLINE void pad(uint32_t cb, uint32_t n) {
    if (n > 0) {
        cb_reserve_back(cb, n);
        cb_push_back(cb, n);
    }
}
// one fp32 tile of DST slot d (already computed) into cb (caller reserved)
FORCE_INLINE void emit(uint32_t d, uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_to(d, cb);
    tile_regs_release();
}
// DST slots 0 .. n - 1 into cb, in order (caller reserved)
FORCE_INLINE void emit_n(uint32_t n, uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t j = 0; j < n; ++j) {
        pack_tile(j, cb);
    }
    tile_regs_release();
}
// diag(coefficient column k of CB_ACOEF) -> page k of CB_ADIAG (caller reserved the pair)
FORCE_INLINE void diag_of(uint32_t k) {
    tile_regs_acquire();
    load(CB_AID, 0, 0);
    load(CB_ACOEF, k, 1);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 1);
    emit(0, CB_ADIAG);
}

// this core's partial of one token (or of one batch folded into the running partial `prev`) into `dst`
FORCE_INLINE void partial(uint32_t nk, uint32_t mi, Part dst, Part prev) {
    cb_reserve_back(dst.s, ST);
    cb_reserve_back(dst.o, DT);
    if (nk == 0) {  // no rows here: m = -1e30, l = 0, O = 0 (the combine weighs it with exp(-1e30 - M) = 0)
        for (uint32_t i = 0; i < ST + DT; ++i) {
            tile_regs_acquire();
            fill_tile_init();
            if (i == 0) {
                fill_tile(0, -1.0e30f);
            } else {
                fill_tile(0, 0.0f);
            }
            emit(0, i < ST ? dst.s : dst.o);
        }
        cb_push_back(dst.s, ST);
        cb_push_back(dst.o, DT);
        return;
    }
    const bool run = prev.s != NONE.s;
    cb_wait_front(CB_AQ, DT);
    cb_wait_front(CB_AK, NKMAX * DT);  // nk row tiles + the BRISC's padding (att_common.hpp NKMAX)
    // S_i
    for (uint32_t i = 0; i < nk; ++i) {
        cb_reserve_back(CB_AS, 1);
        tile_regs_acquire();
        reconfig_data_format(CB_AK, CB_AQ);
        mm_init_qk(CB_AQ, CB_AK, 1);
        for (uint32_t d = 0; d < DT; ++d) {
            mm_tile_qk(CB_AQ, CB_AK, d, i * DT + d, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, SCALE_BITS);
        if (i == mi) {  // the additive mask of the last valid row (0, or -1e30 past it; fetched by the BRISC)
            cb_wait_front(CB_AMASK, 1);
            load(CB_AMASK, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            cb_pop_front(CB_AMASK, 1);
        }
        emit(0, CB_AS);
        cb_push_back(CB_AS, 1);
    }
    pad(CB_AS, NKMAX - nk);
    cb_wait_front(CB_AS, NKMAX);
    // m = rowmax (m' = max(m_prev, rowmax) for a running partial)
    cb_reserve_back(CB_AM, 1);
    tile_regs_acquire();
    load(CB_AS, 0, 0);
    for (uint32_t i = 1; i < nk; ++i) {
        load(CB_AS, i, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
    sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    if (run) {
        cb_wait_front(prev.s, ST);
        load(prev.s, 0, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0);
    }
    emit(0, CB_AM);
    cb_push_back(CB_AM, 1);
    cb_wait_front(CB_AM, 1);
    if (run) {  // a = exp(m_prev - m') <= 1 -> CB_ACOEF page 0 (+ a pad page: the pair stays at the CB base), diag(a)
        cb_reserve_back(CB_ACOEF, 2);
        tile_regs_acquire();
        load(prev.s, 0, 0);
        load(CB_AM, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        exp_tile_init<false>();
        exp_tile<false>(0);
        emit(0, CB_ACOEF);
        cb_push_back(CB_ACOEF, 1);
        pad(CB_ACOEF, 1);
        cb_wait_front(CB_ACOEF, 2);
        cb_reserve_back(CB_ADIAG, 2);
        diag_of(0);
        cb_push_back(CB_ADIAG, 1);
        pad(CB_ADIAG, 1);
        cb_wait_front(CB_ADIAG, 2);
    }
    // E_i = exp(S_i - m): bf16 for the PV matmul, fp32 for l
    for (uint32_t i = 0; i < nk; ++i) {
        cb_reserve_back(CB_AP, 1);
        cb_reserve_back(CB_AE, 1);
        tile_regs_acquire();
        load(CB_AS, i, 0);
        load(CB_AM, 0, 1);
        sfpu_sub_bcast_col_init();
        sfpu_sub_bcast_col(0, 1);
        exp_tile_init<false>();
        exp_tile<false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_AP);
        pack_to(0, CB_AE);
        tile_regs_release();
        cb_push_back(CB_AP, 1);
        cb_push_back(CB_AE, 1);
    }
    pad(CB_AP, NKMAX - nk);
    pad(CB_AE, NKMAX - nk);
    cb_pop_front(CB_AS, NKMAX);
    // stats: m, l = rowsum sum_i E_i (+ a l_prev)
    tile_regs_acquire();
    load(CB_AM, 0, 0);
    emit(0, dst.s);
    cb_pop_front(CB_AM, 1);
    cb_wait_front(CB_AE, NKMAX);
    tile_regs_acquire();
    load(CB_AE, 0, 0);
    for (uint32_t i = 1; i < nk; ++i) {
        load(CB_AE, i, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    if (run) {
        load(prev.s, 1, 1);
        load(CB_ACOEF, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    emit(0, dst.s);
    cb_push_back(dst.s, ST);  // the stats first: the parent's coefficients do not wait for O
    cb_pop_front(CB_AE, NKMAX);
    // O = sum_i E_i V_i (+ diag(a) O_prev), all accumulated in DST; each tile pushed as it is packed (the tree streams)
    cb_wait_front(CB_AP, NKMAX);
    if (run) {
        cb_wait_front(prev.o, DT);
    }
    for (uint32_t d0 = 0; d0 < DT; d0 += OB) {  // OB tiles per DST acquire
        tile_regs_acquire();
        reconfig_data_format(CB_AK, CB_AP);
        mm_init_qk(CB_AP, CB_AK, 0);
        for (uint32_t j = 0; j < OB; ++j) {
            for (uint32_t i = 0; i < nk; ++i) {
                mm_tile_qk(CB_AP, CB_AK, i, i * DT + d0 + j, j);
            }
        }
        if (run) {
            reconfig_data_format(prev.o, CB_ADIAG);
            matmul_init(CB_ADIAG, prev.o, 0);
            for (uint32_t j = 0; j < OB; ++j) {
                matmul_tiles(CB_ADIAG, prev.o, 0, d0 + j, j);
            }
        }
        emit_n(OB, dst.o);
        cb_push_back(dst.o, OB);
    }
    cb_pop_front(CB_AP, NKMAX);
    cb_pop_front(CB_AK, NKMAX * DT);
    cb_pop_front(CB_AQ, DT);
    if (run) {
        cb_pop_front(CB_ADIAG, 2);
        cb_pop_front(CB_ACOEF, 2);
        cb_pop_front(prev.s, ST);
        cb_pop_front(prev.o, DT);
    }
}

// (a = own partial, b = the child's in CB_ARX / CB_ARXO) -> out
FORCE_INLINE void combine(Part a, Part out) {
    cb_wait_front(a.s, ST);
    cb_wait_front(CB_ARX, ST);
    cb_reserve_back(out.s, ST);
    cb_reserve_back(out.o, DT);
    cb_reserve_back(CB_ACOEF, 2);
    tile_regs_acquire();
    load(a.s, 0, 0);
    load(CB_ARX, 0, 1);
    binary_max_tile_init();
    binary_max_tile(0, 1, 2);  // M
    sub_binary_tile_init();
    sub_binary_tile(0, 2, 0);
    sub_binary_tile(1, 2, 1);
    exp_tile_init<false>();
    exp_tile<false>(0);  // a
    exp_tile<false>(1);  // b
    tile_regs_commit();
    tile_regs_wait();
    pack_to(2, out.s);
    pack_to(0, CB_ACOEF);
    pack_to(1, CB_ACOEF);
    tile_regs_release();
    cb_push_back(CB_ACOEF, 2);
    cb_wait_front(CB_ACOEF, 2);
    // l = a l_a + b l_b
    tile_regs_acquire();
    load(a.s, 1, 0);
    load(CB_ACOEF, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    load(CB_ARX, 1, 2);
    load(CB_ACOEF, 1, 3);
    mul_binary_tile_init();
    mul_binary_tile(2, 3, 2);
    add_binary_tile_init();
    add_binary_tile(0, 2, 0);
    emit(0, out.s);
    cb_push_back(out.s, ST);
    cb_pop_front(a.s, ST);
    cb_pop_front(CB_ARX, ST);
    // O = diag(a) O_a + diag(b) O_b
    cb_reserve_back(CB_ADIAG, 2);
    diag_of(0);
    diag_of(1);
    cb_push_back(CB_ADIAG, 2);
    cb_pop_front(CB_ACOEF, 2);
    cb_wait_front(CB_ADIAG, 2);
    reconfig_data_format(a.o, CB_ADIAG);
    matmul_init(CB_ADIAG, a.o, 0);
    for (uint32_t d0 = 0; d0 < DT; d0 += OB) {  // streamed: a group of both inputs as it lands, the group out at once
        cb_wait_front(a.o, d0 + OB);
        cb_wait_front(CB_ARXO, d0 + OB);
        tile_regs_acquire();
        for (uint32_t j = 0; j < OB; ++j) {
            matmul_tiles(CB_ADIAG, a.o, 0, d0 + j, j);
            matmul_tiles(CB_ADIAG, CB_ARXO, 1, d0 + j, j);
        }
        emit_n(OB, out.o);
        cb_push_back(out.o, OB);
    }
    cb_pop_front(CB_ADIAG, 2);
    cb_pop_front(a.o, DT);
    cb_pop_front(CB_ARXO, DT);
}

// the group root: o = diag(1 / l) O -> CB_AOUT (bf16)
FORCE_INLINE void finalize(Part p) {
    cb_wait_front(p.s, ST);
    cb_reserve_back(CB_ACOEF, 2);
    tile_regs_acquire();
    load(p.s, 1, 0);
    recip_tile_init();
    recip_tile(0);
    emit(0, CB_ACOEF);
    cb_push_back(CB_ACOEF, 1);
    pad(CB_ACOEF, 1);
    cb_wait_front(CB_ACOEF, 2);
    cb_reserve_back(CB_ADIAG, 2);
    diag_of(0);
    cb_push_back(CB_ADIAG, 1);
    pad(CB_ADIAG, 1);
    cb_pop_front(CB_ACOEF, 2);
    cb_wait_front(CB_ADIAG, 2);
    cb_reserve_back(CB_AOUT, DT);
    reconfig_data_format(p.o, CB_ADIAG);
    matmul_init(CB_ADIAG, p.o, 0);
    for (uint32_t d0 = 0; d0 < DT; d0 += OB) {  // streamed like the combine
        cb_wait_front(p.o, d0 + OB);
        tile_regs_acquire();
        for (uint32_t j = 0; j < OB; ++j) {
            matmul_tiles(CB_ADIAG, p.o, 0, d0 + j, j);
        }
        emit_n(OB, CB_AOUT);
        cb_push_back(CB_AOUT, OB);
    }
    cb_pop_front(CB_ADIAG, 2);
    cb_pop_front(p.s, ST);
    cb_pop_front(p.o, DT);
}

#if defined(MLA_DSA)
// I4: the fp32 score tiles of this core's npt pooled-key tiles -> CB_ASC, one per tile
FORCE_INLINE void score(uint32_t npt) {
    cb_wait_front(CB_AQI, 4);
    cb_wait_front(CB_AWS, 1);
    // rev 37: SB pooled-key tiles per DST acquire (mladsa::SB; 1 = the rev <= 36 loop): S_j = relu(q_idx K_j^T) into
    // DST slot j, packed together; then wsc @ S_j into slot j, packed together. Rev 41: every batch takes SB tiles of
    // CB_APK / CB_ASR / CB_ASC (the last one padded: its nb real tiles first) -- a short batch moved the CB pointers
    // off the batch grid, and the next full batch straddled the CB end (watcher cb_push_back assert, card0 u34 05:21Z)
    for (uint32_t i0 = 0; i0 < npt; i0 += mladsa::SB) {
        const uint32_t nb = npt - i0 < mladsa::SB ? npt - i0 : mladsa::SB;
        cb_wait_front(CB_APK, 4 * mladsa::SB);
        cb_reserve_back(CB_ASR, mladsa::SB);
        tile_regs_acquire();
        reconfig_data_format(CB_APK, CB_AQI);
        matmul_init(CB_AQI, CB_APK, 1);
        for (uint32_t j = 0; j < nb; ++j) {
            for (uint32_t d = 0; d < 4; ++d) {
                matmul_tiles(CB_AQI, CB_APK, d, 4 * j + d, j);
            }
        }
        relu_tile_init();
        for (uint32_t j = 0; j < nb; ++j) {
            relu_tile(j);
        }
        emit_n(nb, CB_ASR);
        cb_push_back(CB_ASR, mladsa::SB);
        cb_pop_front(CB_APK, 4 * mladsa::SB);
        cb_wait_front(CB_ASR, mladsa::SB);
        cb_reserve_back(CB_ASC, mladsa::SB);
        tile_regs_acquire();
        reconfig_data_format(CB_ASR, CB_AWS);
        matmul_init(CB_AWS, CB_ASR, 0);
        for (uint32_t j = 0; j < nb; ++j) {
            matmul_tiles(CB_AWS, CB_ASR, 0, j, j);
        }
        emit_n(nb, CB_ASC);
        cb_push_back(CB_ASC, mladsa::SB);
        cb_pop_front(CB_ASR, mladsa::SB);
    }
    cb_pop_front(CB_AQI, 4);
    cb_pop_front(CB_AWS, 1);
}
#endif

}  // namespace

#if !defined(MLA_DSA_SHARD)
void kernel_main() {
    const uint32_t c = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t t_rows = get_arg_val<uint32_t>(ph_rt0 + 1);
#if defined(MLA_ATT_LANES)
    const uint32_t t0 = get_arg_val<uint32_t>(ph_rt0 + 2);  // the token lane (att_common.hpp LANES)
#else
    const uint32_t t0 = 0;
#endif
    compute_kernel_hw_startup(CB_AS, CB_AS);
    cb_wait_front(CB_AID, 1);  // the identity tile (the BRISC's first read), kept for the phase
    // this core's tree role: n_comb combines (receive levels before its send), then send or (the root) finalize
    uint32_t n_comb = 0;
    bool sends = false;
    for (uint32_t L = 0; L < LV && !sends; ++L) {
        if (mlaatt::recv_at(c, L, C)) {
            ++n_comb;
        } else if (mlaatt::send_at(c, L)) {
            sends = true;
        }
    }
    const Part pa{CB_APA, CB_APAO}, pb{CB_APB, CB_APBO}, send{CB_ASEND, CB_ASENDO};
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        cb_wait_front(CB_AINFO, 1);
        const uint32_t nk = read_tile_value(CB_AINFO, 0, 0);
        const uint32_t mi = read_tile_value(CB_AINFO, 0, 1);
        const uint32_t kind = read_tile_value(CB_AINFO, 0, 2);
        cb_pop_front(CB_AINFO, 1);
        // the token's partial: one batch (dense) or the selected pools' batches (DSA, folded into a running partial);
        // ONE partial() call site (two would make GCC outline the SFPU max-reduce init, whose SETC16 needs an
        // immediate: "impossible constraint in 'asm'"). The last batch goes where
        // the token's partial goes (send when the core sends at level 0, else A), never onto the running partial.
        const Part first = (sends && n_comb == 0) ? send : pa;
#if defined(MLA_ATT_LANES)
        bool batched = kind == mlaatt::KIND_BATCHED;  // the token lanes' dense batches (att_dm0.cpp writes word 2)
#else
        bool batched = false;  // (att_dm0.cpp leaves word 2 unwritten without lanes)
#endif
#if defined(MLA_DSA)
        if (kind == 0x5C) {
            score(nk);
            batched = true;
        }
#endif
        (void)kind;
        Part prev = NONE, out = pa;
        for (bool last = false; !last;) {
            uint32_t bk = nk, bm = mi;
            last = true;
            if (batched) {
                cb_wait_front(CB_AINFO, 1);
                bk = read_tile_value(CB_AINFO, 0, 0);
                bm = read_tile_value(CB_AINFO, 0, 1);
                last = read_tile_value(CB_AINFO, 0, 2) != 0;
                cb_pop_front(CB_AINFO, 1);
            }
            const Part dst = !last ? out : (first.s == prev.s ? pb : first);
            partial(bk, bm, dst, prev);
            prev = dst;
            out = (dst.s == pa.s) ? pb : pa;
        }
        Part cur = prev;
        Part nxt = (cur.s == pa.s) ? pb : pa;
        for (uint32_t k = 0; k < n_comb; ++k) {
            const Part to = (sends && k + 1 == n_comb) ? send : nxt;
            combine(cur, to);
            nxt = cur;
            cur = to;
        }
        if (!sends) {
            finalize(cur);
        }
    }
    cb_pop_front(CB_AID, 1);
}
#else  // MLA_DSA_SHARD (kernels/mla/dsa_shard.hpp): split-K over chips -- pre-sweep of the scores, then two passes
// (the peer's heads, this chip's heads) over the chip's own selected rows. The root of pass 0 lets its last combine
// land in the tree's SEND CBs (as a sending core does): the NCRISC ships that fp32 partial to the peer. The root of
// pass 1 receives the peer's partial of this chip's heads in the tree's RECEIVE CBs after its children (the NCRISC
// lands it there: one more tree level), combines it and finalises. (The first form packed bf16 hi + lo terms into CBs
// whose producer / consumer roles swapped between the passes; the TRISC's cb_wait_front then passed on the pass-0
// counts and read its own stale partial: the d0 card1 o_lat PCC 0.990 = the chip's half-context attention.)
// TP8 (MLA_DSA_SHARD_LINE = 4, kernels/mla/dsa_shard.hpp): the pass-1 root's combine with the column peer's partial
// (RS round 1) writes the column-combined partial P1 into the SEND CBs (the NCRISC ships its 8-row slices to the row,
// RS round 2); after every token's pass 1, per token: the own slice of P1 lands in the receive CBs (take: copied into a
// running partial -- stats exact via the fp32 unpack-to-DST, O through srcA as tf32, as every combine's matmul
// operand), then the 3 row peers' slices in column order (combine each), then finalize.
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_shard.hpp"
namespace {
constexpr uint32_t KIND_SCORE = 0x5C;
// the receive CBs (CB_ARX / CB_ARXO, the NCRISC's landed message) -> dst (a running partial)
FORCE_INLINE void take(Part dst) {
    cb_wait_front(CB_ARX, ST);
    cb_reserve_back(dst.s, ST);
    tile_regs_acquire();
    load(CB_ARX, 0, 0);
    load(CB_ARX, 1, 1);
    emit_n(ST, dst.s);
    cb_push_back(dst.s, ST);
    cb_pop_front(CB_ARX, ST);
    cb_reserve_back(dst.o, DT);
    for (uint32_t d0 = 0; d0 < DT; d0 += OB) {
        cb_wait_front(CB_ARXO, d0 + OB);
        tile_regs_acquire();
        for (uint32_t j = 0; j < OB; ++j) {
            load(CB_ARXO, d0 + j, j);
        }
        emit_n(OB, dst.o);
        cb_push_back(dst.o, OB);
    }
    cb_pop_front(CB_ARXO, DT);
}
}  // namespace

void kernel_main() {
    const uint32_t c = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t t_rows = get_arg_val<uint32_t>(ph_rt0 + 1);
#if defined(MLA_ATT_LANES)  // rev 60 (W): the token lane (att_common.hpp LANES; c / C / LV lane-local)
    const uint32_t t0 = get_arg_val<uint32_t>(ph_rt0 + 2);
#else
    const uint32_t t0 = 0;
#endif
    compute_kernel_hw_startup(CB_AS, CB_AS);
    cb_wait_front(CB_AID, 1);
    uint32_t n_comb = 0;
    bool sends = false;
    for (uint32_t L = 0; L < LV && !sends; ++L) {
        if (mlaatt::recv_at(c, L, C)) {
            ++n_comb;
        } else if (mlaatt::send_at(c, L)) {
            sends = true;
        }
    }
    const Part pa{CB_APA, CB_APAO}, pb{CB_APB, CB_APBO}, send{CB_ASEND, CB_ASENDO};
    // ---- pre-sweep: I4 of every sparse token (plan (npt, 1, 0x5C); dense tokens (0, 0, 0x5C): nothing to score)
    for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
        cb_wait_front(CB_AINFO, 1);
        const uint32_t npt = read_tile_value(CB_AINFO, 0, 0);
        const uint32_t is_sparse = read_tile_value(CB_AINFO, 0, 1);
        const uint32_t kind = read_tile_value(CB_AINFO, 0, 2);
        cb_pop_front(CB_AINFO, 1);
        (void)kind;
        if (is_sparse) {
            score(npt);
        }
    }
    // ---- THE DENSE WINDOW (MLA_DSA_DENSE_WINDOW, att_dsa_shard_dm0.cpp): the reader's mode header -- 1 = an all-dense
    // launch: ONE pass (this chip's heads over the replicated window), the root finalises its tree's combine (no peer
    // partial, no ship)
#if defined(MLA_DSA_DENSE_WINDOW)
    cb_wait_front(CB_AINFO, 1);
    const bool all_dense = read_tile_value(CB_AINFO, 0, 0) != 0;
    cb_pop_front(CB_AINFO, 1);
#else
    constexpr bool all_dense = false;
#endif
    // ---- the two passes: per token the pass header (., ., 0x5D) then its batches (the DSA batched form)
    for (uint32_t pass = all_dense ? 1 : 0; pass < 2; ++pass) {
        const bool ship = !sends && pass == 0;  // the root of pass 0: its final partial goes to the send CBs
        const bool to_send = sends || ship;
        for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
            cb_wait_front(CB_AINFO, 1);
            cb_pop_front(CB_AINFO, 1);  // the pass header: every token is batched
            const Part first = (to_send && n_comb == 0) ? send : pa;
            Part prev = NONE, out = pa;
            for (bool last = false; !last;) {
                cb_wait_front(CB_AINFO, 1);
                const uint32_t bk = read_tile_value(CB_AINFO, 0, 0);
                const uint32_t bm = read_tile_value(CB_AINFO, 0, 1);
                last = read_tile_value(CB_AINFO, 0, 2) != 0;
                cb_pop_front(CB_AINFO, 1);
                const Part dst = !last ? out : (first.s == prev.s ? pb : first);
                partial(bk, bm, dst, prev);
                prev = dst;
                out = (dst.s == pa.s) ? pb : pa;
            }
            Part cur = prev;
            Part nxt = (cur.s == pa.s) ? pb : pa;
            for (uint32_t k = 0; k < n_comb; ++k) {
                const Part to = (to_send && k + 1 == n_comb) ? send : nxt;
                combine(cur, to);
                nxt = cur;
                cur = to;
            }
            if (to_send) {
                continue;  // a sending core, or the root of pass 0 (the NCRISC ships CB_ASEND / CB_ASENDO)
            }
            if (all_dense) {
                finalize(cur);  // the window pass: the tree's combine is the whole attention of this chip's heads
                continue;
            }
            if constexpr (mlashard::LINE > 1) {
                combine(cur, send);  // TP8 RS round 1 -> P1 in the send CBs (the NCRISC ships its slices: round 2)
            } else {
                combine(
                    cur, nxt);  // the peer's partial of this chip's heads: the seventh tree level (CB_ARX / CB_ARXO)
                finalize(nxt);
            }
        }
    }
    if constexpr (mlashard::LINE > 1) {
        if (!sends &&
            !all_dense) {  // TP8 RS round 2 on the root: own slice, then the row peers' slices in column order
            for (uint32_t t = t0; t < t_rows; t += mlaatt::LANES) {
                take(pa);
                Part cur = pa, nxt = pb;
                for (uint32_t k = 1; k < mlashard::LINE; ++k) {
                    combine(cur, nxt);
                    const Part tmp = cur;
                    cur = nxt;
                    nxt = tmp;
                }
                finalize(cur);
            }
        }
    }
    cb_pop_front(CB_AID, 1);
}
#endif  // MLA_DSA_SHARD
