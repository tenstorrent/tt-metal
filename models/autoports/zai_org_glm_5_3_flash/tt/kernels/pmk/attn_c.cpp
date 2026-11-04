// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK attention phase (kernels/pmk/attn_common.hpp) — compute of an attention core, per work item (h, qt0 .. qt0 + QG
// - 1): every K batch announced in CB_AINFO is applied to each of the item's QG running partials IN PLACE (region j of
// CB_APA = stats m / l of q tile j, region j of CB_APAO = its O; the TRISC seeks both CB pointers to region j before
// the update, so the reads of a DST group precede the pack that overwrites them). The arithmetic is Stage 14's
// kernels/mla/att_c.cpp partial(): S_i = Q_j K_i^T * scale (+ the triangle on the diagonal chunk tile, + a membership
// mask tile where the plan says so), m' = max(m, rowmax S), a = exp(m - m'), E = exp(S - m'), l' = a l + rowsum E,
// O' = diag(a) O + E V with diag(a) = I * bcast_col(a) on the SFPU. A chunk tile k is skipped for q tile qt < k. After
// the item's last batch: o_j = diag(1 / l_j) O_j -> CB_AOUT (bf16), QG tile rows.
// Batches with from_rm take their K tiles from CB_AKRM (row-major 32 x 512 rows gathered by the BRISC) through
// tilize_block into CB_AKT; the others find them in CB_AK already.
// Rev 8 (README §10): a membership mask arrives as an OPERAND (CB_AMASK, bf16: column 0 of row r = the row's
// membership byte over the K tile's 8 pools as an integer, attn_common.hpp mask_col_row). It is expanded here: one
// tile matmul B_col @ E8 (CB_AID page 2: row 0 = 2^-s on pool s's four columns) -> T = byte x 2^-s per element,
// exact; then attn_mask_sfpu.hpp pmk_mask_apply adds the exact fp32 -1e9 where floor(T) is even (the pool is not a
// member) with the scale multiply fused. The diagonal chunk tile ALWAYS adds the causal triangle (dense and sparse:
// the sparse operand marks the open pool allowed). Masked keys tie and exp to 0: the output is rev 6's (README §9a).
// CB_AID is bf16 (triangle 0 / bf16(-1e9), identity, E8); copy_tile expands it to the fp32 DST exactly. The T matmul
// runs at the kernel's config fidelity (HiFi4 -- the byte needs its 8 significant bits; PMK_ATT_FID_QK only lowers
// the two bf16 Q K / P V products).
// Rev 9 (README §11; the rev 8 sparse unit is TRISC-bound: softmax 52 %, QK 21 %, PV 18 % of 11.1 ms busy):
//   PMK_ATT_EXP_APPROX: E = exp(S - m') and a = exp(m - m') on the SFPU's exp_21f path (15 SFP instructions per row,
//     bf16-accurate; the accurate fp32 path is ~25) -- E is packed to bf16 for the P V product anyway;
//   PMK_ATT_MAX_LAZY (= the fp32 bits of a threshold T): the running max m is still tracked every batch, but the
//     rescale (a = exp(m - m'), diag(a), the 16-tile diag(a) O product, a l) runs only when some row's max grew by
//     more than T: otherwise the batch keeps m as its stabiliser (E = exp(S - m) <= e^T, exact in fp32 for T <= ~80;
//     the online softmax is exact for ANY stabiliser sequence) and O / l accumulate in place. The decision is one
//     word: the column reduce of (m' - m) packed to CB_AM page 1, read at (0, 0) through the unpack mailbox
//     (positive floats order as their bits). A row whose first batches are all masked (m ~ -1e9) still rescales
//     at its first real batch (the growth is ~1e9 > T: a = 0 annihilates the garbage, README §8c.7).
//   E = exp(S - m) for the batch's K tiles in pairs per DST acquire (DST 0 / 2, m in DST 1): the same values, half
//     the acquire / commit / pack handshakes and m loads.
// Compile-time args (PH_CT0 +): CB_AQ, CB_AK, CB_AKRM, CB_AMASK, CB_AINFO, CB_AS, CB_AM, CB_AP, CB_AE, CB_APA, CB_APB
//   (unused), CB_APAO, CB_APBO (unused), CB_ACOEF, CB_ADIAG, CB_AID, CB_AOUT, SCALE_BITS, CB_AKT.
// Runtime args: none (the plan stream drives the compute; KIND_END ends it).
// Defines: PMK_ATT_NK, PMK_QG, PMK_ATT_FID_QK (MathFidelity of the two bf16 matmuls), PMK_ATT_OB (O tiles per DST
//   acquire), PMK_PROBE (cycle counters: QK matmul, softmax SFPU, PV + rescale matmul, waits on the plan / K -> the
//   PROBE CB words 0..3, word 4 = 0xC0DE when done; the NCRISC flushes them with its marks).

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
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/attn_mask_sfpu.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/attn_common.hpp"

namespace {

constexpr uint32_t CB_AQ = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_AK = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_AKRM = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_AMASK = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t CB_AINFO = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t CB_AS = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t CB_AM = get_compile_time_arg_val(PH_CT0 + 6);
constexpr uint32_t CB_AP = get_compile_time_arg_val(PH_CT0 + 7);
constexpr uint32_t CB_AE = get_compile_time_arg_val(PH_CT0 + 8);
constexpr uint32_t CB_APA = get_compile_time_arg_val(PH_CT0 + 9);
constexpr uint32_t CB_APAO = get_compile_time_arg_val(PH_CT0 + 11);
constexpr uint32_t CB_ACOEF = get_compile_time_arg_val(PH_CT0 + 13);
constexpr uint32_t CB_ADIAG = get_compile_time_arg_val(PH_CT0 + 14);
constexpr uint32_t CB_AID = get_compile_time_arg_val(PH_CT0 + 15);
constexpr uint32_t CB_AOUT = get_compile_time_arg_val(PH_CT0 + 16);
constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(PH_CT0 + 17);
constexpr uint32_t CB_AKT = get_compile_time_arg_val(PH_CT0 + 18);
constexpr uint32_t DT = pmkatt::DT, NK = pmkatt::NK, QG = pmkatt::QG, ST = pmkatt::STAT_TILES;
#if defined(PMK_ATT_OB)
constexpr uint32_t OB = PMK_ATT_OB;
#else
constexpr uint32_t OB = 4;
#endif
#if defined(PMK_ATT_EXP_APPROX)
constexpr bool EXP_APPROX = true;
#else
constexpr bool EXP_APPROX = false;
#endif
#if defined(PMK_ATT_EXP_21F)
constexpr bool EXP_21F = true;  // rev 10: E on the exp_21f fp32 path (a keeps the accurate exp)
#else
constexpr bool EXP_21F = false;
#endif
#if defined(PMK_ATT_MAX_LAZY)
constexpr bool MAX_LAZY = true;
constexpr uint32_t LAZY_T_BITS = PMK_ATT_MAX_LAZY;  // fp32 bits of T (0x42000000 = 32.0f)
#else
constexpr bool MAX_LAZY = false;
constexpr uint32_t LAZY_T_BITS = 0;
#endif
// the two exps of the online softmax (rev 9: exp_21f when EXP_APPROX; both paths clamp inputs < -88.5 to 0, so the
// masked keys' -1e9 give exactly 0)
FORCE_INLINE void exp_init_() { exp_tile_init<EXP_APPROX>(); }
FORCE_INLINE void exp_(uint32_t d) { exp_tile<EXP_APPROX>(d); }
// the E exp (per K tile): EXP_21F -> the bf16-accurate fp32 SFPI path (attn_mask_sfpu.hpp), else exp_
FORCE_INLINE void exp_e_init_() {
    if constexpr (EXP_21F) {
        add_binary_tile_init();
    } else {
        exp_init_();
    }
}
FORCE_INLINE void exp_e_(uint32_t d) {
    if constexpr (EXP_21F) {
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, pmk_exp21f, d, d, d, VectorMode::RC)));
    } else {
        exp_(d);
    }
}
static_assert(OB >= 1 && OB <= 4 && DT % OB == 0, "O groups: at most 4 fp32 tiles per DST acquire (half sync)");
static_assert(NK * QG <= 31, "mask bits: one per (tile, q tile) in a 32-bit plan word");

#if defined(PMK_PROBE)
// cycle counters (the AICLK wall clock, low 32 bits): 0 = QK matmul (+ scale / mask), 1 = softmax SFPU (max, exp,
// sums, coefficients), 2 = PV + rescale matmul, 3 = waits (plan page, K batch, Q)
uint32_t g_cyc[4] = {0, 0, 0, 0};
FORCE_INLINE uint32_t now() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
#define PT_DECL() uint32_t _t0 = 0
#define PT_BEGIN() _t0 = now()
#define PT_END(k) g_cyc[k] += now() - _t0
#else
#define PT_DECL() \
    do {          \
    } while (0)
#define PT_BEGIN() \
    do {           \
    } while (0)
#define PT_END(k) \
    do {          \
    } while (0)
#endif

#if defined(PMK_ATT_FID_QK)
constexpr ckernel::MathFidelity QK_FID = static_cast<ckernel::MathFidelity>(PMK_ATT_FID_QK);
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

FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, i, d);
}
FORCE_INLINE void pack_to(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}
FORCE_INLINE void pad(uint32_t cb, uint32_t n) {
    if (n > 0) {
        cb_reserve_back(cb, n);
        cb_push_back(cb, n);
    }
}
FORCE_INLINE void emit(uint32_t d, uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_to(d, cb);
    tile_regs_release();
}
FORCE_INLINE void emit_n(uint32_t n, uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t j = 0; j < n; ++j) {
        pack_tile(j, cb);
    }
    tile_regs_release();
}
// diag(coefficient column k of CB_ACOEF) -> the next page of CB_ADIAG (caller reserved); CB_AID page 1 = the identity
FORCE_INLINE void diag_of(uint32_t k) {
    tile_regs_acquire();
    load(CB_AID, 1, 0);
    load(CB_ACOEF, k, 1);
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 1);
    emit(0, CB_ADIAG);
}

// one K batch (nk tiles in KCB, the plan's kind / k0 / masks) applied to q tile j (Q page j of CB_AQ) whose running
// partial lives in region j (`run`: it exists). For chunk batches tiles with k > qt are skipped (nk_j < nk); a batch
// with no applicable tile leaves the partial untouched (the caller skips the call). The mask operand of (tile i, q
// tile j) is CB_AMASK page popcount(mbits below bit i QG + j) (the BRISC pushes them in (i, j) order).
// In-place protocol per region: the previous batch's pages are waited for and POPPED first (the data stays; the CB's
// free-page accounting would otherwise block the reserve once every region holds pages), the pointers are seeked to
// the region, the reserve lands on the same pages, and the reads use the seeked read pointer.
FORCE_INLINE void update(
    uint32_t KCB, uint32_t nk_j, uint32_t j, uint32_t qt, uint32_t kind, uint32_t k0, uint32_t mbits, bool run) {
    PT_DECL();
    if (run) {
        pmkatt::cb_seek(CB_APA, j * ST);
        pmkatt::cb_seek(CB_APAO, j * DT);
        cb_wait_front(CB_APA, ST);
        cb_wait_front(CB_APAO, DT);
        cb_pop_front(CB_APA, ST);
        cb_pop_front(CB_APAO, DT);
    }
    pmkatt::cb_seek(CB_APA, j * ST);
    pmkatt::cb_seek(CB_APAO, j * DT);
    cb_reserve_back(CB_APA, ST);
    cb_reserve_back(CB_APAO, DT);
    // S_i = Q_j K_i^T * scale (+ the triangle on the diagonal chunk tile, + the membership mask)
    for (uint32_t i = 0; i < nk_j; ++i) {
        PT_BEGIN();
        cb_reserve_back(CB_AS, 1);
        tile_regs_acquire();
        reconfig_data_format(KCB, CB_AQ);
        mm_init_qk(CB_AQ, KCB, 1);
        for (uint32_t d = 0; d < DT; ++d) {
            mm_tile_qk(CB_AQ, KCB, j * DT + d, i * DT + d, 0);
        }
        const uint32_t bit = i * QG + j;
        if ((mbits >> bit) & 1u) {
            // T = B_col @ E8 into DST 1 (zero at the acquire), then S = S scale + (-1e9 where not a member)
            reconfig_data_format(CB_AID, CB_AMASK);
            matmul_init(CB_AMASK, CB_AID, 0);
            matmul_tiles(CB_AMASK, CB_AID, __builtin_popcount(mbits & ((1u << bit) - 1u)), 2, 1);
            add_binary_tile_init();
            MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, pmk_mask_apply, 0, 1, 0, VectorMode::RC, SCALE_BITS)));
        } else {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, SCALE_BITS);
        }
        if (kind == pmkatt::KIND_CHUNK && k0 + i == qt) {
            load(CB_AID, 0, 1);  // the causal triangle on the diagonal (dense and sparse)
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        emit(0, CB_AS);
        cb_push_back(CB_AS, 1);
        PT_END(0);
    }
    PT_BEGIN();
    pad(CB_AS, NK - nk_j);
    cb_wait_front(CB_AS, NK);
    // m' = max(m, rowmax S) -> CB_AM page 0 (page 1: MAX_LAZY's column-reduced growth m' - m); resc = this batch
    // rescales the partial (every running batch; MAX_LAZY: only when some row's max grew by more than T)
    cb_reserve_back(CB_AM, 2);
    tile_regs_acquire();
    load(CB_AS, 0, 0);
    for (uint32_t i = 1; i < nk_j; ++i) {
        load(CB_AS, i, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
    sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    bool resc = run;
    if (run) {
        load(CB_APA, 0, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0);
    }
    if (run && MAX_LAZY) {
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 2);  // DST 2 = m' - m >= 0 (column 0 carries the row values)
        sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_COL>(2);  // row 0 = the max over rows
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_AM);
        pack_tile(0, CB_AM);
        pack_tile(2, CB_AM);
        tile_regs_release();
        cb_push_back(CB_AM, 2);
        cb_wait_front(CB_AM, 2);
        resc = read_tile_value(CB_AM, 1, pmkatt::f32_idx(0, 0)) > LAZY_T_BITS;
    } else {
        emit(0, CB_AM);
        cb_push_back(CB_AM, 1);
        pad(CB_AM, 1);
        cb_wait_front(CB_AM, 2);
    }
    // the stabiliser of this batch: m' (CB_AM page 0), or the kept m (CB_APA page 0) when not rescaling
    const uint32_t mcb = (run && !resc) ? CB_APA : CB_AM;
    if (resc) {  // a = exp(m - m') -> CB_ACOEF page 0 (+ a pad page), diag(a) -> CB_ADIAG page 0 (+ pad)
        cb_reserve_back(CB_ACOEF, 2);
        tile_regs_acquire();
        load(CB_APA, 0, 0);
        load(CB_AM, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        exp_init_();
        exp_(0);
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
    // E_i = exp(S_i - m): bf16 for the PV matmul, fp32 for l; two K tiles per DST acquire (S in DST 0 / 2, m in 1)
    for (uint32_t i = 0; i < nk_j; i += 2) {
        const uint32_t n2 = nk_j - i < 2 ? nk_j - i : 2;
        cb_reserve_back(CB_AP, n2);
        cb_reserve_back(CB_AE, n2);
        tile_regs_acquire();
        load(mcb, 0, 1);
        for (uint32_t x = 0; x < n2; ++x) {
            load(CB_AS, i + x, 2 * x);
            sfpu_sub_bcast_col_init();
            sfpu_sub_bcast_col(2 * x, 1);
            exp_e_init_();
            exp_e_(2 * x);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t x = 0; x < n2; ++x) {
            pack_to(2 * x, CB_AP);
            pack_to(2 * x, CB_AE);
        }
        tile_regs_release();
        cb_push_back(CB_AP, n2);
        cb_push_back(CB_AE, n2);
    }
    pad(CB_AP, NK - nk_j);
    pad(CB_AE, NK - nk_j);
    cb_pop_front(CB_AS, NK);
    // stats in place: page 0 = the stabiliser (m', or the kept m), page 1 = l' = a l + rowsum E (l read before the
    // write)
    tile_regs_acquire();
    load(mcb, 0, 0);
    emit(0, CB_APA);
    cb_pop_front(CB_AM, 2);
    cb_wait_front(CB_AE, NK);
    tile_regs_acquire();
    load(CB_AE, 0, 0);
    for (uint32_t i = 1; i < nk_j; ++i) {
        load(CB_AE, i, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    if (resc) {  // l' = a l + rowsum E
        load(CB_APA, 1, 1);
        load(CB_ACOEF, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    } else if (run) {  // MAX_LAZY, no rescale: l' = l + rowsum E
        load(CB_APA, 1, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    emit(0, CB_APA);
    cb_push_back(CB_APA, ST);
    cb_pop_front(CB_AE, NK);
    PT_END(1);
    // O in place: O' = sum_i E_i V_i (+ diag(a) O), OB tiles per DST acquire; group g's O tiles are read (matmul
    // operands) before the group is packed over them
    PT_BEGIN();
    cb_wait_front(CB_AP, NK);
    for (uint32_t d0 = 0; d0 < DT; d0 += OB) {
        tile_regs_acquire();
        reconfig_data_format(KCB, CB_AP);
        mm_init_qk(CB_AP, KCB, 0);
        for (uint32_t g = 0; g < OB; ++g) {
            for (uint32_t i = 0; i < nk_j; ++i) {
                mm_tile_qk(CB_AP, KCB, i, i * DT + d0 + g, g);
            }
        }
        if (resc) {  // + diag(a) O
            reconfig_data_format(CB_APAO, CB_ADIAG);
            matmul_init(CB_ADIAG, CB_APAO, 0);
            for (uint32_t g = 0; g < OB; ++g) {
                matmul_tiles(CB_ADIAG, CB_APAO, 0, d0 + g, g);
            }
        } else if (run) {  // MAX_LAZY, no rescale: + I O (the identity, CB_AID page 1: O accumulates in place)
            reconfig_data_format(CB_APAO, CB_AID);
            matmul_init(CB_AID, CB_APAO, 0);
            for (uint32_t g = 0; g < OB; ++g) {
                matmul_tiles(CB_AID, CB_APAO, 1, d0 + g, g);
            }
        }
        emit_n(OB, CB_APAO);
        cb_push_back(CB_APAO, OB);
    }
    if (resc) {
        cb_pop_front(CB_ADIAG, 2);
        cb_pop_front(CB_ACOEF, 2);
    }
    cb_pop_front(CB_AP, NK);
    PT_END(2);
}

// o_j = diag(1 / l_j) O_j -> CB_AOUT (bf16), region j consumed
FORCE_INLINE void finalize(uint32_t j) {
    pmkatt::cb_seek(CB_APA, j * ST);
    pmkatt::cb_seek(CB_APAO, j * DT);
    cb_wait_front(CB_APA, ST);
    cb_wait_front(CB_APAO, DT);
    cb_reserve_back(CB_ACOEF, 2);
    tile_regs_acquire();
    load(CB_APA, 1, 0);
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
    reconfig_data_format(CB_APAO, CB_ADIAG);
    matmul_init(CB_ADIAG, CB_APAO, 0);
    for (uint32_t d0 = 0; d0 < DT; d0 += OB) {
        tile_regs_acquire();
        for (uint32_t g = 0; g < OB; ++g) {
            matmul_tiles(CB_ADIAG, CB_APAO, 0, d0 + g, g);
        }
        emit_n(OB, CB_AOUT);
        cb_push_back(CB_AOUT, OB);
    }
    cb_pop_front(CB_ADIAG, 2);
    cb_pop_front(CB_APA, ST);
    cb_pop_front(CB_APAO, DT);
}

}  // namespace

void kernel_main() {
    PT_DECL();
    compute_kernel_hw_startup(CB_AS, CB_AS);
    cb_wait_front(CB_AID, 3);  // the triangle (page 0), the identity (page 1), E8 (page 2), kept for the phase
    bool have[QG];
    for (uint32_t j = 0; j < QG; ++j) {
        have[j] = false;
    }
    bool in_item = false;
    for (;;) {
        PT_BEGIN();
        cb_wait_front(CB_AINFO, 1);
        const uint32_t w0 = read_tile_value(CB_AINFO, 0, 0);
        const uint32_t w1 = read_tile_value(CB_AINFO, 0, 1);
        const uint32_t w2 = read_tile_value(CB_AINFO, 0, 2);
        cb_pop_front(CB_AINFO, 1);
        if (w0 == pmkatt::KIND_END) {
            PT_END(3);
            break;
        }
        const uint32_t nk = w0 & 0xFFu, kind = (w0 >> 8) & 0xFu;
        const bool from_rm = ((w0 >> 12) & 1u) != 0;
        const uint32_t k0 = w1 & 0xFFFFu;
        const uint32_t qg = (w1 >> 16) & 0xFFu;
        const uint32_t qt0 = (w1 >> 24) & 0xFFu;  // the item's first q tile (attn_dm0.cpp: w1 bits 24..31)
        const bool last = (w2 & 1u) != 0;
        const uint32_t mbits = w2 >> 1;
        if (!in_item) {  // a new item: its Q tiles are in (the BRISC pushes them before the first batch)
            cb_wait_front(CB_AQ, QG * DT);
            in_item = true;
        }
        const uint32_t KCB = from_rm ? CB_AKT : CB_AK;
        if (from_rm) {
            cb_wait_front(CB_AKRM, NK * DT);
            cb_reserve_back(CB_AKT, NK * DT);
            // tilize_init's own format setup (state_configure) is a no-op in this build (it needs the compute-kernel
            // sentinel define): without these the tilize unpacked with the previous step's srcA format (fp32 APAO / AS)
            // and packed fp32 into the bf16 AKT pages (16, 2026-10-22 review: the sparse context K / V scrambled, PCC
            // 0.5)
            reconfig_data_format_srca(CB_AKRM);
            pack_reconfig_data_format(CB_AKT);
            tilize_init(CB_AKRM, DT, CB_AKT);
            // K tile i = the i-th 32-row band of the row-major block -> AKT tiles [i DT, (i + 1) DT): both indices (16,
            // 2026-10-22: with the defaults every call re-read band 0 and rewrote tiles 0 .. DT - 1, so K tile 1 of a
            // batch was stale L1 -- the sparse context's wrong rows moved with QG / the store placement; dense never
            // takes it)
            for (uint32_t i = 0; i < nk; ++i) {
                tilize_block(CB_AKRM, DT, CB_AKT, i * DT, i * DT);
            }
            tilize_uninit(CB_AKRM, CB_AKT);
            cb_push_back(CB_AKT, NK * DT);
            cb_pop_front(CB_AKRM, NK * DT);
        }
        cb_wait_front(KCB, NK * DT);
        // a masked batch carries EXACTLY NK x QG mask pages (the real ones first, in (i, j) order, then padding): a
        // variable count (3 on a QG 2 diagonal batch, 1 at QG 1) misaligned the 2 NK QG-page ring -> reads past its end
        const uint32_t nmask = mbits ? NK * QG : 0;
        if (nmask) {
            cb_wait_front(CB_AMASK, nmask);
        }
        PT_END(3);
        // q tile j's share of the batch: context tiles all apply; chunk tiles k0 .. k0 + nk - 1 apply while k <= qt
        // (the tiles are ascending, so the applicable ones are a prefix)
        for (uint32_t j = 0; j < qg; ++j) {
            const uint32_t qt = qt0 + j;
            uint32_t nk_j = nk;
            if (kind == pmkatt::KIND_CHUNK) {
                nk_j = 0;
                for (uint32_t i = 0; i < nk; ++i) {
                    if (k0 + i <= qt) {
                        nk_j = i + 1;
                    }
                }
            }
            if (nk_j > 0) {
                update(KCB, nk_j, j, qt, kind, k0, mbits, have[j]);
                have[j] = true;
            }
        }
        if (nmask) {
            cb_pop_front(CB_AMASK, nmask);
        }
        cb_pop_front(KCB, NK * DT);
        if (last) {
            for (uint32_t j = 0; j < qg; ++j) {
                if (have[j]) {
                    finalize(j);
                } else {  // a q tile with no key (cannot happen: chunk tile 0 always applies) -> zeros
                    cb_reserve_back(CB_AOUT, DT);
                    for (uint32_t d = 0; d < DT; ++d) {
                        tile_regs_acquire();
                        fill_tile_init();
                        fill_tile(0, 0.0f);
                        emit(0, CB_AOUT);
                    }
                    cb_push_back(CB_AOUT, DT);
                }
                have[j] = false;
            }
            cb_pop_front(CB_AQ, QG * DT);
            in_item = false;
        }
    }
    cb_pop_front(CB_AID, 3);
#if defined(PMK_PROBE)
    // the PACK thread's counters -> the probe CB's base (16-B words in the interface; no get_write_ptr on the compute
    // RISCs, and only the unpack / pack threads own cb_interface). The pack thread waits on the math thread per
    // tile group (tile_regs_wait), so its bucket profile is the pipeline's; every thread keeps its own g_cyc.
    PACK(({
        const LocalCBInterface& ci = get_local_cb_interface(PMK_PROBE);
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>((ci.fifo_limit - ci.fifo_size) << 4);
        for (uint32_t k = 0; k < 4; ++k) {
            p[k] = g_cyc[k];
        }
        p[4] = 0xC0DEu;
    }));
#endif
}
