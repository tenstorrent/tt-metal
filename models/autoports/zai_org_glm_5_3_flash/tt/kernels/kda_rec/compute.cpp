// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// kda_rec compute: per item (b, h) one KDA decode step exactly as ttnn's fused_recurrent_kda compute (fp32 DST on the
// SFPU, every CB unpacked straight to DST; see that kernel for the derivation), the state packed once into CB_ST
// (T = 1: no next-token ping-pong), then the sigmoid-gated o RMSNorm of the op chain's KDA output:
//   on = o / sqrt(mean(o^2) + eps) * o_norm * sigmoid(gate_out)          (o, gate_out: the item's [1, V] rows, fp32)
// packed to bf16 (the o_proj input). DST in full-sync mode: 8 fp32 tiles.
// Compile-time args (PH_CT0 + i): Kt, Vt, L2NORM, SCALE_BITS, EPS_BITS (o_norm eps), INV_V_BITS (1 / V).
// Runtime args (ph_rt0 + i): items, core index (head = index / REC_SPREAD, value-column block = index % REC_SPREAD).

#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/kda_rec_common.hpp"
#if defined(REC_V2)
#include "api/compute/reconfig_data_format.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/kda_rec_v2_sfpu.hpp"
#endif

namespace {

using namespace ckernel;
using namespace kda_rec;

constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t Vt = get_compile_time_arg_val(PH_CT0 + 1);
constexpr bool L2NORM = get_compile_time_arg_val(PH_CT0 + 2) != 0;
constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t NORM_EPS_BITS = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t INV_V_BITS = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t KV = Kt * Vt;
constexpr uint32_t VTC = Vt / REC_SPREAD;  // this core's value-column tiles (REC_SPREAD cores per head)
constexpr uint32_t KVC = Kt * VTC;
static_assert(Vt % REC_SPREAD == 0, "REC_SPREAD divides the value tiles");
constexpr uint32_t EPS_BITS = 0x358637BD;  // 1e-6f (q / k l2norm)
static_assert(Kt >= 1 && Kt <= 4, "a state column block must fit DST slots 0..3");
static_assert(Vt >= 1 && Vt <= 4, "the o row must fit DST slots 0..3 (norm epilogue)");

// DST[out] = DST[data] * DST[col][r][0] (column broadcast, out of place).
ALWI void mul_bcast_col_to(uint32_t data, uint32_t col, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_sfpu_binary_bcast_full_tile_,
        (ckernel::BinaryOp::MUL, ckernel::BroadcastType::COL),
        data,
        col,
        out,
        VectorMode::None)));
}

// SFPU elementwise ops restricted to the faces that carry data: VectorMode::C (faces 0, 2) for column-vector tiles,
// VectorMode::R (faces 0, 1) for row-vector tiles.
template <ckernel::BinaryOp OP>
ALWI void binary_faces(uint32_t a, uint32_t b, uint32_t out, VectorMode mode) {
    if constexpr (OP == ckernel::BinaryOp::MUL) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_mul,
            (APPROX, OP, 8, DST_ACCUM_MODE),
            a,
            b,
            out,
            mode)));
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, OP, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
            a,
            b,
            out,
            mode)));
    }
}

template <int OP>
ALWI void scalar_faces(uint32_t idst, uint32_t bits, VectorMode mode) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, OP, 8, DST_ACCUM_MODE), idst, mode, bits));
}

ALWI void sqrt_faces(uint32_t idst, VectorMode mode) {
    MATH(
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt, (APPROX, 8, DST_ACCUM_MODE, false), idst, mode));
}

// Column tiles x[0..Kt) in DST[0..Kt) (a second copy in DST[4..4+Kt) is consumed): DST[0..Kt) <- x / sqrt(sum x^2 +
// eps), the 1 / norm going through DST[4] row 0, column 0.
ALWI void normalize_in_place() {
    mul_binary_tile_init();
    for (uint32_t j = 0; j < Kt; j++) {
        binary_faces<ckernel::BinaryOp::MUL>(4 + j, 4 + j, 4 + j, VectorMode::C);
    }
    add_binary_tile_init();
    for (uint32_t j = 1; j < Kt; j++) {
        binary_faces<ckernel::BinaryOp::ADD>(4, 4 + j, 4, VectorMode::C);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(4);
    binop_with_scalar_tile_init();
    scalar_faces<ADD_UNARY>(4, EPS_BITS, VectorMode::R);
    sqrt_tile_init();
    sqrt_faces(4, VectorMode::R);
    recip_tile_init();
    recip_tile(4, VectorMode::R);
    sfpu_mul_bcast_row_init();
    for (uint32_t j = 0; j < Kt; j++) {
        sfpu_mul_bcast_row(j, 4);
    }
}

// Column tiles of cb_x into DST[0..Kt) (and DST[4..4+Kt) when normalising).
ALWI void load_vector(uint32_t cb_x) {
    copy_init(cb_x);
    for (uint32_t j = 0; j < Kt; j++) {
        copy_tile(cb_x, j, j);
        if constexpr (L2NORM) {
            copy_tile(cb_x, j, 4 + j);
        }
    }
}

// qcol, kcol, kfull, dcol of one item.
ALWI void prep() {
    cb_reserve_back(CB_QCOL, Kt);
    tile_regs_acquire();
    load_vector(CB_Q);
    if constexpr (L2NORM) {
        normalize_in_place();
    }
    binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < Kt; j++) {
        scalar_faces<MUL_UNARY>(j, SCALE_BITS, VectorMode::C);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Kt; j++) {
        pack_tile(j, CB_QCOL);
    }
    tile_regs_release();
    cb_push_back(CB_QCOL, Kt);

    cb_reserve_back(CB_KCOL, Kt);
    cb_reserve_back(CB_KFULL, Kt);
    tile_regs_acquire();
    load_vector(CB_K);
    if constexpr (L2NORM) {
        normalize_in_place();
    }
    fill_tile_init();
    for (uint32_t j = 0; j < Kt; j++) {
        fill_tile(4 + j, 0.0f);
    }
    sfpu_add_bcast_col_init();
    for (uint32_t j = 0; j < Kt; j++) {
        sfpu_add_bcast_col(4 + j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Kt; j++) {
        pack_tile(j, CB_KCOL);
        pack_tile(4 + j, CB_KFULL);
    }
    tile_regs_release();
    cb_push_back(CB_KCOL, Kt);
    cb_push_back(CB_KFULL, Kt);

    cb_reserve_back(CB_DCOL, Kt);
    tile_regs_acquire();
    copy_init(CB_G);
    for (uint32_t j = 0; j < Kt; j++) {
        copy_tile(CB_G, j, j);
    }
    exp_tile_init<false>();
    for (uint32_t j = 0; j < Kt; j++) {
        exp_tile<false>(j, VectorMode::C);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Kt; j++) {
        pack_tile(j, CB_DCOL);
    }
    tile_regs_release();
    cb_push_back(CB_DCOL, Kt);
}

// DST[4] row 0 <- colsum over the block of DST[0..Kt) *col x (x given as Kt column tiles in cb_xcol).
ALWI void contract(uint32_t cb_xcol) {
    copy_init(cb_xcol);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        copy_tile(cb_xcol, mi, 4 + mi);
    }
    sfpu_mul_bcast_col_init();
    for (uint32_t mi = 0; mi < Kt; mi++) {
        mul_bcast_col_to(mi, 4 + mi, 4 + mi);
    }
    add_binary_tile_init();
    for (uint32_t mi = 1; mi < Kt; mi++) {
        add_binary_tile(4, 4 + mi, 4);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(4);
}

// One item, this core's VTC column blocks: CB_S0 -> CB_ST, o -> CB_O (state tiles mi * VTC + ni). DST: the block's
// state tiles in 0..Kt, operands / products in 4..7 (delta in 5 during the rank-1 update).
ALWI void step() {
    constexpr uint32_t D_DELTA = 5;
    constexpr uint32_t free_slots[3] = {4, 6, 7};
    cb_wait_front(CB_S0, KVC);
    cb_wait_front(CB_QCOL, Kt);
    cb_wait_front(CB_KCOL, Kt);
    cb_wait_front(CB_KFULL, Kt);
    cb_wait_front(CB_DCOL, Kt);
    cb_wait_front(CB_V, VTC);
    cb_wait_front(CB_BETA, 1);
    cb_reserve_back(CB_ST, KVC);
    cb_reserve_back(CB_O, VTC);
#if defined(REC_DEBUG)
    cb_reserve_back(CB_TAP, Vt);
#endif
    for (uint32_t ni = 0; ni < VTC; ni++) {
        tile_regs_acquire();
        copy_init(CB_S0);
        for (uint32_t mi = 0; mi < Kt; mi++) {
            copy_tile(CB_S0, mi * VTC + ni, mi);
            copy_tile(CB_DCOL, mi, 4 + mi);
        }
        sfpu_mul_bcast_col_init();
        for (uint32_t mi = 0; mi < Kt; mi++) {
            sfpu_mul_bcast_col(mi, 4 + mi);
        }
        contract(CB_KCOL);
        copy_init(CB_V);
        copy_tile(CB_V, ni, D_DELTA);
        copy_tile(CB_BETA, 0, 6);
        sub_binary_tile_init();
        binary_faces<ckernel::BinaryOp::SUB>(D_DELTA, 4, D_DELTA, VectorMode::R);
        mul_binary_tile_init();
        binary_faces<ckernel::BinaryOp::MUL>(D_DELTA, 6, D_DELTA, VectorMode::R);
        for (uint32_t m0 = 0; m0 < Kt; m0 += 3) {
            const uint32_t n = (Kt - m0 < 3) ? (Kt - m0) : 3;
            copy_init(CB_KFULL);
            for (uint32_t j = 0; j < n; j++) {
                copy_tile(CB_KFULL, m0 + j, free_slots[j]);
            }
            sfpu_mul_bcast_row_init();
            for (uint32_t j = 0; j < n; j++) {
                sfpu_mul_bcast_row(free_slots[j], D_DELTA);
            }
            add_binary_tile_init();
            for (uint32_t j = 0; j < n; j++) {
                add_binary_tile(m0 + j, free_slots[j], m0 + j);
            }
        }
        contract(CB_QCOL);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t mi = 0; mi < Kt; mi++) {
            pack_tile<true>(mi, CB_ST, mi * VTC + ni);
        }
        pack_tile(4, CB_O);
#if defined(REC_DEBUG)
        pack_tile(4, CB_TAP);  // debug tap: the pre-norm o row of block ni
#endif
        tile_regs_release();
    }
    cb_push_back(CB_ST, KVC);
    cb_push_back(CB_O, VTC);
#if defined(REC_DEBUG)
    cb_push_back(CB_TAP, Vt);
#endif
    cb_pop_front(CB_S0, KVC);
    cb_pop_front(CB_QCOL, Kt);
    cb_pop_front(CB_KCOL, Kt);
    cb_pop_front(CB_KFULL, Kt);
    cb_pop_front(CB_DCOL, Kt);
    cb_pop_front(CB_V, VTC);
    cb_pop_front(CB_BETA, 1);
}

// DST[4] row r, column 0 <- sum over this core's VTC o tiles (in DST[0..VTC) or CB_O) of o[r]^2; DST[4..4+VTC) used.
ALWI void partial_sumsq() {
    copy_init(CB_O);
    for (uint32_t j = 0; j < VTC; j++) {
        copy_tile(CB_O, j, 4 + j);
    }
    mul_binary_tile_init();
    for (uint32_t j = 0; j < VTC; j++) {
        binary_faces<ckernel::BinaryOp::MUL>(4 + j, 4 + j, 4 + j, VectorMode::R);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(4, VTC, 1);  // row sums -> DST[4] column 0
}

#if REC_SPREAD > 1
// The head's partial sums of squares (fp32 tiles, only [0][0] meaningful: row 0 is the item's o row) of the column
// blocks p != blk from CB_PIN (page = p < blk ? p : p - 1), this core's own in DST[4] (recomputed, bitwise the value it
// sent) -> DST[4] = the head's total, summed as a fixed pairwise tree ((b0 + b1) + (b2 + b3)): fp32 addition is
// commutative, so every core of the head computes the same bits whatever its own block.
ALWI void head_sumsq(uint32_t blk) {
    auto pin = [blk](uint32_t p) { return p < blk ? p : p - 1; };
    copy_init(CB_PIN);
    copy_tile(CB_PIN, pin(blk ^ 1), 5);  // the pair partner
    if constexpr (REC_SPREAD == 4) {
        const uint32_t o0 = (blk & 2) ^ 2;  // the other pair, in block order
        copy_tile(CB_PIN, pin(o0), 6);
        copy_tile(CB_PIN, pin(o0 + 1), 7);
    }
    add_binary_tile_init();
    add_binary_tile(4, 5, 4);
    if constexpr (REC_SPREAD == 4) {
        add_binary_tile(6, 7, 6);
        add_binary_tile(4, 6, 4);
    }
}
#endif

// CB_O (o row, VTC tiles) -> CB_ON (bf16): o / sqrt(mean(o^2) + eps) * w * sigmoid(gate_out), the mean over the whole
// head's V (REC_SPREAD > 1: phase A packs this core's partial sum of squares to CB_PSUM, the writer sends it to the
// head's other cores and assembles theirs in CB_PIN; phase B adds them). DST: o in 0..VTC, the squares / the weight /
// the gate in 4..4+VTC; 1 / rms at DST[4] column 0.
ALWI void gated_norm(uint32_t blk) {
    cb_wait_front(CB_O, VTC);
    cb_wait_front(CB_GO, VTC);
#if REC_SPREAD > 1
    cb_reserve_back(CB_PSUM, 1);
    tile_regs_acquire();
    partial_sumsq();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(4, CB_PSUM);
    tile_regs_release();
    cb_push_back(CB_PSUM, 1);
    cb_wait_front(CB_PIN, REC_SPREAD - 1);
#endif
    cb_reserve_back(CB_ON, VTC);
    tile_regs_acquire();
    copy_init(CB_O);
    for (uint32_t j = 0; j < VTC; j++) {
        copy_tile(CB_O, j, j);
    }
    partial_sumsq();
#if REC_SPREAD > 1
    head_sumsq(blk);
#endif
    binop_with_scalar_tile_init();
    scalar_faces<MUL_UNARY>(4, INV_V_BITS, VectorMode::R);
    scalar_faces<ADD_UNARY>(4, NORM_EPS_BITS, VectorMode::R);
    sqrt_tile_init();
    sqrt_faces(4, VectorMode::R);
    recip_tile_init();
    recip_tile(4, VectorMode::R);
    sfpu_mul_bcast_col_init();
    for (uint32_t j = 0; j < VTC; j++) {
        sfpu_mul_bcast_col(j, 4);  // row r of o_j scaled by DST[4][r][0]
    }
    copy_init(CB_W);
    for (uint32_t j = 0; j < VTC; j++) {
        copy_tile(CB_W, j, 4 + j);
    }
    mul_binary_tile_init();
    for (uint32_t j = 0; j < VTC; j++) {
        binary_faces<ckernel::BinaryOp::MUL>(j, 4 + j, j, VectorMode::R);
    }
    copy_init(CB_GO);
    for (uint32_t j = 0; j < VTC; j++) {
        copy_tile(CB_GO, j, 4 + j);
    }
    sigmoid_tile_init();
    for (uint32_t j = 0; j < VTC; j++) {
        sigmoid_tile<VectorMode::R>(4 + j);
    }
    mul_binary_tile_init();
    for (uint32_t j = 0; j < VTC; j++) {
        binary_faces<ckernel::BinaryOp::MUL>(j, 4 + j, j, VectorMode::R);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_ON);
    for (uint32_t j = 0; j < VTC; j++) {
        pack_tile(j, CB_ON);
    }
    pack_reconfig_data_format(CB_O);
    tile_regs_release();
    cb_push_back(CB_ON, VTC);
    cb_pop_front(CB_O, VTC);
    cb_pop_front(CB_GO, VTC);
#if REC_SPREAD > 1
    cb_pop_front(CB_PIN, REC_SPREAD - 1);
#endif
}

}  // namespace

#if defined(REC_V2)
// Stage 14c v2 (kda_rec_v2_sfpu.hpp): per item three DST windows -
//   W1: VEC -> prep (d, norms, qd / kd / kn, qk); per key tile mi: S[mi][0..3] -> slots 0..3, kd / qd band broadcasts,
//       pass 1 (contraction accumulators); mid (u, w, delta, o, delta patterns) -> pack VEC (CB_VEC2), OD (CB_DELTA);
//   W2..W5, per key tile mi: S[mi][0..3], VEC2, DELTA -> d / kn band broadcasts, pass 2 (S d + kn delta) -> CB_ST;
//   W6: DELTA (o), VEC2 (gate_out), W -> gated norm -> row 0 of slots 0..3 -> CB_ON (bf16).
// (DST does not persist across a commit / release, so VEC and OD go through L1 between the windows.)
namespace {
template <int A, int B>
ALWI void v2_bcast(uint32_t mi) {
    switch (mi) {
        case 0:
            MATH((
                SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_bcast, (0, A, B), 0, VectorMode::RC_custom)));
            break;
        case 1:
            MATH((
                SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_bcast, (1, A, B), 0, VectorMode::RC_custom)));
            break;
        case 2:
            MATH((
                SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_bcast, (2, A, B), 0, VectorMode::RC_custom)));
            break;
        default:
            MATH((
                SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_bcast, (3, A, B), 0, VectorMode::RC_custom)));
            break;
    }
}
}  // namespace

void kernel_main() {
    namespace v2 = ckernel::sfpu::kdav2;
    const uint32_t items = get_arg_val<uint32_t>(ph_rt0 + 0) & 0xFFFF;
    const uint32_t hc = get_arg_val<uint32_t>(ph_rt0 + 1);
    if (hc >= REC_HEADS || items == 0) {
        return;
    }
    static_assert(REC_SPREAD == 1 && Kt == 4 && Vt == 4, "kda_rec v2: one head per core, head dim 128");
    compute_kernel_hw_startup(CB_Q, CB_Q, CB_ST);
    cb_wait_front(CB_W, 1);
    for (uint32_t it = 0; it < items; it++) {
        // ---- W1: prep, pass 1, mid
        cb_wait_front(CB_Q, 1);
        cb_wait_front(CB_S0, KV);
        cb_reserve_back(CB_VEC2, 1);
        cb_reserve_back(CB_DELTA, 1);
        tile_regs_acquire();
        copy_init(CB_Q);
        copy_tile(CB_Q, 0, 4);
        MATH((SFPU_UNARY_INIT(unused)));
        MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_prep, (SCALE_BITS), 0, VectorMode::RC_custom)));
        for (uint32_t mi = 0; mi < Kt; mi++) {
            copy_init(CB_S0);
            for (uint32_t ni = 0; ni < Vt; ni++) {
                copy_tile(CB_S0, mi * Vt + ni, ni);
            }
            MATH((SFPU_UNARY_INIT(unused)));
            v2_bcast<v2::VK, v2::VQ>(mi);
            if (mi == 0) {
                MATH((SFPU_UNARY_CALL(
                    DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_pass1, (true), 0, VectorMode::RC_custom)));
            } else {
                MATH((SFPU_UNARY_CALL(
                    DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_pass1, (false), 0, VectorMode::RC_custom)));
            }
        }
        MATH((
            SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_mid, 0, VectorMode::RC_custom)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(4, CB_VEC2);
        pack_tile(7, CB_DELTA);
        tile_regs_release();
        cb_push_back(CB_VEC2, 1);
        cb_push_back(CB_DELTA, 1);
        cb_pop_front(CB_Q, 1);

        // ---- W2..W5: pass 2 per key tile
        cb_wait_front(CB_VEC2, 1);
        cb_wait_front(CB_DELTA, 1);
        cb_reserve_back(CB_ST, KV);
        for (uint32_t mi = 0; mi < Kt; mi++) {
            tile_regs_acquire();
            copy_init(CB_S0);
            for (uint32_t ni = 0; ni < Vt; ni++) {
                copy_tile(CB_S0, mi * Vt + ni, ni);
            }
            copy_init(CB_VEC2);
            copy_tile(CB_VEC2, 0, 4);
            copy_init(CB_DELTA);
            copy_tile(CB_DELTA, 0, 7);
            MATH((SFPU_UNARY_INIT(unused)));
            MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_masks, 0, VectorMode::RC_custom)));
            v2_bcast<v2::VG, v2::VKN>(mi);
            MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_pass2, 0, VectorMode::RC_custom)));
#if defined(REC_V2_PROBE)
            if (mi == 0) {
                MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_probe, 0, VectorMode::RC_custom)));
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t ni = 0; ni < Vt; ni++) {
                pack_tile<true>(ni, CB_ST, mi * Vt + ni);
            }
            tile_regs_release();
        }
        cb_push_back(CB_ST, KV);
        cb_pop_front(CB_S0, KV);

        // ---- W6: gated norm -> CB_ON
        cb_reserve_back(CB_ON, Vt);
        tile_regs_acquire();
        copy_init(CB_DELTA);
        copy_tile(CB_DELTA, 0, 7);
        copy_init(CB_VEC2);
        copy_tile(CB_VEC2, 0, 4);
        copy_init(CB_W);
        copy_tile(CB_W, 0, 5);
        MATH((SFPU_UNARY_INIT(unused)));
        MATH((SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, kda_rec_v2_norm, (NORM_EPS_BITS, INV_V_BITS), 0, VectorMode::RC_custom)));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_ON);
        for (uint32_t ni = 0; ni < Vt; ni++) {
            pack_tile(ni, CB_ON);
        }
        pack_reconfig_data_format(CB_ST);
        tile_regs_release();
        cb_push_back(CB_ON, Vt);
        cb_pop_front(CB_VEC2, 1);
        cb_pop_front(CB_DELTA, 1);
    }
    cb_pop_front(CB_W, 1);
}
#else
void kernel_main() {
    const uint32_t items = get_arg_val<uint32_t>(ph_rt0 + 0) & 0xFFFF;  // (b0 << 16) | n (Stage 14 user groups)
    const uint32_t hc = get_arg_val<uint32_t>(ph_rt0 + 1);  // core index: head hc / REC_SPREAD, block hc % REC_SPREAD
    if (hc / REC_SPREAD >= REC_HEADS) {
        return;  // a core past the shard's heads: no recurrence work
    }
    const uint32_t blk = hc % REC_SPREAD;
    if (items == 0) {
        return;  // a user group without rows (Stage 14 REC_UG, rows < groups)
    }
    compute_kernel_hw_startup(CB_Q, CB_Q, CB_O);
    cb_wait_front(CB_W, VTC);
    for (uint32_t it = 0; it < items; it++) {
        cb_wait_front(CB_Q, Kt);
        cb_wait_front(CB_K, Kt);
        cb_wait_front(CB_G, Kt);
        prep();
        cb_pop_front(CB_Q, Kt);
        cb_pop_front(CB_K, Kt);
        cb_pop_front(CB_G, Kt);
        step();
        gated_norm(blk);
    }
    cb_pop_front(CB_W, VTC);
}
#endif  // REC_V2
