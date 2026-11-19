// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK KDA scan cores — compute (TRISC): core (head, v-block) runs the chunk recurrence over the valid chunks for its
// Vt value-column tiles of the head's state:
//   U = T_inv (v_beta - kd S);  Y = q_decay S + intra U;  S' = S * final_decay + k_dec_t U
// = compute_recurrent of ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/compute/
// recurrent_chunk_scan.cpp (matrix_multiply / elementwise / multiply_by_decay / compute_value_new /
// compute_chunk_output / update_state) ported to circular buffers, chunk inputs from one merged ring; then the
// sigmoid-gated RMSNorm epilogue of kernels/kda_rec (compute.cpp + the REC_SPREAD partial-sum exchange): the row sum
// of squares over this core's Vt tiles is published (CB_PSUM, the writer sends it to the head's other cores) and the
// peers' partials arrive in CB_PIN; on = y * rsqrt(mean + eps) * o_norm * sigmoid(gate_out) -> bf16 CB_ON (H4).
// Compile-time (PH_CT0 +): Kt, Vt, NCH, EPS_BITS, INV_D_BITS (1 / 128), SPREAD.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/kda_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"
// (3'') GLM53_PMK_SCAN_ZONES=1 (standalone KdaPhase only: PMK_SCAN_ZONES_SA; composites ignore it, MK_PHASED): device-
// profiler zones around the scan's per-chunk steps on the math thread (each step in its own block scope)
#if defined(PMK_SCAN_ZONES_SA) && !defined(MK_PHASED) && defined(TRISC_MATH)
#include "api/debug/kernel_profiler.hpp"
#define PMK_SZ(n) DeviceZoneScopedN("PMK_S_" n)
#else
#define PMK_SZ(n)
#endif
#if defined(PMK_KDA_RNE) || defined(PMK_KDA_RNE_PACK)  // (GLM53_PMK_KDA_RNE) every fp32 tile a later FPU op reads is
                                                       // packed tf32-RNE-rounded: the FPU's
// srcA / srcB read truncates fp32 to tf32 (a -2^-11 mean magnitude bias per operand; 14's round_probe2), the RNE
// pre-round makes that read exact (14 17:30:25Z: the op path's KDA y deficit 0.994 -> 0.998)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/rne_tf32.hpp"
#endif

using namespace pk;

namespace {

constexpr uint32_t max_dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

// the tf32 RNE pre-round of DST tiles [0, n) before their pack (no-op without PMK_KDA_RNE); its SFPU init goes BEFORE
// the FPU op's init (the FPU init then reprograms the address modes: rne_tf32.hpp)
FORCE_INLINE void rne_init() {
#if defined(PMK_KDA_RNE)
    kda_rne_tf32_init();
#endif
}
FORCE_INLINE void rne_tiles(uint32_t n) {
#if defined(PMK_KDA_RNE)
    for (uint32_t t = 0; t < n; ++t) {
        kda_rne_tf32_tile(t);
    }
#else
    (void)n;
#endif
}

// out[Mt,Nt] (reserved / pushed on o) = A[Mt,K_] (tiles a_off + ..) @ B[K_,Nt] (tiles b_off + ..)
template <uint32_t Mt, uint32_t K_, uint32_t Nt>
FORCE_INLINE void matrix_multiply(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, uint32_t o) {
    constexpr uint32_t sc = kda::MatmulSubblock<Mt, Nt>::columns;
    constexpr uint32_t sr = kda::MatmulSubblock<Mt, Nt>::rows;
    cb_reserve_back(o, Mt * Nt);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    rne_init();
    matmul_block_init(a, b, false, sc, sr, K_);
    for (uint32_t mi = 0; mi < Mt; mi += sr) {
        for (uint32_t ni = 0; ni < Nt; ni += sc) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < K_; ++k) {
                matmul_block(a, b, a_off + mi * K_ + k, b_off + k * Nt + ni, 0, false, sc, sr, K_);
            }
            rne_tiles(sr * sc);  // every output here feeds a later matmul / FPU elementwise
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < sr; ++r) {
                for (uint32_t cc = 0; cc < sc; ++cc) {
                    pack_tile(r * sc + cc, o, (mi + r) * Nt + ni + cc);
                }
            }
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

template <bool Add, bool Rne = true>
FORCE_INLINE void elementwise(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    reconfig_data_format(a, b);
    if constexpr (Rne) {
        rne_init();
    }
    if constexpr (Add) {
        add_init(a, b);
    } else {
        sub_init(a, b);
    }
    for (uint32_t s = 0; s < n; s += max_dst_tiles) {
        const uint32_t bt = n - s < max_dst_tiles ? n - s : max_dst_tiles;
        tile_regs_acquire();
        if constexpr (Add) {
            add_block(a, b, a_off + s, b_off + s, 0, bt);
        } else {
            sub_block(a, b, a_off + s, b_off + s, 0, bt);
        }
        if constexpr (Rne) {
            rne_tiles(bt);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < bt; ++t) {
            pack_tile(t, o, s + t);
        }
        tile_regs_release();
    }
    cb_push_back(o, n);
}

#if !defined(PMK_KDA_CARRY_FP32)
// o[k, v] = state[k, v] * decay column k (mul_tiles_bcast_cols; multiply_by_decay of the op)
FORCE_INLINE void multiply_by_decay(
    uint32_t state, uint32_t decay, uint32_t d_off, uint32_t o, uint32_t Kt, uint32_t Vt) {
    const uint32_t n = Kt * Vt;
    cb_reserve_back(o, n);
    reconfig_data_format(state, decay);
    rne_init();
    mul_bcast_cols_init(state, decay);
    for (uint32_t s = 0; s < n; s += max_dst_tiles) {
        const uint32_t bt = n - s < max_dst_tiles ? n - s : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t t = 0; t < bt; ++t) {
            const uint32_t i = s + t;
            mul_tiles_bcast_cols(state, decay, i, d_off + i / Vt, t);
        }
        rne_tiles(bt);  // S * decay: read by the FPU add of the state update
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < bt; ++t) {
            pack_tile(t, o, s + t);
        }
        tile_regs_release();
    }
    cb_push_back(o, n);
}
#endif

#if defined(PMK_KDA_CARRY_FP32)
// (GLM53_PMK_KDA_CARRY=fp32; Stage 14 kprec7's op fix b0a41e568b7, coordinator 2026-10-26 04:57:37Z) the state chain on
// the SFPU: the FPU eltwise multiply (mul_tiles_bcast_cols) returns S x decay TRUNCATED to tf32 (~ -2^-12 per chunk,
// whatever the operands' rounding), compounding on the long-memory rows chunk after chunk. Here the new state is formed
// EXACTLY in DEST: S (tf32-exact: its previous pack rounded it), the decay column (the item's rounded fd pack) and the
// update U (its matmul's rounded pack) read through SrcA without loss, S * d by the SFPU column broadcast, + U by the
// SFPU add, ONE rounded pack (the gasket under RNE pack; kda_rne_tf32_tile under RNE 1). Two output tiles per DEST
// block (slots 0..3).
FORCE_INLINE void sfpu_state_update(
    uint32_t state, uint32_t decay, uint32_t d_off, uint32_t update, uint32_t o, uint32_t Kt, uint32_t Vt) {
    cb_reserve_back(o, Kt * Vt);
    for (uint32_t key = 0; key < Kt; ++key) {
        for (uint32_t v = 0; v < Vt; v += 2) {
            const uint32_t n = (v + 1 < Vt) ? 2 : 1;
            tile_regs_acquire();
            reconfig_data_format_srca(state);
            copy_init(state);
            for (uint32_t t = 0; t < n; ++t) {
                copy_tile(state, key * Vt + v + t, 2 * t);
            }
            reconfig_data_format_srca(decay);
            copy_init(decay);
            for (uint32_t t = 0; t < n; ++t) {
                copy_tile(decay, d_off + key, 2 * t + 1);
            }
            sfpu_mul_bcast_col_init();
            for (uint32_t t = 0; t < n; ++t) {
                sfpu_mul_bcast_col(2 * t, 2 * t + 1);  // S[r][c] * d[r], fp32
            }
            reconfig_data_format_srca(update);
            copy_init(update);
            for (uint32_t t = 0; t < n; ++t) {
                copy_tile(update, key * Vt + v + t, 2 * t + 1);
            }
            add_binary_tile_init();
            for (uint32_t t = 0; t < n; ++t) {
                add_binary_tile(2 * t, 2 * t + 1, 2 * t);  // + k_dec_t U, fp32
#if defined(PMK_KDA_RNE)
                kda_rne_tf32_tile(2 * t);  // the SFPU form's rounding (the pack form rounds in the gasket)
#endif
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < n; ++t) {
                pack_tile(2 * t, o, key * Vt + v + t);
            }
            tile_regs_release();
        }
    }
    cb_push_back(o, Kt * Vt);
}
#endif

FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, i, d);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t Vt = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t INV_D_BITS = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t SPREAD = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t PG = in_pages(Kt, Vt);
    constexpr uint32_t I_VB = in_vbeta(Kt, Vt), I_KD = in_kd(Kt, Vt), I_QD = in_qdec(Kt, Vt), I_IN = in_intra(Kt, Vt);
    constexpr uint32_t I_TI = in_tinv(Kt, Vt), I_FD = in_fd(Kt, Vt), I_KDT = in_kdt(Kt, Vt), I_GO = in_go(Kt, Vt);
    static_assert(Vt <= 4 && KV <= 2 * max_dst_tiles, "scan: value tiles per core");

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN, CB_S, CB_SCR);
#if defined(PMK_KDA_RNE_PACK)
    kda_rne_pack_arm();  // the packer gasket rounds every fp32 pack (y included; the bf16 ON reconfig clears it)
#endif
    cb_wait_front(CB_W, Vt + 1);
    const uint32_t nvalid = read_tile_value(CB_W, 0, CTL_ROW);
    cb_wait_front(CB_S, KV);

    for (uint32_t c = 0; c < nvalid; ++c) {
        cb_wait_front(CB_IN, PG);
        const uint32_t cur = c == 0 ? CB_S : CB_SR;
        const bool last = c + 1 == nvalid;
        const uint32_t dest = last ? CB_FS : CB_SR;
        if (c) {
            PMK_SZ("wait_s");
            cb_wait_front(CB_SR, KV);
        }
        {
            PMK_SZ("u");
            // U = T_inv (v_beta - kd S)
            matrix_multiply<1, Kt, Vt>(CB_IN, I_KD, cur, 0, CB_SCR);
            cb_wait_front(CB_SCR, Vt);
            elementwise<false>(CB_IN, I_VB, CB_SCR, 0, CB_OI, Vt);
            cb_wait_front(CB_OI, Vt);
            cb_pop_front(CB_SCR, Vt);
            matrix_multiply<1, 1, Vt>(CB_IN, I_TI, CB_OI, 0, CB_VN);
            cb_wait_front(CB_VN, Vt);
            cb_pop_front(CB_OI, Vt);
        }
        {
            PMK_SZ("y");
            // Y = q_decay S + intra U
            matrix_multiply<1, Kt, Vt>(CB_IN, I_QD, cur, 0, CB_SCR);
            cb_wait_front(CB_SCR, Vt);
            matrix_multiply<1, 1, Vt>(CB_IN, I_IN, CB_VN, 0, CB_OI);
            cb_wait_front(CB_OI, Vt);
            elementwise<true, false>(
                CB_SCR, 0, CB_OI, 0, CB_O, Vt);  // y: the epilogue reads it by copy / SFPU (no pre-round)
            cb_wait_front(CB_O, Vt);
            cb_pop_front(CB_SCR, Vt);
            cb_pop_front(CB_OI, Vt);
        }
        {
            PMK_SZ("state");
            // S' = S * final_decay + k_dec_t U
            matrix_multiply<Kt, 1, Vt>(CB_IN, I_KDT, CB_VN, 0, CB_SU);
            cb_wait_front(CB_SU, KV);
            cb_pop_front(CB_VN, Vt);
#if defined(PMK_KDA_CARRY_FP32)  // S * d + U in DEST (fp32), one rounded pack
            sfpu_state_update(cur, CB_IN, I_FD, CB_SU, dest, Kt, Vt);
            cb_pop_front(cur, KV);
            cb_pop_front(CB_SU, KV);
#else
            multiply_by_decay(cur, CB_IN, I_FD, CB_ST, Kt, Vt);
            cb_wait_front(CB_ST, KV);
            elementwise<true>(CB_ST, 0, CB_SU, 0, dest, KV);
            cb_pop_front(cur, KV);
            cb_pop_front(CB_ST, KV);
            cb_pop_front(CB_SU, KV);
#endif
        }

        // gated RMSNorm: partial sum of squares (column tile) -> PSUM (to the peers) and OI (own copy)
        {
            PMK_SZ("sumsq");
            cb_reserve_back(CB_SCR, Vt);
            reconfig_data_format_srca(CB_O);
            copy_init(CB_O);
            tile_regs_acquire();
            for (uint32_t v = 0; v < Vt; ++v) {
                copy_tile(CB_O, v, v);
            }
            square_tile_init();
            for (uint32_t v = 0; v < Vt; ++v) {
                square_tile(v);
            }
            rne_tiles(Vt);  // y^2: read by the FPU reduce
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t v = 0; v < Vt; ++v) {
                pack_tile(v, CB_SCR, v);
            }
            tile_regs_release();
            cb_push_back(CB_SCR, Vt);
            cb_wait_front(CB_SCR, Vt);
            cb_reserve_back(CB_PSUM, 1);
            cb_reserve_back(CB_OI, Vt);  // a Vt-block (only tile 0 is read): blocks on a ring never straddle its wrap
            reconfig_data_format(CB_SCR, CB_W);
            rne_init();
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SCR, CB_W, CB_PSUM);
            tile_regs_acquire();
            for (uint32_t v = 0; v < Vt; ++v) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SCR, CB_W, v, Vt, 0);
            }
            rne_tiles(
                1);  // the own sum is read back from OI (an FPU-mode CB: srcA) -- the same value the peers get exact
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_PSUM, 0);
            pack_tile(0, CB_OI, 0);
            tile_regs_release();
            reduce_uninit();
            cb_push_back(CB_PSUM, 1);
            cb_push_back(CB_OI, Vt);
            cb_pop_front(CB_SCR, Vt);
        }
        // inv = rsqrt(sum / d + eps) over the head's SPREAD partials; on = y * inv * w * sigmoid(gate_out)
        if constexpr (SPREAD > 1) {
            PMK_SZ("wait_pin");
            cb_wait_front(CB_PIN, SPREAD - 1);
        }
        {
            PMK_SZ("epilogue");
            cb_wait_front(CB_OI, Vt);
            cb_reserve_back(CB_ON, Vt);
            tile_regs_acquire();
            load(CB_OI, 0, 0);
            if constexpr (SPREAD > 1) {
                for (uint32_t p = 0; p < SPREAD - 1; ++p) {
                    load(CB_PIN, p, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
            }
            binop_with_scalar_tile_init();
            mul_unary_tile(0, INV_D_BITS);
            add_unary_tile(0, EPS_BITS);
            rsqrt_tile_init();
            rsqrt_tile(0);
            for (uint32_t v = 0; v < Vt; ++v) {
                load(CB_O, v, 3 + v);
                sfpu_mul_bcast_col_init();
                sfpu_mul_bcast_col(3 + v, 0);
                load(CB_W, v, 2);
                sfpu_mul_bcast_row_init();
                sfpu_mul_bcast_row(3 + v, 2);
                load(CB_IN, I_GO + v, 2);
                sigmoid_tile_init();
                sigmoid_tile(2);
                mul_binary_tile_init();
                mul_binary_tile(3 + v, 2, 3 + v);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_ON);
            for (uint32_t v = 0; v < Vt; ++v) {
                pack_tile(3 + v, CB_ON, v);
            }
            tile_regs_release();
            pack_reconfig_data_format(CB_SCR);
#if defined(PMK_KDA_RNE_PACK)
            kda_rne_pack_arm();
#endif
            cb_push_back(CB_ON, Vt);
        }
        cb_pop_front(CB_O, Vt);
        cb_pop_front(CB_OI, Vt);
        if constexpr (SPREAD > 1) {
            cb_pop_front(CB_PIN, SPREAD - 1);
        }
        cb_pop_front(CB_IN, PG);
    }
#if defined(PMK_KDA_RNE_PACK)
    kda_rne_pack_disarm();  // the composite's next phase packs unrounded
#endif
}
