// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK mHC phases — compute kernel (PMK_MEMBER / PMK_ROOT, mhc_common.hpp). The pre is Stage 7a's
// kernels/mhc_pre_pf/compute.cpp PF_FPU arithmetic (one row block per group) on the RESIDENT residual (CB_X is never
// popped; CB_FN is popped after phase 1 and CB_W after the finish so the stream RISC can prefetch the next sublayer's);
// the post is Stage 7a's kernels/mhc_post_pf7a/compute.cpp pass (mix_sfpu.hpp, one SFPU pass per output tile) over
// the resident residual's column tiles, its 4 output tiles per column staged in CB_PSTAGE for the stream RISC's
// in-place copy-back. Full-sync fp32 DST (the post needs 8 tiles; the pre's <= 3 run under it too, as Stage 14's
// owner composites do).
// Member, per sublayer: [TOKEN] -> phase 1 (M = x @ fn^T, S = diag(x x^T): FPU) -> CB_PART; collapse
// h_j = sum_s diag(pre_s) @ x_{s,j} (FPU) -> CB_H; S2 = diag(h h^T) (FPU) -> CB_S2; finish hn_j = h_j * inv * w_j
// (SFPU) -> CB_OUT; post: per column j, per stream s: out_s = post_s y + sum_i C[i][s] x_i (SFPU FMAs) -> CB_PSTAGE.
// Root, per sublayer: 7a's root_round1 / root_coeffs / root_round2.
// Row blocks: NRB per group and sublayer (PMK_NRB); PMK_RESIDENT = 0 streams each block through CB_X (popped after the
// collapse, and again after the block's post), = 1 keeps the one block resident (never popped).
// Parts (PMK_MHC_PART): P1 = pre of sublayer 0 ([TOKEN] phase 1 / collapse / finish per block); P10 = post of sublayer
// 0, TOKEN, pre of sublayer 1; P16 = post of sublayer 1; 0 = all three. Root: P1 / P10 = NRB rounds, P16 nothing.
// PMK_PROBE (root): after every coefficient pass the N_TAPS tiles (z planes as loaded, the pass, the pass with ITERS =
// 1 and with ITERS = 2 on fresh copies of the planes) go to CB_DBG for the sync RISC (F_PROBE).
// Compile-time (PH_CT0 +): ITERS, HC_EPS, 1/16384, X_EPS, 1/4096, H_EPS (fp32 bits). Member runtime (ph_rt0 +):
// token_wait0 (resident: 1 = a P16 copy-back precedes this P1 in the program -> wait its token). Root runtime:
// the rounds of this part (NRB per part, NSUB x NRB in the all-in-one mode).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#include "mhc_common.hpp"
// ckernel::sfpu::pmk_coeff_planes (the pass + the round-end marker; rev 5): a header, see there. NO trailing comment on
// the include line: the composite hoister (tt/pmk_layer.py composite_source) only absolutises a bare #include "name"
#include "mhc_root_sfpu.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_sfpu.hpp"
#if defined(PMK_MEMBER)
// the post's one-pass mix (7a's mhc_post_pf7a) pulls the LLK's internal sfpu/ckernel_sfpu_binary_bcast.h into the
// translation unit; the root's TU stays exactly 7a's mhc_pre_pf root TU (silicon 10:08Z / 10:3xZ: the root's Sinkhorn
// loop produced the ITERS = 1 result (comb max |diff| 0.7373 = the host's iters=1 candidate to 4 digits) although
// KERNEL_COMPILE_TIME_ARGS carried 20; the pass and its call are byte-identical to 7a's, so the TU-level differences --
// this include and the compile-time-bounded outer round loop below -- are removed)
#include "mhc_mix_sfpu.hpp"
#endif

using namespace pmk_mhc;

namespace {

constexpr uint32_t ITERS = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t HC_EPS_BITS = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t INV_WIDTH_BITS = get_compile_time_arg_val(PH_CT0 + 2);   // 1 / 16384
constexpr uint32_t X_EPS_BITS = get_compile_time_arg_val(PH_CT0 + 3);       // rms eps of the streams
constexpr uint32_t INV_HIDDEN_BITS = get_compile_time_arg_val(PH_CT0 + 4);  // 1 / 4096
constexpr uint32_t H_EPS_BITS = get_compile_time_arg_val(PH_CT0 + 5);       // rms eps of the collapsed input
constexpr uint32_t NSUB = PMK_NSUB;
constexpr uint32_t NRB = PMK_NRB;

// dst[d] <- tile i of cb (exact fp32 for UnpackToDestFp32 CBs)
FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    copy_init(cb);
    copy_tile(cb, i, d);
}

FORCE_INLINE void add_dst(uint32_t a, uint32_t b) {  // dst[a] += dst[b]
    add_binary_tile_init();
    add_binary_tile(a, b, a);
}

FORCE_INLINE void pack_to(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}

// dst[d] <- sum of n consecutive fp32 tiles of cb starting at `first`, stride `stride` (dst[tmp] scratch)
FORCE_INLINE void sum_tiles(uint32_t cb, uint32_t first, uint32_t stride, uint32_t n, uint32_t d, uint32_t tmp) {
    load(cb, first, d);
    for (uint32_t k = 1; k < n; ++k) {
        load(cb, first + k * stride, tmp);
        add_dst(d, tmp);
    }
}

// dst[d] (a per-column partial-sum tile) <- rsqrt(rowsum(dst[d]) * inv_n + eps) in column 0
FORCE_INLINE void row_rms_inv(uint32_t d, uint32_t inv_n_bits, uint32_t eps_bits) {
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(d);
    binop_with_scalar_tile_init();
    mul_unary_tile(d, inv_n_bits);
    add_unary_tile(d, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(d);
}

#if defined(PMK_MEMBER)
// 7a m_phase1 (PF_FPU) on the resident residual: M = sum_k x_k @ fn_k (dst 0), S = diag(sum_k x_k @ x_k^T) (dst 2)
FORCE_INLINE void m_phase1() {
    CircularBuffer cb_part(CB_PART), cb_x(CB_X), cb_x32(CB_X32);
    cb_x.wait_front(X_TILES);  // resident: satisfied forever; streamed: the block landed
    cb_x32.wait_front(X_TILES);
    cb_part.reserve_back(2);
    tile_regs_acquire();
    reconfig_data_format(CB_FN, CB_X);
    matmul_init(CB_X, CB_FN);
    for (uint32_t k = 0; k < X_TILES; ++k) {
        matmul_tiles(CB_X, CB_FN, k, k, 0);
    }
    reconfig_data_format(CB_X, CB_X);
    matmul_init(CB_X, CB_X, 1);
    for (uint32_t k = 0; k < X_TILES; ++k) {
        matmul_tiles(CB_X, CB_X, k, k, 2);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_PART);
    pack_to(2, CB_PART);
    tile_regs_release();
    cb_part.push_back(2);
}

// 7a m_collapse (PF_FPU): h_j = sum_s diag(pre_s) @ x_{s,j} -> CB_H (+ alias H16); S2 = diag(h h^T) -> CB_S2
FORCE_INLINE void m_collapse() {
    CircularBuffer cb_pre(CB_PRE), cb_h(CB_H), cb_h16(CB_H16), cb_s2(CB_S2);
    cb_pre.wait_front(N_STREAMS);
    cb_h.reserve_back(CPC);
    cb_h16.reserve_back(CPC);
    reconfig_data_format(CB_X, CB_PRE);
    matmul_init(CB_PRE, CB_X);
    for (uint32_t j = 0; j < CPC; ++j) {
        tile_regs_acquire();
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            matmul_tiles(CB_PRE, CB_X, s, s * CPC + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_H);
        tile_regs_release();
    }
    cb_h.push_back(CPC);
    cb_h16.push_back(CPC);
    cb_pre.pop_front(N_STREAMS);
#if !PMK_RESIDENT
    CircularBuffer cb_x(CB_X), cb_x32(CB_X32);
    cb_x.pop_front(X_TILES);  // streamed: the block's last pre read
    cb_x32.pop_front(X_TILES);
#endif

    cb_h16.wait_front(CPC);
    cb_s2.reserve_back(1);
    reconfig_data_format(CB_H16, CB_H16);
    matmul_init(CB_H16, CB_H16, 1);
    tile_regs_acquire();
    for (uint32_t j = 0; j < CPC; ++j) {
        matmul_tiles(CB_H16, CB_H16, j, j, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_S2);
    tile_regs_release();
    cb_s2.push_back(1);
    cb_h16.pop_front(CPC);
    reconfig_data_format_srca(CB_H);  // the finish's copy_tile reads: fp32 CBs unpacked straight to DST
}

// 7a m_finish: hn_j = h_j * bcast_col(inv) * bcast_row(w_j) -> CB_OUT
FORCE_INLINE void m_finish() {
    CircularBuffer cb_inv(CB_INV), cb_h(CB_H), cb_out(CB_OUT);
    cb_inv.wait_front(1);
    cb_out.reserve_back(CPC);
    for (uint32_t j = 0; j < CPC; ++j) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_H);  // fp32 CBs (unpack-to-dest) ...
        load(CB_H, j, 0);
        load(CB_INV, 0, 1);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        reconfig_data_format_srca(CB_W);  // ... and the bf16 norm-weight tile alternate on SrcA (silicon 10:08Z:
        load(CB_W, j, 1);                 // without this the bf16 tile was unpacked as fp32 -> hn wrong by O(1))
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_OUT);
        tile_regs_release();
    }
    cb_out.push_back(CPC);
    cb_h.pop_front(CPC);
    cb_inv.pop_front(1);
}

// 7a mhc_post_pf7a compute on the block in CB_X: per column j and stream s one DST pass. The block's y tiles (CPC)
// and coefficient tiles (8) are the front of CB_Y / CB_PCOEF (pushed per block, popped here per block: lane-landed
// y of NRB blocks sits at k * CPC pages = the ring position, DRAM-read y is a one-block ring)
FORCE_INLINE void m_post() {
    CircularBuffer cb_y(CB_Y), cb_pc(CB_PCOEF), cb_st(CB_PSTAGE), cb_x(CB_X), cb_x32(CB_X32);
    cb_x.wait_front(X_TILES);
    cb_x32.wait_front(X_TILES);
    cb_y.wait_front(CPC);
    cb_pc.wait_front(2 * N_STREAMS);
    for (uint32_t j = 0; j < CPC; ++j) {
        cb_st.reserve_back(N_STREAMS);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            tile_regs_acquire();
            reconfig_data_format_srca(CB_Y);  // bf16 / fp32 y and the fp32 residual + coefficients alternate
            copy_init(CB_Y);
            copy_tile(CB_Y, j, D_Y);
            reconfig_data_format_srca(CB_X32);
            copy_init(CB_X32);
            for (uint32_t i = 0; i < N_STREAMS; ++i) {
                copy_tile(CB_X32, i * CPC + j, D_X0 + i);
            }
            copy_tile(CB_PCOEF, 2 * s, D_K);  // (CB_PCOEF: the same fp32 format / unpack mode)
            copy_tile(CB_PCOEF, 2 * s + 1, D_L);
            sfpu_bcast_col_init();  // col-0 lane mask (LREG6) + the broadcast replay; the copies may reprogram addr
                                    // mods
            MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, pmk_mix_tile, D_Y, D_K, D_OUT, VectorMode::None)));
            tile_regs_commit();
            tile_regs_wait();
            pack_to(D_OUT, CB_PSTAGE);
            tile_regs_release();
        }
        cb_st.push_back(N_STREAMS);
    }
    cb_y.pop_front(CPC);
    cb_pc.pop_front(2 * N_STREAMS);
#if !PMK_RESIDENT
    cb_x.pop_front(X_TILES);  // streamed: the block's post read
    cb_x32.pop_front(X_TILES);
#endif
}
#endif  // PMK_MEMBER

#if defined(PMK_ROOT)

// DST tile 0 -> CB_COEF page 0, verified (pack thread; a no-op on the other TRISCs). Out-of-order packs (tile index
// 0: no fifo_wr_tile_ptr bump -- the CB is ONE page; rev 4's three plain packs advanced the tile pointer past it into
// CB_SB's scale / bias tiles = the SP 1,024 'rounds >= 2' z growth and SP 256's comb 14.25, README §9e) until the
// packed page carries the pass's end marker in plane PLANE_MARKER (lanes 0 and 31), then one final pack that starts
// after the SFPU's last store was observed. Bounded by PACK_MAX_ATTEMPTS. Returns the attempts (pack thread).
FORCE_INLINE uint32_t pack_coef_verified(uint32_t marker) {
    uint32_t attempts = 0;
#if defined(TRISC_PACK)
    pack_reconfig_data_format(CB_COEF);
    // BYTES (class 2, 2026-10-24): on a TRISC the CB interface holds 16-B units and CircularBuffer::get_write_ptr()
    // returns them unshifted (api/dataflow/circular_buffer.h:123-127). Unshifted, these polls read the kernel-config
    // ring (the marker was never seen) and pack_diag's two stores below landed there: at L4 A1 (TP8 resident, CB_RED
    // 0xa3680) on 0xb068 / 0xb070 = this root's NCRISC text when A1's config sits at the ring start (fillctl 02:05Z:
    // ring text != ELF at +0x938, the PC spinning at +0x93c)
    const uint32_t page = CircularBuffer(CB_COEF).get_write_ptr() << cb_addr_shift;
    volatile uint32_t* lane0 = reinterpret_cast<volatile uint32_t*>(page + 4u * plane_off(PLANE_MARKER, 0));
    volatile uint32_t* lane31 = reinterpret_cast<volatile uint32_t*>(page + 4u * plane_off(PLANE_MARKER, TILE - 1));
    bool seen = false;
    while (!seen && attempts < PACK_MAX_ATTEMPTS) {
        pack_tile<true>(0, CB_COEF, 0);
        ++attempts;
        for (uint32_t i = 0; i < PACK_POLL_READS && !seen; ++i) {
            asm volatile("fence");  // Blackhole: the RISC's write-through L1 cache (invalidate_l1_cache)
            seen = (*lane0 == marker) && (*lane31 == marker);
        }
    }
    pack_tile<true>(0, CB_COEF, 0);  // final: after the SFPU finished (or after the bounded wait: the diag says)
#else
    (void)marker;
#endif
    return attempts;
}

// the attempt counts -> the consumed CB_RED page (free until the next round's z pack), plane PLANE_DIAG lanes 0 / 1
FORCE_INLINE void pack_diag(uint32_t a1, uint32_t a2) {
#if defined(TRISC_PACK)
    const uint32_t red = CircularBuffer(CB_RED).get_write_ptr() << cb_addr_shift;  // bytes (see pack_coef_verified)
    *reinterpret_cast<volatile uint32_t*>(red + 4u * plane_off(PLANE_DIAG, 0)) = a1;
    *reinterpret_cast<volatile uint32_t*>(red + 4u * plane_off(PLANE_DIAG, 1)) = a2;
    asm volatile("fence");
#else
    (void)a1;
    (void)a2;
#endif
}

FORCE_INLINE void root_round1() {
    CircularBuffer cb_g(CB_GATHER), cb_red(CB_RED);
    cb_g.wait_front(2 * P);
    cb_red.reserve_back(1);
    tile_regs_acquire();
    sum_tiles(CB_GATHER, 0, 2, P, 0, 2);  // M
    sum_tiles(CB_GATHER, 1, 2, P, 1, 2);  // S
    load(CB_SB, 2, 2);                    // S holds diag(x x^T): keep the diagonal
    mul_binary_tile_init();
    mul_binary_tile(1, 2, 1);
    row_rms_inv(1, INV_WIDTH_BITS, X_EPS_BITS);  // column 0 of dst 1: 1 / rms(x) per token
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 1);  // m = M * inv
    load(CB_SB, 0, 2);
    sfpu_mul_bcast_row_init();
    sfpu_mul_bcast_row(0, 2);  // z = m * scale
    load(CB_SB, 1, 2);
    sfpu_add_bcast_row_init();
    sfpu_add_bcast_row(0, 2);  // z += bias
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_RED);
    tile_regs_release();
    cb_red.push_back(1);
    cb_g.pop_front(2 * P);
}

FORCE_INLINE void root_coeffs(uint32_t r) {
    // Rev 5 (README §9e). Rev 3 packed the pass ONCE and the COEF tile held a snapshot of the pass (SP 256: exactly
    // the Sinkhorn loop-start state, SP 1,024: mid-pass states per block; the probe's re-run pass -- tap 1 -- was
    // right at SP 256 and wrong at SP 1,024), i.e. the pack does not reliably see the finished pass. Rev 4 repeated
    // the pass and packed three times into the ONE-page CB_COEF without a push: pack_tile bumps fifo_wr_tile_ptr,
    // so packs 2 / 3 landed in CB_SB's scale / bias tiles (the next CB) -- every later round's z = m * scale + bias
    // on garbage (SP 1,024 blocks >= 8 and sublayer 1: z x 50..100 per round; SP 256 comb 14.25) -- the 22:06Z
    // 'rounds >= 2 read wrong rows' reading. Now: cycle 1 = load, init, pass + marker, VERIFIED pack (out-of-order
    // into page 0 until the marker is in L1, then once more); cycle 2 = a plain reload / pack (the probe's measured
    // sequence); cycle 3 = load, pass + marker, verified pack; one push. The attempt counts go to the diag lanes.
    CircularBuffer cb_plane(CB_PLANE), cb_coef(CB_COEF);
    const uint32_t marker = marker_bits(r);     // cycle 1's end marker
    const uint32_t marker_f = marker_final(r);  // cycle 3's (distinct: cycle 2's plain pack may still be in flight)
    cb_plane.wait_front(1);
    cb_coef.reserve_back(1);
    tile_regs_acquire();  // cycle 1
    load(CB_PLANE, 0, 0);
    MATH((SFPU_UNARY_INIT(unused)));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, pmk_coeff_planes, (ITERS, HC_EPS_BITS), 0, VectorMode::RC_custom, marker)));
    tile_regs_commit();
    tile_regs_wait();
    const uint32_t a1 = pack_coef_verified(marker);
    tile_regs_release();
    tile_regs_acquire();  // cycle 2: the plain reload / pack of the probe sequence (page 0, overwritten below)
    load(CB_PLANE, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_COEF);
    pack_tile<true>(0, CB_COEF, 0);
    tile_regs_release();
    tile_regs_acquire();  // cycle 3: the pass again (the probe's tap 1), verified pack
    load(CB_PLANE, 0, 0);
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, pmk_coeff_planes, (ITERS, HC_EPS_BITS), 0, VectorMode::RC_custom, marker_f)));
    tile_regs_commit();
    tile_regs_wait();
    const uint32_t a2 = pack_coef_verified(marker_f);
    pack_diag(a1, a2);
    tile_regs_release();
    cb_coef.push_back(1);
#if defined(PMK_PROBE)
    // taps of this round: 0 the z planes as loaded, 1 the pass again (ITERS), 2 the pass with ITERS = 1, 3 with
    // ITERS = 2 -- each on a fresh copy of the plane tile, packed as is (plane layout; the host re-lays them)
    CircularBuffer cb_dbg(CB_DBG);
    cb_dbg.reserve_back(N_TAPS);
    tile_regs_acquire();
    load(CB_PLANE, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_DBG);
    tile_regs_release();
    tile_regs_acquire();
    load(CB_PLANE, 0, 0);
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, mhc_coeff_planes, (ITERS, HC_EPS_BITS), 0, VectorMode::RC_custom)));
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_DBG);
    tile_regs_release();
    tile_regs_acquire();
    load(CB_PLANE, 0, 0);
    MATH((
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, mhc_coeff_planes, (1u, HC_EPS_BITS), 0, VectorMode::RC_custom)));
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_DBG);
    tile_regs_release();
    tile_regs_acquire();
    load(CB_PLANE, 0, 0);
    MATH((
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, mhc_coeff_planes, (2u, HC_EPS_BITS), 0, VectorMode::RC_custom)));
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_DBG);
    tile_regs_release();
    cb_dbg.push_back(N_TAPS);
#endif
    cb_plane.pop_front(1);
}

FORCE_INLINE void root_round2() {
    CircularBuffer cb_g(CB_GATHER), cb_invr(CB_INVR);
    cb_g.wait_front(2 * P);
    cb_invr.reserve_back(1);
    tile_regs_acquire();
    sum_tiles(CB_GATHER, 0, 1, P, 0, 1);
    load(CB_SB, 2, 1);  // diag(h h^T): keep the diagonal
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    row_rms_inv(0, INV_HIDDEN_BITS, H_EPS_BITS);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_INVR);
    tile_regs_release();
    cb_invr.push_back(1);
    cb_g.pop_front(2 * P);
}
#endif  // PMK_ROOT

}  // namespace

void kernel_main() {
#if defined(PMK_MEMBER)
    const uint32_t token_wait0 = get_arg_val<uint32_t>(ph_rt0 + 0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_FN, CB_PART);
    CircularBuffer cb_fn(CB_FN), cb_w(CB_W), cb_tok(CB_TOKEN);
    auto pre = [&](bool token) {
#if PMK_RESIDENT
        if (token) {  // the preceding post's in-place copy-back has landed
            cb_tok.wait_front(1);
            cb_tok.pop_front(1);
        }
#else
        (void)token;
#endif
        cb_fn.wait_front(X_TILES);
        cb_w.wait_front(CPC);
        for (uint32_t k = 0; k < NRB; ++k) {
            m_phase1();
            if (k + 1 == NRB) {
                cb_fn.pop_front(X_TILES);  // the stream RISC prefetches the next sublayer's slab from here on
            }
            m_collapse();
            m_finish();
        }
        cb_w.pop_front(CPC);
    };
    auto post = [&]() {
        for (uint32_t k = 0; k < NRB; ++k) {
            m_post();
        }
    };
    (void)pre;  // (a single-part build uses only its own bodies)
    (void)post;
#if PMK_MHC_PART == 1
    pre(token_wait0 != 0);
#elif PMK_MHC_PART == 10
    post();
    pre(true);
#elif PMK_MHC_PART == 16
    post();
#else
    pre(token_wait0 != 0);
    post();
    pre(true);
    post();
#endif
#elif defined(PMK_ROOT)
    const uint32_t n_rounds = get_arg_val<uint32_t>(ph_rt0 + 0);  // NSUB x NRB (runtime, as 7a's row-block loop)
    compute_kernel_hw_startup(CB_GATHER, CB_RED);
    CircularBuffer cb_sb(CB_SB);
    // rev 6: the SB tiles (scale / bias / identity) are the SUBLAYER's: waited before its NRB rounds, popped after
    // them (the stream part / the integrator's prelude pushes the next sublayer's; a part runs one sublayer)
    for (uint32_t r = 0; r < n_rounds; ++r) {  // (sublayer, row block) rounds in the sync RISC's order
        if (r % NRB == 0) {
            cb_sb.wait_front(3);
        }
        root_round1();
        root_coeffs(r);
        root_round2();
        if (r % NRB == NRB - 1) {
            cb_sb.pop_front(3);
        }
    }
#endif
}
