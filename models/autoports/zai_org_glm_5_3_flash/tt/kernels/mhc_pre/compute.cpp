// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC "pre" — compute kernel (all roles; MHC_K / MHC_LEADER / MHC_ROOT select the code, see mhc_pre.py).
//
// K core (8 x tiles = its 2 column tiles of the 4 streams, index s * 2 + j):
//   1. M = sum_j x_j @ fn_j  (FPU, fp32 DST, HiFi4)        -> CB_PART[0]   (rows = tokens, cols = the 24 mixes)
//      S = sum_j x_j * x_j   (SFPU, exact fp32)            -> CB_PART[1]   (per-column partial sums of x^2)
//   2. (leader) sum the 8 (M, S) partials of its column    -> CB_GSUM
//   3. h_j = sum_s x_{s,j} * pre_s (column broadcast)       -> CB_H[j];  S2 = sum_j h_j * h_j -> CB_S2
//   4. (leader) sum the 8 S2 partials                       -> CB_GSUM[0]
//   5. hn_j = h_j * inv (column broadcast) * w_j (row broadcast) -> CB_OUT (bf16 or fp32)
// Root:
//   1. M, S = sums of the 8 group partials; ss = row sums of S; inv = rsqrt(ss / W + eps); z = (M * inv) * scale + bias
//      -> CB_RED
//   2. CB_PLANE (z in the plane layout, written by the sync RISC) -> mhc_coeff_planes (one SFPU pass) -> CB_COEF
//   3. inv_h = rsqrt(rowsum(sum of the 8 S2 partials) / D + eps)  -> CB_INVR
// Every fp32 CB read by copy_tile is unpacked straight to DST (UnpackToDestFp32, set by the host), so the SFPU
// stages see exact fp32; CB_X (matmul operand) shares its allocation with CB_X32 (copy operand).

#include <cstdint>

#include "api/compute/common.h"
#include "api/debug/kernel_profiler.hpp"
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

#include "mhc_pre_common.hpp"
#include "mhc_pre_sfpu.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"

using namespace mhc;

namespace {

constexpr uint32_t ITERS = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t HC_EPS_BITS = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t INV_WIDTH_BITS = get_compile_time_arg_val(PH_CT0 + 2);   // 1 / 16384
constexpr uint32_t X_EPS_BITS = get_compile_time_arg_val(PH_CT0 + 3);       // rms eps of the streams
constexpr uint32_t INV_HIDDEN_BITS = get_compile_time_arg_val(PH_CT0 + 4);  // 1 / 4096
constexpr uint32_t H_EPS_BITS = get_compile_time_arg_val(PH_CT0 + 5);       // rms eps of the collapsed input

// dst[d] <- tile i of cb (exact fp32 for UnpackToDestFp32 CBs)
FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    copy_init(cb);
    copy_tile(cb, i, d);
}

FORCE_INLINE void add_dst(uint32_t a, uint32_t b) {  // dst[a] += dst[b]
    add_binary_tile_init();
    add_binary_tile(a, b, a);
}

FORCE_INLINE void square_dst(uint32_t d) {
    square_tile_init();
    square_tile(d);
}

// pack dst[d] into the next page of cb (caller reserved / pushes)
FORCE_INLINE void pack_to(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}

// dst[0] <- sum of n consecutive fp32 tiles of cb starting at `first`, stride `stride`
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

#if defined(MHC_K)
FORCE_INLINE void k_phase1() {
    CircularBuffer cb_x(CB_X), cb_x32(CB_X32), cb_fn(CB_FN), cb_part(CB_PART);
    cb_x.wait_front(X_TILES);
    cb_x32.wait_front(X_TILES);
    cb_fn.wait_front(X_TILES);
    cb_part.reserve_back(2);

    tile_regs_acquire();
    // M = sum_j x_j @ fn_j into dst 0 (fp32 accumulation over the core's 8 K tiles); the unpackers were configured
    // for the matmul by compute_kernel_hw_startup<SrcOrder::Reverse> (in0 = x -> SrcB, in1 = fn -> SrcA)
    matmul_init(CB_X, CB_FN);
    for (uint32_t j = 0; j < X_TILES; ++j) {
        matmul_tiles(CB_X, CB_FN, j, j, 0);
    }
    // every later copy_tile reads an fp32 CB unpacked straight to DST: SrcA -> Float32 once (copy_init does not
    // reconfigure data formats)
    reconfig_data_format_srca(CB_X32);
    // S = sum_j x_j^2 into dst 2 (dst 1 scratch)
    load(CB_X32, 0, 2);
    square_dst(2);
    for (uint32_t j = 1; j < X_TILES; ++j) {
        load(CB_X32, j, 1);
        square_dst(1);
        add_dst(2, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_PART);
    pack_to(2, CB_PART);
    tile_regs_release();
    cb_part.push_back(2);
    cb_fn.pop_front(X_TILES);
}

FORCE_INLINE void k_collapse() {
    CircularBuffer cb_pre(CB_PRE), cb_h(CB_H), cb_s2(CB_S2), cb_x(CB_X), cb_x32(CB_X32);
    cb_pre.wait_front(N_STREAMS);
#if defined(MHC_FPU_COLLAPSE)
    // h_j = sum_s diag(pre_s) @ x_{s,j}: CB_PRE holds the diagonal tiles (sync.cpp; Default unpack), CB_X is x through
    // the FPU like phase 1's matmul -> 4 matmuls accumulated in DST per column (HiFi4 on tf32 operands, fp32 sums; rows
    // >= T of D are 0 -> h rows >= T are exactly 0), instead of 4 x (2 loads + a column-broadcast SFPU multiply) + 3
    // SFPU adds per column with their inits (B=1 probe: ~15 us of the 'pre in -> hn packed' interval)
    reconfig_data_format(CB_X, CB_PRE);  // SrcA = in1 (x), SrcB = in0 (D) under SrcOrder::Reverse
    matmul_init(CB_PRE, CB_X);
    for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
        cb_h.reserve_back(1);
        tile_regs_acquire();
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            matmul_tiles(CB_PRE, CB_X, s, s * COLS_PER_CORE + j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_H);
        tile_regs_release();
        cb_h.push_back(1);
    }
    reconfig_data_format_srca(CB_X32);  // the later copy_tile reads: fp32 CBs unpacked straight to DST
#else
    for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
        cb_h.reserve_back(1);
        tile_regs_acquire();
        // h_j = sum_s x_{s,j} * bcast_col(pre_s)
        load(CB_X32, 0 * COLS_PER_CORE + j, 0);
        load(CB_PRE, 0, 2);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 2);
        for (uint32_t s = 1; s < N_STREAMS; ++s) {
            load(CB_X32, s * COLS_PER_CORE + j, 1);
            load(CB_PRE, s, 2);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(1, 2);
            add_dst(0, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_H);
        tile_regs_release();
        cb_h.push_back(1);
    }
#endif
    cb_pre.pop_front(N_STREAMS);
    cb_x.pop_front(X_TILES);
    cb_x32.pop_front(X_TILES);
#if defined(MHC_HN_FIRST)
}

// Stage 14 MHC_HN_FIRST: S2 after hn (the deferred 1/rms(h) is off the hn path; the leaders' round-2 sums too)
FORCE_INLINE void k_s2() {
    CircularBuffer cb_h(CB_H), cb_s2(CB_S2);
#endif

    // S2 = sum_j h_j^2 (per-column partial sums)
    cb_h.wait_front(COLS_PER_CORE);
    cb_s2.reserve_back(1);
    tile_regs_acquire();
#if !defined(MHC_DIAG_NOS2)  // (probe-only bisection: S2 = whatever DST holds, wrong numerics)
    load(CB_H, 0, 0);
    square_dst(0);
    for (uint32_t j = 1; j < COLS_PER_CORE; ++j) {
        load(CB_H, j, 1);
        square_dst(1);
        add_dst(0, 1);
    }
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_S2);  // (MHC_DIAG_NOS2 below: probe-only bisection)
    tile_regs_release();
    cb_s2.push_back(1);
#if defined(MHC_HN_FIRST)
    cb_h.pop_front(COLS_PER_CORE);
#endif
}

FORCE_INLINE void k_finish() {
    CircularBuffer cb_inv(CB_INV), cb_w(CB_W), cb_h(CB_H), cb_out(CB_OUT);
#if !defined(MHC_DEFER_INV)
    cb_inv.wait_front(1);
#endif
    cb_w.wait_front(COLS_PER_CORE);
    cb_out.reserve_back(COLS_PER_CORE);
#if defined(MHC_HN_FIRST)
    cb_h.wait_front(COLS_PER_CORE);  // (normally the S2 step waited for them)
#endif
    for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
        tile_regs_acquire();
        load(CB_H, j, 0);
#if !defined(MHC_DEFER_INV)
        load(CB_INV, 0, 1);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
#endif
        // MHC_DEFER_INV (Stage 14): hn = h * w without 1 / rms(h): published before round 2 completes; the consumers
        // (the readers' matmul outputs) apply the per-row 1 / rms(h) the root sends them (the matmul is linear in hn)
#if !defined(MHC_DIAG_NOFINISH)  // (probe-only bisection: hn = h, wrong numerics)
        load(CB_W, j, 1);
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_OUT);
        tile_regs_release();
    }
    cb_out.push_back(COLS_PER_CORE);
#if defined(MHC_HN_FIRST)
    cb_w.pop_front(COLS_PER_CORE);  // CB_H stays for k_s2; CB_INV is drained after round 2 (kernel_main)
#else
    cb_h.pop_front(COLS_PER_CORE);
#if defined(MHC_DEFER_INV)
    cb_inv.wait_front(1);  // round 2 still runs here (CB balance; this core's copy is not needed)
#endif
    cb_inv.pop_front(1);
    cb_w.pop_front(COLS_PER_CORE);
#endif
}
#endif  // MHC_K

#if defined(MHC_LEADER)
// group sums of round 1: CB_GATHER holds 8 x (M, S) -> CB_GSUM (M sum, S sum)
FORCE_INLINE void leader_round1() {
    CircularBuffer cb_g(CB_GATHER), cb_gs(CB_GSUM);
    cb_g.wait_front(2 * GROUP);
    cb_gs.reserve_back(2);
    tile_regs_acquire();
    sum_tiles(CB_GATHER, 0, 2, GROUP, 0, 2);
    sum_tiles(CB_GATHER, 1, 2, GROUP, 1, 2);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_GSUM);
    pack_to(1, CB_GSUM);
    tile_regs_release();
    cb_gs.push_back(2);
    cb_g.pop_front(2 * GROUP);
}

// group sum of round 2: CB_GATHER holds 8 S2 partials -> CB_GSUM page 0
FORCE_INLINE void leader_round2() {
    CircularBuffer cb_g(CB_GATHER), cb_gs(CB_GSUM);
    cb_g.wait_front(GROUP);
    cb_gs.reserve_back(1);
    tile_regs_acquire();
    sum_tiles(CB_GATHER, 0, 1, GROUP, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_GSUM);
    tile_regs_release();
    cb_gs.push_back(1);
    cb_g.pop_front(GROUP);
}
#endif  // MHC_LEADER

#if defined(MHC_ROOT)
FORCE_INLINE void root_round1() {
    CircularBuffer cb_g(CB_GATHER), cb_red(CB_RED), cb_sb(CB_SB);
    cb_g.wait_front(2 * N_GROUPS);
    cb_sb.wait_front(2);
#if defined(DEBUG_TAPS)
    {
        CircularBuffer cb_dbg(CB_DBG);
        cb_dbg.reserve_back(2);
        tile_regs_acquire();
        sum_tiles(CB_GATHER, 0, 2, N_GROUPS, 0, 2);
        sum_tiles(CB_GATHER, 1, 2, N_GROUPS, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_DBG);
        pack_to(1, CB_DBG);
        tile_regs_release();
        cb_dbg.push_back(2);
    }
#endif
    cb_red.reserve_back(1);
    tile_regs_acquire();
    sum_tiles(CB_GATHER, 0, 2, N_GROUPS, 0, 2);  // M
    sum_tiles(CB_GATHER, 1, 2, N_GROUPS, 1, 2);  // S
    row_rms_inv(1, INV_WIDTH_BITS, X_EPS_BITS);  // column 0 of dst 1: 1 / rms(x) per token
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 1);  // m = M * inv
    load(CB_SB, 0, 2);
    sfpu_mul_bcast_row_init();
    sfpu_mul_bcast_row(0, 2);  // z = m * scale (row 0 of the scale tile: per-mix scale)
    load(CB_SB, 1, 2);
    sfpu_add_bcast_row_init();
    sfpu_add_bcast_row(0, 2);  // z += bias
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_RED);
    tile_regs_release();
    cb_red.push_back(1);
    cb_g.pop_front(2 * N_GROUPS);
    cb_sb.pop_front(2);
}

FORCE_INLINE void root_coeffs() {
    CircularBuffer cb_plane(CB_PLANE), cb_coef(CB_COEF);
    cb_plane.wait_front(1);
#if defined(MHC_PRE_EARLY)
    // Stage 14: page 0 = the pre planes alone (the K cores' collapse starts while the Sinkhorn below still runs);
    // page 1 = every coefficient (the same values for pre: the same SFPU arithmetic on the same plane tile)
    cb_coef.reserve_back(1);
    tile_regs_acquire();
    load(CB_PLANE, 0, 0);
    MATH((SFPU_UNARY_INIT(unused)));
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, mhc_pre_planes, (HC_EPS_BITS), 0, VectorMode::RC_custom)));
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_COEF);
    tile_regs_release();
    cb_coef.push_back(1);
#if defined(MHC_PRE_SPLIT)
    cb_plane.pop_front(1);   // page 0 held the pre planes only
    cb_plane.wait_front(1);  // page 1: every plane (relaid after the pre message went out)
#endif
#endif
    cb_coef.reserve_back(1);
    tile_regs_acquire();
    load(CB_PLANE, 0, 0);
    MATH((SFPU_UNARY_INIT(unused)));
#if defined(MHC_PLANE_PROBE)
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, mhc_plane_probe, 0, VectorMode::RC_custom)));
#else
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, mhc_coeff_planes, (ITERS, HC_EPS_BITS), 0, VectorMode::RC_custom)));
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_COEF);
    tile_regs_release();
    cb_coef.push_back(1);
    cb_plane.pop_front(1);
}

FORCE_INLINE void root_round2() {
    CircularBuffer cb_g(CB_GATHER), cb_invr(CB_INVR);
    cb_g.wait_front(N_GROUPS);
    cb_invr.reserve_back(1);
    tile_regs_acquire();
    sum_tiles(CB_GATHER, 0, 1, N_GROUPS, 0, 1);
    row_rms_inv(0, INV_HIDDEN_BITS, H_EPS_BITS);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_INVR);
    tile_regs_release();
    cb_invr.push_back(1);
    cb_g.pop_front(N_GROUPS);
}
#endif  // MHC_ROOT

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("MH-COMPUTE");  // device-profiler zone (no-op without the profiler)
#if defined(MHC_K)
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_FN, CB_PART);
    k_phase1();
#if defined(MHC_LEADER)
    leader_round1();
#endif
    k_collapse();
#if defined(MHC_HN_FIRST)
    // Stage 14 (with MHC_DEFER_INV): hn = h * w first, then S2 and (leaders) the round-2 group sums; the 1/rms(h)
    // message of round 2 is drained last (CB balance; the consumers get it from the root)
    k_finish();
    k_s2();
#if defined(MHC_LEADER)
    leader_round2();
#endif
    {
        CircularBuffer cb_inv(CB_INV);
        cb_inv.wait_front(1);
        cb_inv.pop_front(1);
    }
#else
#if defined(MHC_LEADER)
    leader_round2();
#endif
    k_finish();
#endif
#elif defined(MHC_ROOT)
    compute_kernel_hw_startup(CB_GATHER, CB_RED);
    root_round1();
    root_coeffs();
    root_round2();
#endif
}
