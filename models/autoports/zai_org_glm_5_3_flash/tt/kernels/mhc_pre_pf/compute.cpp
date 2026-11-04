// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill mHC "pre" — compute kernel (PF_MEMBER / PF_ROOT, mhc_pre_pf_common.hpp). The per-row-block arithmetic is the
// decode program's (kernels/mhc_pre/compute.cpp), looped over the group's row blocks:
// Member, per row block (X_TILES x tiles, index s * CPC + j):
//   1. M = sum_k x_k @ fn_k (FPU, fp32 DST, HiFi4) -> CB_PART[0]; S = sum_k x_k^2 (SFPU exact fp32) -> CB_PART[1]
//   2. h_j = sum_s x_{s,j} * pre_s (column broadcast, exact fp32) -> CB_H[j];  S2 = sum_j h_j^2 -> CB_S2
//   3. hn_j = h_j * inv (column broadcast) * w_j (row broadcast) -> CB_OUT
// Root, per row block:
//   1. M, S = sums of the P partials; inv = rsqrt(rowsum(S) / 16384 + eps); z = (M * inv) * scale + bias -> CB_RED
//   2. CB_PLANE (z in the plane layout, written by the sync RISC) -> mhc_coeff_planes (one SFPU pass) -> CB_COEF
//   3. inv_h = rsqrt(rowsum(sum of the P S2 partials) / 4096 + eps) -> CB_INVR
// Compile-time: ITERS, HC_EPS, 1/16384, X_EPS, 1/4096, H_EPS (fp32 bits), P, CPC. Runtime: first row block, row block
// stride (G), row blocks.

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

#include "mhc_pre_pf_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_sfpu.hpp"

using namespace mhcpf;

namespace {

constexpr uint32_t ITERS = get_compile_time_arg_val(0);
constexpr uint32_t HC_EPS_BITS = get_compile_time_arg_val(1);
constexpr uint32_t INV_WIDTH_BITS = get_compile_time_arg_val(2);   // 1 / 16384
constexpr uint32_t X_EPS_BITS = get_compile_time_arg_val(3);       // rms eps of the streams
constexpr uint32_t INV_HIDDEN_BITS = get_compile_time_arg_val(4);  // 1 / 4096
constexpr uint32_t H_EPS_BITS = get_compile_time_arg_val(5);       // rms eps of the collapsed input
constexpr uint32_t P = get_compile_time_arg_val(6);                // members per group
constexpr uint32_t CPC = get_compile_time_arg_val(7);              // column tiles per member and stream
constexpr uint32_t X_TILES = N_STREAMS * CPC;
// PF_SKIP (timing bisect, PF_FPU path; wrong numerics, same CB / semaphore protocol): bit 0 the phase-1 matmuls, bit 1
// the root's coefficient / Sinkhorn pass, bit 2 the collapse + S2 matmuls, bit 3 the finish's loads / multiplies
#if !defined(PF_SKIP)
#define PF_SKIP 0
#endif

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

#if defined(PF_MEMBER)
FORCE_INLINE void m_phase1() {
    CircularBuffer cb_x(CB_X), cb_x32(CB_X32), cb_part(CB_PART);
    cb_part.reserve_back(2);
    tile_regs_acquire();
    // M = sum_k x_k @ fn_k into dst 0, one stream's CPC tiles at a time as the stream RISC lands them; the unpackers:
    // in0 = x -> SrcB, in1 = fn -> SrcA (SrcOrder::Reverse at startup), reconfigured per row block (the copy_tile
    // reads below switch SrcA to the fp32 x32 format)
    reconfig_data_format(CB_FN, CB_X);
    matmul_init(CB_X, CB_FN);
    for (uint32_t s = 0; s < N_STREAMS; ++s) {
        cb_x.wait_front((s + 1) * CPC);
#if !(PF_SKIP & 1)
        for (uint32_t j = s * CPC; j < (s + 1) * CPC; ++j) {
            matmul_tiles(CB_X, CB_FN, j, j, 0);
        }
#endif
    }
    cb_x32.wait_front(X_TILES);
#if defined(PF_FPU) && (PF_SKIP & 1)
#elif defined(PF_FPU)
    // S = diag(sum_k x_k @ x_k^T) into dst 2 (in1 transposed: row r . row r = sum_c x[r][c]^2 on the diagonal)
    reconfig_data_format(CB_X, CB_X);
    matmul_init(CB_X, CB_X, 1);
    for (uint32_t k = 0; k < X_TILES; ++k) {
        matmul_tiles(CB_X, CB_X, k, k, 2);
    }
#else
    reconfig_data_format_srca(CB_X32);
    // S = sum_k x_k^2 into dst 2 (dst 1 scratch)
    load(CB_X32, 0, 2);
    square_dst(2);
    for (uint32_t k = 1; k < X_TILES; ++k) {
        load(CB_X32, k, 1);
        square_dst(1);
        add_dst(2, 1);
    }
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_PART);
    pack_to(2, CB_PART);
    tile_regs_release();
    cb_part.push_back(2);
}

FORCE_INLINE void m_collapse() {
    CircularBuffer cb_pre(CB_PRE), cb_h(CB_H), cb_s2(CB_S2), cb_x(CB_X), cb_x32(CB_X32);
    cb_pre.wait_front(N_STREAMS);
    cb_h.reserve_back(CPC);
#if defined(PF_FPU)
    CircularBuffer cb_h16(CB_H16);
    cb_h16.reserve_back(CPC);
    // h_j = sum_s diag(pre_s) @ x_{s,j}: SrcA = in1 (x), SrcB = in0 (the diagonal tiles) under SrcOrder::Reverse
    reconfig_data_format(CB_X, CB_PRE);
    matmul_init(CB_PRE, CB_X);
    for (uint32_t j = 0; j < CPC; ++j) {
        tile_regs_acquire();
#if !(PF_SKIP & 4)
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            matmul_tiles(CB_PRE, CB_X, s, s * CPC + j, 0);
        }
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_H);
        tile_regs_release();
    }
    cb_h.push_back(CPC);
    cb_h16.push_back(CPC);
    cb_pre.pop_front(N_STREAMS);
    cb_x.pop_front(X_TILES);
    cb_x32.pop_front(X_TILES);

    // S2 = diag(sum_j h_j @ h_j^T)
    cb_h16.wait_front(CPC);
    cb_s2.reserve_back(1);
    reconfig_data_format(CB_H16, CB_H16);
    matmul_init(CB_H16, CB_H16, 1);
    tile_regs_acquire();
#if !(PF_SKIP & 4)
    for (uint32_t j = 0; j < CPC; ++j) {
        matmul_tiles(CB_H16, CB_H16, j, j, 0);
    }
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_S2);
    tile_regs_release();
    cb_s2.push_back(1);
    cb_h16.pop_front(CPC);
    reconfig_data_format_srca(CB_H);  // the finish's copy_tile reads: fp32 CBs unpacked straight to DST
}
#else
    for (uint32_t j = 0; j < CPC; ++j) {
        tile_regs_acquire();
        // h_j = sum_s x_{s,j} * bcast_col(pre_s)
        load(CB_X32, j, 0);
        load(CB_PRE, 0, 2);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 2);
        for (uint32_t s = 1; s < N_STREAMS; ++s) {
            load(CB_X32, s * CPC + j, 1);
            load(CB_PRE, s, 2);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(1, 2);
            add_dst(0, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_H);
        tile_regs_release();
    }
    cb_h.push_back(CPC);
    cb_pre.pop_front(N_STREAMS);
    cb_x.pop_front(X_TILES);
    cb_x32.pop_front(X_TILES);

    // S2 = sum_j h_j^2 (per-column partial sums)
    cb_h.wait_front(CPC);
    cb_s2.reserve_back(1);
    tile_regs_acquire();
    load(CB_H, 0, 0);
    square_dst(0);
    for (uint32_t j = 1; j < CPC; ++j) {
        load(CB_H, j, 1);
        square_dst(1);
        add_dst(0, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_S2);
    tile_regs_release();
    cb_s2.push_back(1);
}
#endif

FORCE_INLINE void m_finish() {
    CircularBuffer cb_inv(CB_INV), cb_h(CB_H), cb_out(CB_OUT);
    cb_inv.wait_front(1);
    cb_out.reserve_back(CPC);
    for (uint32_t j = 0; j < CPC; ++j) {
        tile_regs_acquire();
#if !(PF_SKIP & 8)
        load(CB_H, j, 0);
        load(CB_INV, 0, 1);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        load(CB_W, j, 1);
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_OUT);
        tile_regs_release();
    }
    cb_out.push_back(CPC);
    cb_h.pop_front(CPC);
    cb_inv.pop_front(1);
}
#endif  // PF_MEMBER

#if defined(PF_ROOT)
FORCE_INLINE void root_round1() {
    CircularBuffer cb_g(CB_GATHER), cb_red(CB_RED);
    cb_g.wait_front(2 * P);
    cb_red.reserve_back(1);
    tile_regs_acquire();
    sum_tiles(CB_GATHER, 0, 2, P, 0, 2);  // M
    sum_tiles(CB_GATHER, 1, 2, P, 1, 2);  // S
#if defined(PF_FPU)
    load(CB_SB, 2, 2);  // S holds diag(x x^T): keep the diagonal
    mul_binary_tile_init();
    mul_binary_tile(1, 2, 1);
#endif
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
    cb_g.pop_front(2 * P);
}

FORCE_INLINE void root_coeffs() {
    CircularBuffer cb_plane(CB_PLANE), cb_coef(CB_COEF);
    cb_plane.wait_front(1);
    cb_coef.reserve_back(1);
    tile_regs_acquire();
    load(CB_PLANE, 0, 0);
#if !(PF_SKIP & 2)
    MATH((SFPU_UNARY_INIT(unused)));
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

// round 2: the P S2 partials sit in pages 0..P-1 of a full 2P-page push (every round consumes the whole CB, so each
// round's pages start at the CB base, where the members write)
FORCE_INLINE void root_round2() {
    CircularBuffer cb_g(CB_GATHER), cb_invr(CB_INVR);
    cb_g.wait_front(2 * P);
    cb_invr.reserve_back(1);
    tile_regs_acquire();
    sum_tiles(CB_GATHER, 0, 1, P, 0, 1);
#if defined(PF_FPU)
    load(CB_SB, 2, 1);  // diag(h h^T): keep the diagonal
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
#endif
    row_rms_inv(0, INV_HIDDEN_BITS, H_EPS_BITS);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_INVR);
    tile_regs_release();
    cb_invr.push_back(1);
    cb_g.pop_front(2 * P);
}
#endif  // PF_ROOT

}  // namespace

void kernel_main() {
    const uint32_t rb0 = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t n_rb = get_arg_val<uint32_t>(2);
#if defined(PF_MEMBER)
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_FN, CB_PART);
    CircularBuffer cb_fn(CB_FN), cb_w(CB_W);
    cb_fn.wait_front(X_TILES);  // resident for the whole program
    cb_w.wait_front(CPC);
    for (uint32_t rb = rb0; rb < n_rb; rb += step) {
        m_phase1();
        m_collapse();
        m_finish();
    }
    cb_fn.pop_front(X_TILES);
    cb_w.pop_front(CPC);
#elif defined(PF_ROOT)
    compute_kernel_hw_startup(CB_GATHER, CB_RED);
    CircularBuffer cb_sb(CB_SB);
    cb_sb.wait_front(3);  // resident
    for (uint32_t rb = rb0; rb < n_rb; rb += step) {
        root_round1();
        root_coeffs();
        root_round2();
    }
    cb_sb.pop_front(3);
#endif
}
