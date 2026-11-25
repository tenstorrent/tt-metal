// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Experimental single-token specialization of chunk GDN's preparation + scan.
// Flat BF16 Q/K/V extraction and FP32 normalization/exp preparation are copied
// from flat_recurrent. Ordered recurrence below is copied from the passing
// raw_recurrent_ordered candidate. This combines those two independent changes.
// CB allocation: 0..14 and16..19 = ordered scan,15 = ordered diagonal decay;
// 20..28 = flat reader/preparation. In particular dl must NOT occupy CB20,
// which is the flat BF16 Q input. All ordered-scan CBs are FP32.
//
// FP32 packs occur after each of these preparation operations:
//   vb = v * beta; kb = k * beta; nkd = -(kb * decay);
//   qd = q * decay; intra = (q @ k^T) * I;
//   kd = k * 1; kt = transpose(kd); dl = I * decay; Tinv = I.
// With just row zero active, the 32-token chunk's triangular inverse is I and
// the inactive rows of intra vanish. Reuse the actual chunk scan implementation:
//   diff = nkd @ S + I @ vb                 [one DST accumulation, FP32 pack]
//   u    = I @ diff                        [FP32 pack]
//   out  = qd @ S + intra @ u               [one DST accumulation, FP32 pack]
//   Snew = dl @ S + kt @ u                  [one DST accumulation, FP32 pack]
// In particular, output does not re-read the rounded updated state. The writer
// aliases Snew to S only after this core has consumed its complete old S slab.

#define GDN_DST_TILES 4
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "norm_stages.hpp"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp"

namespace {
constexpr uint32_t q = 0, k = 1, v = 2, decay = 3, beta = 4, state = 5;
constexpr uint32_t out = 6, final_state = 7, vbeta = 8, kbeta = 9, nkd = 10;
constexpr uint32_t qdecay = 11, intra = 12, tinv = 13, eye = 14;
constexpr uint32_t vnew = 16, diff = 17, kdec = 18, kdec_t = 19, dl = 15, upd = 32, state32 = 33, decay32 = 34;
constexpr GdnScanCbs scan{
    .dl = dl,
    .Tinv = tinv,
    .out = out,
    .vbeta = vbeta,
    .nkd = nkd,
    .qdecay = qdecay,
    .intra = intra,
    .vnew = vnew,
    .ointer = diff,
    .kdec_t = kdec_t,
    .eye = eye};

// Scalar multiply, used for the all-row diagonal decay and k's exact-one factor.
void scalar_mul(uint32_t a, uint32_t scalar, uint32_t o, uint32_t count) {
    CircularBuffer dest(o);
    dest.reserve_back(count);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scalar);
    mul_bcast_scalar_init(a, scalar);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scalar, tile, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, tile);
        tile_regs_release();
    }
    dest.push_back(count);
}
}  // namespace

// One-token scan. Same blocks and roundings as chunk_gdn_math.hpp scan_step<1, 4, 4> for diff / v_new / out,
// but the state update keeps the recurrent state in FP32:
//   shared scan_step:  S_new = (dl*I) @ S + kt @ u     -- an FPU matmul. Its srcA/srcB registers hold at most TF32
//                      (10-bit mantissa, truncated), so EVERY decode token re-truncated the whole state and the decay
//                      itself (exp(g) ~ 0.9999 has a 2^-11 grid there). Over thousands of tokens the long-memory heads
//                      drifted (measured: layer-1 relative error 0.002 -> 0.02 by 4K tokens; logits KL 13x prefill).
//   here:              U = kt @ u on the FPU (an increment, packed FP32 to CB32), then per state tile on the SFPU in
//                      FP32: S_new = exp(g) * S + U, with S (CB33, an L1 copy of CB5), exp(g) (CB34) and U unpacked
//                      straight to DST (UnpackToDestFp32 on those CBs only; CB3/CB5 keep the srcA path because FPU
//                      consumers read them -- setting the FP32 unpack mode on them corrupts those matmuls).
//                      A single acquire mixing the SFPU product and an accumulating matmul did not work.
// diff and out still read S through the FPU (one-off TF32 read, not fed back into the state).
// Stage test: tests/gdn_decode_state_drift.py (2000 long-memory steps vs fp64: 0.19 -> 0.0034 state rel err).
void scan_exact_state(uint32_t cur_S, uint32_t dst) {
    constexpr uint32_t Kt = 4, Vt = 4, cv = Vt, ck = Kt, kv = Kt * Vt;
    {
        WAIT(nkd, ck);
        WAIT(vbeta, cv);
        WAIT(cur_S, kv);
        CircularBuffer(diff).reserve_back(cv);
        reconfig_data_format(cur_S, nkd);
        pack_reconfig_data_format(diff);
        matmul_init(nkd, cur_S, 0);
        tile_regs_acquire();
        for (uint32_t j = 0; j < cv; j++) {
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(nkd, cur_S, ki, ki * Vt + j, j);  // DST[j]  = nkd @ S
            }
            matmul_tiles(eye, vbeta, 0, j, j);  // DST[j] += I @ v_beta
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cv; j++) {
            pack_tile(j, diff, j);
        }
        tile_regs_release();
        CircularBuffer(diff).push_back(cv);
        POP(nkd, ck);
        WAIT(diff, cv);
        POP(vbeta, cv);
    }
    {
        WAIT(tinv, 1);
        mm(tinv, diff, vnew, 1, 1, Vt, false, true);  // v_new = T_inv @ diff
        WAIT(vnew, cv);
        POP(tinv, 1);
        POP(diff, cv);
    }
    {
        WAIT(qdecay, ck);
        WAIT(intra, 1);
        CircularBuffer(out).reserve_back(cv);
        matmul_init(qdecay, cur_S, 0);
        tile_regs_acquire();
        for (uint32_t j = 0; j < cv; j++) {
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(qdecay, cur_S, ki, ki * Vt + j, j);
            }
            matmul_tiles(intra, vnew, 0, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cv; j++) {
            pack_tile(j, out, j);
        }
        tile_regs_release();
        CircularBuffer(out).push_back(cv);
        POP(qdecay, ck);
        POP(intra, 1);
    }
    {
        // Pass A: U = k_t @ v_new on the FPU (an increment; its TF32 operand rounding is not fed back) -> FP32 CB32.
        WAIT(kdec_t, Kt);
        mm(kdec_t, vnew, upd, Kt, 1, Vt, false);
        WAIT(upd, kv);
        POP(kdec_t, Kt);
    }
    {
        // Pass B: S_new = exp(g) * S + U entirely on the SFPU in FP32 (all three unpacked straight to DST).
        WAIT(decay32, 1);
        WAIT(state32, kv);
        CircularBuffer(dst).reserve_back(kv);
        pack_reconfig_data_format(dst);
        for (uint32_t t = 0; t < kv; t++) {
            tile_regs_acquire();
            reconfig_data_format_srca(state32);
            copy_init(state32);
            copy_tile(state32, t, 0);
            reconfig_data_format_srca(decay32);
            copy_init(decay32);
            copy_tile(decay32, 0, 1);
            reconfig_data_format_srca(upd);
            copy_init(upd);
            copy_tile(upd, t, 2);
            sfpu_bcast_col_init();
            sfpu_mul_bcast_col(0, 1);  // DST0 = exp(g) * S
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);  // DST0 += U
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst, t);
            tile_regs_release();
        }
        CircularBuffer(dst).push_back(kv);
        POP(upd, kv);
        POP(state32, kv);
        POP(decay32, 1);
        POP(decay, 1);
    }
    POP(vnew, cv);
    POP(cur_S, kv);
}

// Normalize Q/K and prepare scalar gates in the same launch as recurrence.
void flat_prepare() {
    CircularBuffer rq(20), rk(21), rv(22), rg(23), rb(24);
    rq.wait_front(4);
    rk.wait_front(4);
    rv.wait_front(4);
    rg.wait_front(1);
    rb.wait_front(1);
    flat_norm::row_rsqrt<25, 26, 27, 4, 0x3f800000u, 0x358637bdu>(20);
    flat_norm::normalize<26, 4>(20, 27, false, 0);
    flat_norm::row_rsqrt<25, 26, 27, 4, 0x3f800000u, 0x358637bdu>(21);
    flat_norm::normalize<26, 4>(21, 27, false, 1);
    for (uint32_t which = 0; which < 3; ++which) {
        const uint32_t input = which == 0 ? 22 : (which == 1 ? 23 : 24);
        const uint32_t output = which == 0 ? 2 : (which == 1 ? 3 : 4);
        const uint32_t count = which == 0 ? 4 : 1;
        CircularBuffer dest(output);
        dest.reserve_back(count);
        if (which == 1) {
            CircularBuffer(decay32).reserve_back(1);
        }
        pack_reconfig_data_format(output);
        for (uint32_t t = 0; t < count; ++t) {
            tile_regs_acquire();
            reconfig_data_format_srca(input);
            copy_init(input);
            copy_tile(input, t, 0);
            if (which == 1) {
                exp_tile_init();
                exp_tile(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output, t);
            if (which == 1) {
                pack_tile(0, decay32, 0);  // same FP32 exp(g), for the SFPU state decay
            }
            tile_regs_release();
        }
        dest.push_back(count);
        if (which == 1) {
            CircularBuffer(decay32).push_back(1);
        }
    }
    rq.pop_front(4);
    rk.pop_front(4);
    rv.pop_front(4);
    rg.pop_front(1);
    rb.pop_front(1);
    // Keep the unit scaler for output RMS normalization.
}

// Preserve the BF16 recurrence-output pack and BF16 RMSNorm pack before gating.
void post_norm_gate() {
    constexpr uint32_t epsilon = get_compile_time_arg_val(3);
    CircularBuffer source(6), rounded(29), normalized(30), gamma(20), gate(22), output(31);
    source.wait_front(4);
    rounded.reserve_back(4);
    pack_reconfig_data_format(29);
    for (uint32_t t = 0; t < 4; ++t) {
        tile_regs_acquire();
        reconfig_data_format_srca(6);
        copy_init(6);
        copy_tile(6, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 29, t);
        tile_regs_release();
    }
    rounded.push_back(4);
    source.pop_front(4);
    rounded.wait_front(4);
    flat_norm::row_rsqrt<25, 26, 27, 4, 0x3c000000u, epsilon>(29);
    flat_norm::normalize<26, 4>(29, 20, true, 30);
    rounded.pop_front(4);
    normalized.wait_front(4);
    gate.wait_front(4);
    // binary_ng MUL packs its fused SiLU activation to BF16 before multiply.
    // Reuse the compute-owned rounded CB only after its previous pages are popped.
    rounded.reserve_back(4);
    pack_reconfig_data_format(29);
    for (uint32_t t = 0; t < 4; ++t) {
        tile_regs_acquire();
        reconfig_data_format_srca(22);
        copy_init(22);
        copy_tile(22, t, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 29, t);
        tile_regs_release();
    }
    rounded.push_back(4);
    rounded.wait_front(4);
    output.reserve_back(4);
    pack_reconfig_data_format(31);
    for (uint32_t t = 0; t < 4; ++t) {
        tile_regs_acquire();
        reconfig_data_format_srca(29);
        copy_init(29);
        copy_tile(29, t, 0);
        reconfig_data_format_srca(30);
        copy_init(30);
        copy_tile(30, t, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 31, t);
        tile_regs_release();
    }
    output.push_back(4);
    rounded.pop_front(4);
    normalized.pop_front(4);
    gate.pop_front(4);
    gamma.pop_front(4);
    CircularBuffer(27).pop_front(1);
}

void process_head() {
    flat_prepare();
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    static_assert(Kt == 4 && Vt == 4 && get_compile_time_arg_val(2) == 0);
    reconfig_data_format(q, v);
    pack_reconfig_data_format(out);
    WAIT(q, Kt);
    WAIT(k, Kt);
    WAIT(v, Vt);
    WAIT(decay, 1);
    WAIT(beta, 1);
    WAIT(eye, 1);
    WAIT(state, Kt * Vt);

    bcast_cols_mul(v, beta, vbeta, 1, Vt);
    bcast_cols_mul(k, beta, kbeta, 1, Kt);
    WAIT(kbeta, Kt);
    bcast_cols_mul_neg(kbeta, decay, nkd, 1, Kt);
    POP(kbeta, Kt);
    intra_fused(q, k, eye, intra, 1, Kt);
    bcast_cols_mul(q, decay, qdecay, 1, Kt);
    scalar_mul(k, eye, kdec, Kt);
    WAIT(kdec, Kt);
    CircularBuffer(kdec_t).reserve_back(Kt);
    pack_reconfig_data_format(kdec_t);
    reconfig_data_format_srca(kdec);
    transpose_init(kdec);
    for (uint32_t tile = 0; tile < Kt; ++tile) {
        tile_regs_acquire();
        transpose_tile(kdec, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kdec_t, tile);
        tile_regs_release();
    }
    CircularBuffer(kdec_t).push_back(Kt);
    POP(kdec, Kt);
    cpy_t(eye, 0, tinv);
    POP(q, Kt);
    POP(k, Kt);
    POP(v, Vt);
    POP(beta, 1);  // decay stays: the exact scan consumes and pops it

    // All CBs are FP32; preparation leaves both unpackers and packer in FP32.
    scan_exact_state(state, final_state);
    POP(eye, 1);
    post_norm_gate();
}

void kernel_main() {
    compute_kernel_hw_startup(20, 27, 0);
    const uint32_t iterations = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < iterations; ++i) {
        process_head();
    }
}
