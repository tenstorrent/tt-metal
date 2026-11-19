// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Experimental single-token specialization of chunk GDN's preparation + scan.
// Q/K normalization and exp(g) remain external and identical to raw_recurrent.
// Thus this isolates recurrence arithmetic, not bitwise equivalence to chunk's
// normalization or cumulative-g preparation.
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
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp"

namespace {
constexpr uint32_t q = 0, k = 1, v = 2, decay = 3, beta = 4, state = 5;
constexpr uint32_t out = 6, final_state = 7, vbeta = 8, kbeta = 9, nkd = 10;
constexpr uint32_t qdecay = 11, intra = 12, tinv = 13, eye = 14;
constexpr uint32_t vnew = 16, diff = 17, kdec = 18, kdec_t = 19, dl = 20;
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

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    static_assert(Kt == 4 && Vt == 4 && get_compile_time_arg_val(2) == 0);
    compute_kernel_hw_startup(q, v, out);
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
    scalar_mul(eye, decay, dl, 1);
    cpy_t(eye, 0, tinv);
    POP(q, Kt);
    POP(k, Kt);
    POP(v, Vt);
    POP(decay, 1);
    POP(beta, 1);

    // All CBs are FP32; preparation leaves both unpackers and packer in FP32.
    scan_step<1, Kt, Vt>(scan, state, final_state);
    POP(eye, 1);
}
