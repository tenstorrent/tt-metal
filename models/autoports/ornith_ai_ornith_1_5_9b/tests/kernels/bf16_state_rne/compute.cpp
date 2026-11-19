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
#include "norm_stages.hpp"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp"

#include "api/compute/eltwise_unary/typecast.h"
#include "scan_state_bf16.hpp"

namespace {
constexpr uint32_t q = 0, k = 1, v = 2, decay = 3, beta = 4, state = 5;
constexpr uint32_t out = 6, final_state = 7, vbeta = 8, kbeta = 9, nkd = 10;
constexpr uint32_t qdecay = 11, intra = 12, tinv = 13, eye = 14;
constexpr uint32_t vnew = 16, diff = 17, kdec = 18, kdec_t = 19, dl = 15;
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

// Normalize Q/K and prepare scalar gates in the same launch as recurrence.
void flat_prepare() {
    CircularBuffer rq(20),rk(21),rv(22),rg(23),rb(24);
    rq.wait_front(4);rk.wait_front(4);rv.wait_front(4);rg.wait_front(1);rb.wait_front(1);
    flat_norm::row_rsqrt<25,26,27,4,0x3f800000u,0x358637bdu>(20);
    flat_norm::normalize<26,4>(20,27,false,0);
    flat_norm::row_rsqrt<25,26,27,4,0x3f800000u,0x358637bdu>(21);
    flat_norm::normalize<26,4>(21,27,false,1);
    for(uint32_t which=0;which<3;++which){
        const uint32_t input=which==0?22:(which==1?23:24);
        const uint32_t output=which==0?2:(which==1?3:4);
        const uint32_t count=which==0?4:1;
        CircularBuffer dest(output);dest.reserve_back(count);
        pack_reconfig_data_format(output);
        for(uint32_t t=0;t<count;++t){
            tile_regs_acquire();reconfig_data_format_srca(input);copy_init(input);copy_tile(input,t,0);
            if(which==1){exp_tile_init();exp_tile(0);}
            tile_regs_commit();tile_regs_wait();pack_tile(0,output,t);tile_regs_release();
        }
        dest.push_back(count);
    }
    rq.pop_front(4);rk.pop_front(4);rv.pop_front(4);rg.pop_front(1);rb.pop_front(1);
    // Keep the unit scaler for output RMS normalization.
}


// Preserve the BF16 recurrence-output pack and BF16 RMSNorm pack before gating.
void post_norm_gate() {
    constexpr uint32_t epsilon=get_compile_time_arg_val(3);
    CircularBuffer source(6),rounded(29),normalized(30),gamma(20),gate(22),output(31);
    source.wait_front(4);rounded.reserve_back(4);
    pack_reconfig_data_format(29);
    for(uint32_t t=0;t<4;++t) {
        tile_regs_acquire();reconfig_data_format_srca(6);copy_init(6);copy_tile(6,t,0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,29,t);tile_regs_release();
    }
    rounded.push_back(4);source.pop_front(4);rounded.wait_front(4);
    flat_norm::row_rsqrt<25,26,27,4,0x3c000000u,epsilon>(29);
    flat_norm::normalize<26,4>(29,20,true,30);
    rounded.pop_front(4);normalized.wait_front(4);gate.wait_front(4);
    // binary_ng MUL packs its fused SiLU activation to BF16 before multiply.
    // Reuse the compute-owned rounded CB only after its previous pages are popped.
    rounded.reserve_back(4);pack_reconfig_data_format(29);
    for(uint32_t t=0;t<4;++t) {
        tile_regs_acquire();reconfig_data_format_srca(22);copy_init(22);copy_tile(22,t,0);
        silu_tile_init();silu_tile(0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,29,t);tile_regs_release();
    }
    rounded.push_back(4);rounded.wait_front(4);
    output.reserve_back(4);pack_reconfig_data_format(31);
    for(uint32_t t=0;t<4;++t) {
        tile_regs_acquire();reconfig_data_format_srca(29);copy_init(29);copy_tile(29,t,0);
        reconfig_data_format_srca(30);copy_init(30);copy_tile(30,t,1);
        mul_binary_tile_init();mul_binary_tile(0,1,0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,31,t);tile_regs_release();
    }
    output.push_back(4);rounded.pop_front(4);normalized.pop_front(4);gate.pop_front(4);gamma.pop_front(4);
    CircularBuffer(27).pop_front(1);
}

void process_head() {
    // The shared scan hoists all format reconfigurations assuming FP32 inputs.
    // Stage BF16 persistent state explicitly to its expected FP32 CB5 first.
    CircularBuffer incoming(28),expanded(5);
    incoming.wait_front(16);expanded.reserve_back(16);
    pack_reconfig_data_format(5);
    reconfig_data_format_srca(28);copy_init(28);
    for(uint32_t t=0;t<16;++t) {
        tile_regs_acquire();copy_tile(28,t,0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,5,t);tile_regs_release();
    }
    expanded.push_back(16);incoming.pop_front(16);
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
    scalar_mul(eye, decay, dl, 1);
    cpy_t(eye, 0, tinv);
    POP(q, Kt);
    POP(k, Kt);
    POP(v, Vt);
    POP(decay, 1);
    POP(beta, 1);

    // All CBs are FP32; preparation leaves both unpackers and packer in FP32.
    scan_step_state_bf16<1, Kt, Vt>(scan, state, final_state);
    POP(eye, 1);
    post_norm_gate();
}

void kernel_main() {
    compute_kernel_hw_startup(20,27,0);
    const uint32_t iterations=get_arg_val<uint32_t>(0);
    for(uint32_t i=0;i<iterations;++i) process_head();
}
