// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter streamer compute v2 (TRISC; M6b fix 4b; d_streamer_compute.cpp is the verified v1): after the
// LM head, every logits tile's row maxima (reduce MAX over the 32 columns: column 0 of the output tile) into cb_tmax
// (CB 21, the cb_h buffer: h is consumed by then), with cb_scaler (CB 22: 1.0 in each face's first row, from the
// writer). The top-16 scans (d_topk2.hpp) skip every tile below a row's 16th tile maximum.
//
// v1, see d_streamer_common.hpp. Per layer:
//   coef  = h @ Kp_attn (this core's coefficient columns) -> cb_coef (the writer sends them to every slice)
//   q|k|v = xin @ W_qkv (this core's columns) -> cb_q (attention stub: the writer gathers the q columns as o_in)
//   partial = o_in @ W_o (this core's hidden columns) -> cb_pout
//   coef  = h2 @ Kp_mlp -> cb_coef
//   g | u = m_in @ [G | U];  a = silu(g) * u -> cb_aslice (the writer gathers it into every streamer's act)
//   partial = act @ W_down -> cb_pout
// then logits = h_final @ W_head (this core's vocab columns) -> cb_logits. Every projection is matmul_block on 32 x 32
// activation tiles at the program's fidelity (HiFi2: the BF8 weights; custom_mm is LoFi-only); the weight blocks are
// taken straight from the ring as streamer_compute.cpp (the target's) does.
//
// Runtime args: 0 ng, 1 nd, 2 nq, 3 nv, 4 nc (this core's coefficient tiles)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/experimental/pack_block.h"
#include "d_streamer_common.hpp"

using namespace dres;

namespace {

RingCursor cursor;
uint32_t ring_base;  // cb_addr_shift units
uint32_t consumed_addr_word;

template <uint32_t E, typename F>
FORCE_INLINE void with_block(F fn) {
    using En = Entry<E>;
    const uint32_t at = cursor.place<E>();
    UNPACK((get_local_cb_interface(En::cb).fifo_rd_ptr = ring_base + (at >> cb_addr_shift)));
    cb_wait_front(En::cb, En::sb);
    fn();
    cb_pop_front(En::cb, En::sb);
    UNPACK((TT_SETDMAREG(0, cursor.units & 0xffff, 0, LO_16(4))));
    UNPACK((TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK)));
    UNPACK((TT_STOREREG(4, consumed_addr_word)));
}

// out[n] = in0[32, K] @ W[K, n] for n_tiles columns streamed through ring entry E; columns from `split` on go to out_b
template <uint32_t E>
FORCE_INLINE void matmul(
    uint32_t in0_cb, uint32_t out_cb, uint32_t n_tiles, uint32_t block_tiles, uint32_t out_b = 0, uint32_t split = 0) {
    using En = Entry<E>;
    const bool two = split > 0;
    cb_wait_front(in0_cb, En::Kt);
    reconfig_full_operand<SrcOrder::Reverse>(in0_cb, En::cb);
    pack_reconfig_data_format<true>(out_cb);
    matmul_block_init(in0_cb, En::cb, 0, 1, 1, En::sb);
    pack_block_contiguous_init(out_cb);
    cb_reserve_back(out_cb, block_tiles);
    if (two) {
        cb_reserve_back(out_b, block_tiles);
    }
    for (uint32_t n = 0; n < n_tiles; n++) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < En::nkb; kb++) {
            with_block<E>([&] {
                for (uint32_t k = 0; k < En::sb; k++) {
                    matmul_block(in0_cb, En::cb, kb * En::sb + k, k, 0, 0, 1, 1, En::sb);
                }
            });
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, two && n >= split ? out_b : out_cb, 1);
        tile_regs_release();
    }
    cb_push_back(out_cb, block_tiles);
    if (two) {
        cb_push_back(out_b, block_tiles);
    }
}

// cb_aslice = silu(g) * u over this core's ng gate | up tiles, 4 tiles per DST pass (g in 0..3, u in 4..7)
FORCE_INLINE void silu_mul(uint32_t ng) {
    cb_wait_front(cb_g, kBlk);
    cb_wait_front(cb_u, kBlk);
    cb_reserve_back(cb_aslice, kBlk);
    reconfig_full_operand(cb_g, cb_u);
    pack_reconfig_data_format<true>(cb_aslice);
    pack_init(cb_aslice);
    for (uint32_t v0 = 0; v0 < ng; v0 += 4) {
        const uint32_t W = ng - v0 < 4 ? ng - v0 : 4;
        tile_regs_acquire();
        copy_init(cb_g);
        for (uint32_t i = 0; i < W; i++) {
            copy_tile(cb_g, v0 + i, i);
            copy_tile(cb_u, v0 + i, 4 + i);
        }
        silu_tile_init();
        for (uint32_t i = 0; i < W; i++) {
            silu_tile(i);
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < W; i++) {
            mul_binary_tile(i, 4 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < W; i++) {
            pack_tile<true>(i, cb_aslice, v0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_aslice, kBlk);
    cb_pop_front(cb_g, kBlk);
    cb_pop_front(cb_u, kBlk);
}

}  // namespace

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t nd = get_arg_val<uint32_t>(1);
    const uint32_t nq = get_arg_val<uint32_t>(2);
    const uint32_t nv = get_arg_val<uint32_t>(3);
    const uint32_t nc = get_arg_val<uint32_t>(4);
#ifdef TRISC_UNPACK
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        consumed_addr_word = static_cast<uint32_t>(
            (reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(cb_consumed)) >> 2) & 0x3ffff);
    }
#endif
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_h, cb_w0, cb_q);

    for (uint32_t l = 0; l < layers; l++) {
        // attention half: coefficients of h, then q|k|v of xin, then o_proj of the gathered o_in
        matmul<kKpA>(cb_h, cb_coef, nc, kBlkC);
        cb_pop_front(cb_h, Ht);
        matmul<kQkv>(cb_h, cb_q, nq, kBlk);
        cb_pop_front(cb_h, Ht);
        matmul<kO>(cb_o_in, cb_pout, nd, nd_max);
        cb_pop_front(cb_o_in, kAct);
        // mlp half
        matmul<kKpM>(cb_h, cb_coef, nc, kBlkC);
        cb_pop_front(cb_h, Ht);
        matmul<kGU>(cb_h, cb_g, 2 * ng, kBlk, cb_u, ng);
        cb_pop_front(cb_h, Ht);
        silu_mul(ng);
        matmul<kDown>(cb_act, cb_pout, nd, nd_max);
        cb_pop_front(cb_act, kAct);
    }
    matmul<kHead>(cb_h, cb_logits, nv, nv_max);
    cb_pop_front(cb_h, Ht);

    // fix 4b: the logits tiles' row maxima
    constexpr uint32_t cb_tmax = 21, cb_scaler = 22;
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_logits, nv_max);
    cb_reserve_back(cb_tmax, nv_max);
    reconfig_data_format(cb_logits, cb_scaler);
    pack_reconfig_data_format(cb_tmax);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_logits, cb_scaler, cb_tmax);
    for (uint32_t t0 = 0; t0 < nv; t0 += 8) {
        const uint32_t W = nv - t0 < 8 ? nv - t0 : 8;
        tile_regs_acquire();
        for (uint32_t i = 0; i < W; i++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_logits, cb_scaler, t0 + i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < W; i++) {
            pack_tile<true>(i, cb_tmax, t0 + i);
        }
        tile_regs_release();
    }
    reduce_uninit();
    cb_push_back(cb_tmax, nv_max);
}
