// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident streamer compute (TRISC), see streamer_common.hpp. Per layer:
//   y = h @ W_qkv (this core's columns) -> cb_q (the writer gathers the q columns into every streamer's o_in)
//   partial = o_in @ W_o (this core's hidden columns) -> cb_pout
//   g | u = h2 @ [G | U];  a = gelu_tanh(g) * u -> cb_aslice (the writer gathers it into every streamer's act)
//   partial = act @ W_down -> cb_pout
// then logits = h_final @ W_head (this core's vocab columns) -> cb_logits.
// The weight blocks are taken straight from the ring (with_block points the entry's CB at the block and publishes
// the consumed units for the reader). Projections: R <= 8 rows custom_mm on R x 32 tiles (LoFi, which keeps BF4 weights
// exact and is the ttnn decode projections' fidelity); R = 32 matmul on 32 x 32 tiles at the program's fidelity. The BF8
// LM head always runs matmul_block (R x 32 tiles at R <= 8) at the program's fidelity, HiFi2 as the ttnn head: LoFi
// keeps only 5 bits of a BF8 weight (2.1 % logits error). Adapted from changh95/qwen3.8-27b-p300x2 code/qwen36_resident/kernels/streamer_compute.cpp
// (Apache-2.0); the norms and the residual live on the slice cores here.
//
// Runtime args: 0 ng, 1 nd, 2 nq sliding, 3 nq full, 4 nv

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "streamer_common.hpp"

using namespace gres;

namespace {

constexpr auto kLayout = CustomMmDestLayout::DefaultSplitAcc;

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
    // the reader's room: the consumed units, stored once the unpacker is done with the block (as cb_pop_front does)
    UNPACK((TT_SETDMAREG(0, cursor.units & 0xffff, 0, LO_16(4))));
    UNPACK((TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK)));
    UNPACK((TT_STOREREG(4, consumed_addr_word)));
}

// out[n] = in0[R, K] @ W[K, n] for n_tiles columns streamed through ring entry E; columns from `split` on go to
// out_b (gate | up); each output CB takes one block of block_tiles.
template <uint32_t E>
FORCE_INLINE void matmul(
    uint32_t in0_cb, uint32_t out_cb, uint32_t n_tiles, uint32_t block_tiles, uint32_t out_b = 0, uint32_t split = 0) {
    using En = Entry<E>;
    constexpr bool kCustom = kCustomMm && E != kHead;
    const bool two = split > 0;
    cb_wait_front(in0_cb, En::Kt);
    reconfig_full_operand<SrcOrder::Reverse>(in0_cb, En::cb);
    pack_reconfig_data_format<true>(out_cb);
    if constexpr (kCustom) {
        custom_mm_block_init_short<false, kLayout>(in0_cb, En::cb, out_cb);
    } else {
        matmul_block_init(in0_cb, En::cb, 0, 1, 1, En::sb);
    }
    pack_block_contiguous_init(out_cb);
    cb_reserve_back(out_cb, block_tiles);
    if (two) {
        cb_reserve_back(out_b, block_tiles);
    }
    for (uint32_t n = 0; n < n_tiles; n++) {
        tile_regs_acquire();
        if constexpr (kCustom) {
            for (uint32_t kb = 0; kb + 1 < En::nkb; kb++) {
                with_block<E>([&] { custom_mm_block<kLayout, false>(in0_cb, En::cb, kb * En::sb, 0, 0, En::sb); });
            }
            with_block<E>([&] { custom_mm_block<kLayout>(in0_cb, En::cb, (En::nkb - 1) * En::sb, 0, 0, En::sb); });
        } else {
            for (uint32_t kb = 0; kb < En::nkb; kb++) {
                with_block<E>([&] {
                    for (uint32_t k = 0; k < En::sb; k++) {
                        matmul_block(in0_cb, En::cb, kb * En::sb + k, k, 0, 0, 1, 1, En::sb);
                    }
                });
            }
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
    if constexpr (kCustom) {
        custom_mm_block_uninit<kLayout>(out_cb);
    }
}

// cb_aslice = gelu_tanh(g) * u over the 32x32 views holding this core's ng gate | up tiles, up to 4 views per DST
// pass (g in slots 0..3, u in 4..7)
FORCE_INLINE void gelu_mul(uint32_t ng) {
    cb_wait_front(cb_g, kBlk);
    cb_wait_front(cb_u, kBlk);
    cb_reserve_back(cb_aslice, kBlk);
    reconfig_full_operand(cb_g_full, cb_u_full);
    pack_reconfig_data_format<true>(cb_aslice_full);
    pack_init(cb_aslice_full);
    const uint32_t views = views_of(ng);
    for (uint32_t v0 = 0; v0 < views; v0 += 4) {
        const uint32_t W = views - v0 < 4 ? views - v0 : 4;
        tile_regs_acquire();
        copy_init(cb_g_full);
        for (uint32_t i = 0; i < W; i++) {
            copy_tile(cb_g_full, v0 + i, i);
            copy_tile(cb_u_full, v0 + i, 4 + i);
        }
        gelu_tanh_tile_init();
        for (uint32_t i = 0; i < W; i++) {
            gelu_tanh_tile(i);
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < W; i++) {
            mul_binary_tile(i, 4 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < W; i++) {
            pack_tile<true>(i, cb_aslice_full, v0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_aslice_full, kBlk / kViewTiles);  // wraps the views' write pointer back to the base
    cb_push_back(cb_aslice, kBlk);
    cb_pop_front(cb_g, kBlk);
    cb_pop_front(cb_u, kBlk);
}

}  // namespace

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t nd = get_arg_val<uint32_t>(1);
    const uint32_t nq_s = get_arg_val<uint32_t>(2);
    const uint32_t nq_f = get_arg_val<uint32_t>(3);
    const uint32_t nv = get_arg_val<uint32_t>(4);
#ifdef TRISC_UNPACK
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        consumed_addr_word = static_cast<uint32_t>(
            (reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(cb_consumed)) >> 2) & 0x3ffff);
    }
#endif
    if constexpr (kCustomMm) {
        custom_mm_block_init<false, kLayout>(cb_h, cb_w0, cb_q);
        custom_mm_block_uninit<kLayout>(cb_q);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_h, cb_w0, cb_q);
    }

    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        // mixer half (attention stubbed: o = q, gathered by the writers)
        if (full) {
            matmul<kQkvF>(cb_h, cb_q, nq_f, kBlk);
        } else {
            matmul<kQkvS>(cb_h, cb_q, nq_s, kBlk);
        }
        cb_pop_front(cb_h, Ht);
        if (full) {
            matmul<kOF>(cb_o_in, cb_pout, nd, nd_max);
        } else {
            matmul<kOS>(cb_o_in, cb_pout, nd, nd_max);
        }
        cb_pop_front(cb_o_in, kAct);
        // mlp half
        matmul<kGU>(cb_h, cb_g, 2 * ng, kBlk, cb_u, ng);
        cb_pop_front(cb_h, Ht);
        gelu_mul(ng);
        matmul<kDown>(cb_act, cb_pout, nd, nd_max);
        cb_pop_front(cb_act, kAct);
    }
    matmul<kHead>(cb_h, cb_logits, nv, nv_max);
    cb_pop_front(cb_h, Ht);
}
