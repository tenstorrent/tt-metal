// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident MLP streamer compute (TRISC). Per layer:
//   x      = sum of the num_chips partial slots          (residual replica, cb_x)
//   h      = rmsnorm(x) * gamma[layer % weight_sets]     (cb_h)
//   g | u  = h @ [G | U] over this core's columns        (cb_gu, weights from the ring)
//   a      = silu(g) * u                                 (cb_aslice, sent to every streamer)
//   d      = act @ D over this core's columns            (act = all slices, cb_act)
//   pout   = d (+ x on chip 0, so the chips' partials sum to the new x)
//
// Runtime args: 0 ng (gate tiles of this core), 1 nd (down tiles), 2 pout_off (tile offset of the down
// columns inside the hidden vector).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/rmsnorm.h"
#include "mlp_common.hpp"

using namespace resident_mlp;

namespace {

constexpr uint32_t kDst = 8;  // tiles per DST chunk for the eltwise phases

RingCursor cursor;
uint32_t ring_base;
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

// out[n] = in0[1, K] @ W[K, n] for n_tiles columns streamed through ring entry E.
template <uint32_t E>
FORCE_INLINE void matmul(uint32_t in0_cb, uint32_t out_cb, uint32_t n_tiles, uint32_t block_tiles) {
    using En = Entry<E>;
    cb_wait_front(in0_cb, En::Kt);
    reconfig_full_operand<SrcOrder::Reverse>(in0_cb, En::cb);
    custom_mm_block_init_short<false, true, false>(in0_cb, En::cb, out_cb);
    pack_reconfig_data_format<true>(out_cb);
    pack_block_contiguous_init(out_cb);
    cb_reserve_back(out_cb, block_tiles);
    for (uint32_t n = 0; n < n_tiles; n++) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb + 1 < En::nkb; kb++) {
            with_block<E>([&] { custom_mm_block<false>(in0_cb, En::cb, kb * En::sb, 0, 0, En::sb); });
        }
        with_block<E>([&] { custom_mm_block<true>(in0_cb, En::cb, (En::nkb - 1) * En::sb, 0, 0, En::sb); });
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, out_cb, 1);
        tile_regs_release();
    }
    cb_push_back(out_cb, block_tiles);
    custom_mm_block_uninit<false>();
}

// cb_x = sum over chips of cb_slots
static_assert(num_chips == 1 || num_chips % 2 == 0, "slots are summed in pairs");
FORCE_INLINE void sum_slots() {
    cb_wait_front(cb_slots, num_chips * Ht);
    cb_reserve_back(cb_x, Ht);
    reconfig_full_operand(cb_slots, cb_slots);
    pack_init(cb_x);
    for (uint32_t t0 = 0; t0 < Ht; t0 += kDst) {
        const uint32_t n = t0 + kDst <= Ht ? kDst : Ht - t0;
        tile_regs_acquire();
        if constexpr (num_chips == 1) {
            copy_init(cb_slots);
            for (uint32_t d = 0; d < n; d++) {
                copy_tile(cb_slots, t0 + d, d);
            }
        } else {
            for (uint32_t c = 0; c < num_chips; c += 2) {
                add_init(cb_slots, cb_slots, c > 0 /* acc_to_dest */);
                for (uint32_t d = 0; d < n; d++) {
                    add_tiles(cb_slots, cb_slots, c * Ht + t0 + d, (c + 1) * Ht + t0 + d, d);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < n; d++) {
            pack_tile(d, cb_x);
        }
        tile_regs_release();
    }
    cb_push_back(cb_x, Ht);
    cb_pop_front(cb_slots, num_chips * Ht);
}

// cb_h = x / rms(x) * gamma[set], computed on the 32x32 views of x, gamma and h
FORCE_INLINE void rmsnorm(uint32_t set) {
    static_assert(Hf >= 1 && Hf <= 8 && Hf * 32 == Ht, "hidden must be 1..8 full tiles");
    cb_wait_front(cb_x, Ht);
    cb_reserve_back(cb_h, Ht);
    reconfig_full_operand(cb_x_full, cb_x_full);
    pack_reconfig_data_format<true>(cb_h_full);
    pack_block_contiguous_init(cb_h_full);
    mul_reduce_scalar_init(cb_x_full, cb_x_full);
    add_rsqrt_tile_init();
    tile_regs_acquire();
    mul_reduce_scalar_tile<PoolType::SUM>(
        cb_x_full, cb_x_full, cb_h_full, Hf, __builtin_bit_cast(float, inv_sqrt_hidden_bits));
    mul_reduce_scalar_uninit();
    add_rsqrt_tile<false, VectorMode::RC_custom, 1>(0, eps_bits);
    rmsnorm_mul_bcast_scalar_reuse_tiles_init<Hf>(cb_x_full);
    rmsnorm_mul_bcast_scalar_reuse_tiles<Hf, true>(cb_x_full, 0, 0, 0);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_full);
    for (uint32_t i = 0; i < Hf; i++) {
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_full, set * Hf + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, cb_h_full, Hf);
    tile_regs_release();
    cb_push_back(cb_h_full, Hf);  // only wraps the view's write pointer back to the base
    cb_push_back(cb_h, Ht);
    pack_reconfig_data_format<true>(cb_h);
}

// cb_aslice[i] = silu(g[i]) * u[i]
FORCE_INLINE void silu_mul(uint32_t ng) {
    cb_wait_front(cb_gu, 2 * ng_max);
    cb_reserve_back(cb_aslice, ng_max);
    reconfig_full_operand(cb_gu, cb_gu);
    copy_init(cb_gu);
    silu_tile_init();
    mul_binary_tile_init();
    pack_init(cb_aslice);
    for (uint32_t i = 0; i < ng; i++) {
        tile_regs_acquire();
        copy_tile(cb_gu, i, 0);
        copy_tile(cb_gu, ng + i, 1);
        silu_tile(0);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < 1; d++) {
            pack_tile(d, cb_aslice);
        }
        tile_regs_release();
    }
    cb_push_back(cb_aslice, ng_max);
    cb_pop_front(cb_gu, 2 * ng_max);
}

// cb_pout = cb_dout + x[pout_off : pout_off + nd]
FORCE_INLINE void add_residual(uint32_t nd, uint32_t pout_off) {
    cb_wait_front(cb_dout, nd_max);
    cb_reserve_back(cb_pout, nd_max);
    reconfig_full_operand(cb_dout, cb_x);
    add_init(cb_dout, cb_x);
    pack_init(cb_pout);
    for (uint32_t t0 = 0; t0 < nd; t0 += kDst) {
        const uint32_t n = t0 + kDst <= nd ? kDst : nd - t0;
        tile_regs_acquire();
        for (uint32_t d = 0; d < n; d++) {
            add_tiles(cb_dout, cb_x, t0 + d, pout_off + t0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < n; d++) {
            pack_tile(d, cb_pout);
        }
        tile_regs_release();
    }
    cb_push_back(cb_pout, nd_max);
    cb_pop_front(cb_dout, nd_max);
}

// debug: cb_pout = src[off : off + nd] (1: x, 2: h, 3: act) in place of the partial
constexpr uint32_t dbg = get_compile_time_arg_val(23);
FORCE_INLINE void copy_out(uint32_t src, uint32_t off, uint32_t nd) {
    cb_reserve_back(cb_pout, nd_max);
    reconfig_full_operand(src, src);
    copy_init(src);
    pack_init(cb_pout);
    for (uint32_t i = 0; i < nd; i++) {
        tile_regs_acquire();
        copy_tile(src, off + i, 0);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < 1; d++) {
            pack_tile(d, cb_pout);
        }
        tile_regs_release();
    }
    cb_push_back(cb_pout, nd_max);
}

}  // namespace

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t nd = get_arg_val<uint32_t>(1);
    const uint32_t pout_off = get_arg_val<uint32_t>(2);
#ifdef TRISC_UNPACK
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        consumed_addr_word = static_cast<uint32_t>(
            (reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(cb_consumed)) >> 2) & 0x3ffff);
    }
#endif
    custom_mm_block_init<false, true, false>(cb_h, Entry<0>::cb, cb_gu);
    custom_mm_block_uninit<false>();
    cb_wait_front(cb_gamma, weight_sets * Ht);

    for (uint32_t l = 0; l < layers; l++) {
        sum_slots();
        rmsnorm(l % weight_sets);
        matmul<0>(cb_h, cb_gu, 2 * ng, 2 * ng_max);
        silu_mul(ng);
        if constexpr (dbg != 0) {
            cb_wait_front(cb_act, It);
            if constexpr (dbg == 1) {
                copy_out(cb_x, pout_off, nd);
            } else if constexpr (dbg == 2) {
                copy_out(cb_h, pout_off, nd);
            } else {
                copy_out(cb_act, pout_off, nd);
            }
            matmul<1>(cb_act, cb_dout, nd, nd_max);
            cb_pop_front(cb_act, It);
            cb_wait_front(cb_dout, nd_max);
            cb_pop_front(cb_dout, nd_max);
            cb_pop_front(cb_h, Ht);
            cb_pop_front(cb_x, Ht);
            continue;
        }
        cb_pop_front(cb_h, Ht);
        if constexpr (chip == 0) {
            matmul<1>(cb_act, cb_dout, nd, nd_max);
            cb_pop_front(cb_act, It);
            add_residual(nd, pout_off);
        } else {
            matmul<1>(cb_act, cb_pout, nd, nd_max);
            cb_pop_front(cb_act, It);
        }
        cb_pop_front(cb_x, Ht);
    }
}
