// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident streamer compute (TRISC), see streamer_common.hpp. Per layer:
//   mixer half:
//     h = rmsnorm(x) (x: from the hub)
//     GDN: y = h @ W_qkvzab (this core's columns);  q|k|v columns: y = silu(conv1d_4(y)) over the
//          history streamed in (oldest first), y itself goes out as the newest history entry;
//          z, a, b columns pass through
//     attention: y = h @ W_qkvg, passed through
//     d = o @ W_out (this core's columns, o = the mixer cores' output);  pout = d (+ x on chip 0)
//   mlp half:
//     h = rmsnorm(x);  a = silu(h @ G) * (h @ U)
//     d = act @ D;  pout = d (+ x on chip 0)
// then optionally logits = rmsnorm(x) @ W_head (this core's columns).
//
// Runtime args: 0 ng, 1 nd, 2 pout_off, 3 GDN projection tiles, 4 attention projection tiles, 5 n_conv
// (leading q|k|v columns among this core's GDN projection tiles), 6 lm_head tiles

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
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "streamer_common.hpp"

using namespace resident;

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

// out[n] = in0[1, K] @ W[K, n] for n_tiles columns streamed through ring entry E; columns from
// `split` on go to out_b (gate | up), each output CB takes one block of block_tiles.
template <uint32_t E>
FORCE_INLINE void matmul(
    uint32_t in0_cb, uint32_t out_cb, uint32_t n_tiles, uint32_t block_tiles, uint32_t out_b = 0, uint32_t split = 0) {
    using En = Entry<E>;
    const bool two = split > 0;
    cb_wait_front(in0_cb, En::Kt);
    reconfig_full_operand<SrcOrder::Reverse>(in0_cb, En::cb);
    custom_mm_block_init_short<false, true, false>(in0_cb, En::cb, out_cb);
    pack_reconfig_data_format<true>(out_cb);
    pack_block_contiguous_init(out_cb);
    cb_reserve_back(out_cb, block_tiles);
    if (two) {
        cb_reserve_back(out_b, block_tiles);
    }
    for (uint32_t n = 0; n < n_tiles; n++) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb + 1 < En::nkb; kb++) {
            with_block<E>([&] { custom_mm_block<false>(in0_cb, En::cb, kb * En::sb, 0, 0, En::sb); });
        }
        with_block<E>([&] { custom_mm_block<true>(in0_cb, En::cb, (En::nkb - 1) * En::sb, 0, 0, En::sb); });
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, two && n >= split ? out_b : out_cb, 1);
        tile_regs_release();
    }
    cb_push_back(out_cb, block_tiles);
    if (two) {
        cb_push_back(out_b, block_tiles);
    }
    custom_mm_block_uninit<false>();
}

// cb_h = x / rms(x) per user row, batch > 1, on the FPU. Row R of every 32x32 view of x holds elements of
// user R % batch only (a batch x 32 tile's face rows are 16 elements of one user, and 16 / batch face rows
// repeat the users in order), so: D = sum over the views of x x^T (exact bf16 products, fp32 sums) has the
// view rows' sums of squares on its diagonal; masked by the identity, times ones, and folded by F (ones
// where R = R' mod batch) every element of row R holds the sum of squares of user R % batch; then
// h = x * rsqrt(sum / hidden + eps).
FORCE_INLINE void rmsnorm_rows() {
    constexpr uint32_t V = Hf * batch;  // 32x32 views of x
    cb_wait_front(cb_x, Ht);
    cb_reserve_back(cb_h, Ht);
    cb_reserve_back(cb_sumsq, 2);
    reconfig_full_operand<SrcOrder::Reverse>(cb_x_full, cb_x_full);
    matmul_init(cb_x_full, cb_x_full, 1 /* transpose */);
    pack_reconfig_data_format(cb_sumsq);
    pack_init(cb_sumsq);
    tile_regs_acquire();
    for (uint32_t v = 0; v < V; v++) {
        matmul_tiles(cb_x_full, cb_x_full, v, v, 0);
    }
    reconfig_full_operand_srca(cb_eye);
    copy_init(cb_eye);
    copy_tile(cb_eye, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sumsq);
    tile_regs_release();
    cb_push_back(cb_sumsq, 1);

    cb_wait_front(cb_sumsq, 1);
    reconfig_full_operand<SrcOrder::Reverse>(cb_sumsq, cb_ones_full);
    matmul_init(cb_sumsq, cb_ones_full);
    tile_regs_acquire();
    matmul_tiles(cb_sumsq, cb_ones_full, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sumsq);
    tile_regs_release();
    cb_push_back(cb_sumsq, 1);
    cb_wait_front(cb_sumsq, 2);
    cb_reserve_back(cb_rinv, 1);
    reconfig_full_operand<SrcOrder::Reverse>(cb_fold, cb_sumsq);
    matmul_init(cb_fold, cb_sumsq);
    pack_reconfig_data_format(cb_rinv);
    tile_regs_acquire();
    matmul_tiles(cb_fold, cb_sumsq, 0, 1, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(
        0,
        __builtin_bit_cast(
            uint32_t,
            __builtin_bit_cast(float, inv_sqrt_hidden_bits) * __builtin_bit_cast(float, inv_sqrt_hidden_bits)));
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rinv);
    tile_regs_release();
    cb_push_back(cb_rinv, 1);
    cb_pop_front(cb_sumsq, 2);

    // h = x * r, the same 1 / rms tile for every view
    cb_wait_front(cb_rinv, 1);
    reconfig_full_operand(cb_x_full, cb_rinv);
    mul_init(cb_x_full, cb_rinv);
    pack_reconfig_data_format<true>(cb_h_full);
    pack_init(cb_h_full);
    for (uint32_t v0 = 0; v0 < V; v0 += kDst) {
        const uint32_t n = v0 + kDst <= V ? kDst : V - v0;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; i++) {
            mul_tiles(cb_x_full, cb_rinv, v0 + i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; i++) {
            pack_tile<true>(i, cb_h_full, v0 + i);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_rinv, 1);
    cb_push_back(cb_h_full, V);  // only wraps the view's write pointer back to the base
    cb_push_back(cb_h, Ht);
    pack_reconfig_data_format<true>(cb_h);
}

// cb_h = x / rms(x), computed on the 32x32 views of x and h (the norm weight is folded into the weights)
FORCE_INLINE void rmsnorm() {
    static_assert(Hf >= 1 && Hf <= 8 && Hf * 32 == Ht, "hidden must be 1..8 full tiles");
    if constexpr (batch > 1) {
        rmsnorm_rows();
        return;
    }
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
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, cb_h_full, Hf);
    tile_regs_release();
    cb_push_back(cb_h_full, Hf);  // only wraps the view's write pointer back to the base
    cb_push_back(cb_h, Ht);
    pack_reconfig_data_format<true>(cb_h);
}

// cb_aslice[i] = silu(g[i]) * u[i] for the ng gate | up tiles of this core
FORCE_INLINE void silu_mul(uint32_t ng) {
    cb_wait_front(cb_g, kBlk);
    cb_wait_front(cb_u, kBlk);
    cb_reserve_back(cb_aslice, kBlk);
    reconfig_full_operand(cb_g_full, cb_u_full);
    pack_reconfig_data_format<true>(cb_aslice_full);
    pack_init(cb_aslice_full);
    // up to 4 views per DST pass (g in slots 0..3, u in 4..7), one init per op
    const uint32_t views = views_of(ng);
    for (uint32_t v0 = 0; v0 < views; v0 += 4) {
        const uint32_t W = views - v0 < 4 ? views - v0 : 4;
        tile_regs_acquire();
        copy_init(cb_g_full);
        for (uint32_t i = 0; i < W; i++) {
            copy_tile(cb_g_full, v0 + i, i);
            copy_tile(cb_u_full, v0 + i, 4 + i);
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
            pack_tile<true>(i, cb_aslice_full, v0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_aslice_full, kBlkViews);  // wraps the view's write pointer back to the base
    cb_push_back(cb_aslice, kBlk);
    cb_pop_front(cb_g, kBlk);
    cb_pop_front(cb_u, kBlk);
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

// q|k|v columns: cb_qkvz_out = silu(w0 * y[t-3] + w1 * y[t-2] + w2 * y[t-1] + w3 * y[t]) over the
// streamed-in history (cb_conv_hist, oldest first) and taps; y[t] goes to cb_hist_out; the first n_conv
// of this core's columns are q|k|v, the rest (z, a, b) are copied through (attention: all of them).
// The conv runs once on the 32x32 views of the column block.
FORCE_INLINE void conv_pass(uint32_t nq, uint32_t n_conv, bool gdn) {
    cb_wait_front(cb_qkvz, kBlk);
    cb_reserve_back(cb_qkvz_out, kBlk);
    if (gdn) {
        cb_wait_front(cb_conv_w, 4 * kBlk);
        cb_wait_front(cb_conv_hist, 3 * kBlk);
        cb_reserve_back(cb_hist_out, (verify ? 3 : 1) * kBlkViews);
    }
    if (n_conv > 0) {
        reconfig_full_operand(cb_qkvz_full, cb_qkvz_full);
        pack_reconfig_data_format<true>(cb_qkvz_out_full);
        pack_init(cb_qkvz_out_full);
        // view v of the block: taps and history slots j hold their own view v at j * kBlkViews + v; all eight
        // operands of a view go to DST first (slot 0 keeps y for the history), one init per op
        for (uint32_t v = 0; v < views_of(n_conv); v++) {
            tile_regs_acquire();
            if constexpr (verify) {
                // newest history slot: its row 0, and row 0 of y as row 1 (exact 0/1 products)
                reconfig_full_operand<SrcOrder::Reverse>(cb_shift, cb_conv_hist_full);
                matmul_init(cb_shift, cb_conv_hist_full);
                matmul_tiles(cb_shift, cb_conv_hist_full, 0, 2 * kBlkViews + v, 6);
                matmul_tiles(cb_shift, cb_qkvz_full, 1, v, 6);
                reconfig_full_operand(cb_qkvz_full, cb_qkvz_full);
            }
            copy_init(cb_qkvz_full);
            copy_tile(cb_qkvz_full, v, 0);
            copy_tile(cb_conv_w_full, 3 * kBlkViews + v, 1);
            for (uint32_t j = 0; j < 3; j++) {
                if (!verify || j < 2) {
                    copy_tile(cb_conv_hist_full, j * kBlkViews + v, 2 + 2 * j);
                }
                copy_tile(cb_conv_w_full, j * kBlkViews + v, 3 + 2 * j);
            }
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 1);
            for (uint32_t j = 0; j < 3; j++) {
                mul_binary_tile(2 + 2 * j, 3 + 2 * j, 3 + 2 * j);
            }
            add_binary_tile_init();
            for (uint32_t j = 0; j < 3; j++) {
                add_binary_tile(1, 3 + 2 * j, 1);
            }
            silu_tile_init();
            silu_tile(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(1, cb_qkvz_out_full, v);
            if constexpr (verify) {
                pack_tile<true>(6, cb_hist_out, v);
                pack_tile<true>(0, cb_hist_out, kBlkViews + v);
            } else {
                pack_tile<true>(0, cb_hist_out, v);
            }
            tile_regs_release();
            if constexpr (verify) {
                // (y1, -) for the slot after y
                tile_regs_acquire();
                reconfig_full_operand<SrcOrder::Reverse>(cb_shift, cb_qkvz_full);
                matmul_init(cb_shift, cb_qkvz_full);
                matmul_tiles(cb_shift, cb_qkvz_full, 2, v, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_hist_out, 2 * kBlkViews + v);
                tile_regs_release();
                reconfig_full_operand(cb_qkvz_full, cb_qkvz_full);
            }
        }
        cb_push_back(cb_qkvz_out_full, kBlkViews);  // wraps the view's write pointer back to the base
    }
    if (n_conv < nq) {
        reconfig_full_operand(cb_qkvz, cb_qkvz);
        pack_reconfig_data_format<true>(cb_qkvz_out);
        pack_init(cb_qkvz_out);
        copy_init(cb_qkvz);
        for (uint32_t i = n_conv; i < nq; i++) {
            tile_regs_acquire();
            copy_tile(cb_qkvz, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_qkvz_out, i);
            tile_regs_release();
        }
    }
    pack_reconfig_data_format<true>(cb_qkvz_out);
    cb_push_back(cb_qkvz_out, kBlk);
    cb_pop_front(cb_qkvz, kBlk);
    if (gdn) {
        cb_push_back(cb_hist_out, (verify ? 3 : 1) * kBlkViews);  // with no q|k|v columns the writer skips it
        cb_pop_front(cb_conv_w, 4 * kBlk);
        cb_pop_front(cb_conv_hist, 3 * kBlk);
    }
}

// d = in0 @ W_E over this core's down-type columns; pout = d (+ x on chip 0)
template <uint32_t E>
FORCE_INLINE void project_to_partial(uint32_t in0_cb, uint32_t in0_tiles, uint32_t nd, uint32_t pout_off) {
    if constexpr (chip == 0) {
        matmul<E>(in0_cb, cb_dout, nd, nd_max);
        cb_pop_front(in0_cb, in0_tiles);
        add_residual(nd, pout_off);
    } else {
        matmul<E>(in0_cb, cb_pout, nd, nd_max);
        cb_pop_front(in0_cb, in0_tiles);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t nd = get_arg_val<uint32_t>(1);
    const uint32_t pout_off = get_arg_val<uint32_t>(2);
    const uint32_t nq_gdn = get_arg_val<uint32_t>(3);
    const uint32_t nq_attn = get_arg_val<uint32_t>(4);
    const uint32_t n_conv = get_arg_val<uint32_t>(5);
    const uint32_t nv = get_arg_val<uint32_t>(6);
    const uint32_t ts_addr = get_arg_val<uint32_t>(7);
    // the pack thread stamps the end of each phase's packing
    auto mark = [&](uint32_t l, uint32_t i) {
        PACK(({
            if (ts_addr) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr)[l * kTsWords + 16 + i] =
                    *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
            }
        }));
    };
#ifdef TRISC_UNPACK
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = w.fifo_limit - w.fifo_size;
        consumed_addr_word = static_cast<uint32_t>(
            (reinterpret_cast<std::uintptr_t>(get_cb_tiles_acked_ptr(cb_consumed)) >> 2) & 0x3ffff);
    }
#endif
    custom_mm_block_init<false, true, false>(cb_h, Entry<0>::cb, cb_qkvz);
    custom_mm_block_uninit<false>();
    if constexpr (batch > 1) {
        cb_wait_front(cb_ones_full, 1);
        cb_wait_front(cb_fold, 1);
        cb_wait_front(cb_eye, 1);
    }
    if constexpr (verify) {
        cb_wait_front(cb_shift, 3);
    }

    if constexpr (fc_in) {
        // this chip's K slice of the fc (see streamer_common.hpp); no residual
        constexpr uint32_t skip = (chip % (num_chips / 2)) * (2 * Ht / num_chips);
        rmsnorm();
        if (skip > 0) {
            cb_wait_front(cb_h, Ht);
            cb_pop_front(cb_h, skip);
        }
        matmul<kQkvz>(cb_h, cb_pout, nd, nd_max);
        cb_pop_front(cb_h, Ht - skip);
        cb_pop_front(cb_x, Ht);
    }
    for (uint32_t l = 0; l < layers; l++) {
        const bool attn = is_attn(l);
        // mixer half
        mark(l, 0);
        rmsnorm();
        mark(l, 1);
        if (attn) {
            matmul<kQkvg>(cb_h, cb_qkvz, nq_attn, kBlk);
        } else {
            matmul<kQkvz>(cb_h, cb_qkvz, nq_gdn, kBlk);
        }
        cb_pop_front(cb_h, Ht);
        mark(l, 2);
        conv_pass(attn ? nq_attn : nq_gdn, attn ? 0 : n_conv, !attn);
        mark(l, 3);
        project_to_partial<kOut>(cb_o_in, Ot, nd, pout_off);
        cb_pop_front(cb_x, Ht);
        mark(l, 4);
        // mlp half
        mark(l, 5);
        rmsnorm();
        mark(l, 6);
        matmul<kGateUp>(cb_h, cb_g, 2 * ng, kBlk, cb_u, ng);
        cb_pop_front(cb_h, Ht);
        mark(l, 7);
        silu_mul(ng);
        mark(l, 8);
        project_to_partial<kDown>(cb_act, It, nd, pout_off);
        cb_pop_front(cb_x, Ht);
        mark(l, 9);
    }
    if constexpr (lm_head) {
        rmsnorm();
        matmul<kHead>(cb_h, cb_logits, nv, nv_max);
        cb_pop_front(cb_h, Ht);
        cb_pop_front(cb_x, Ht);
    }
}
