// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group A compute: one (column group, tile row) unit per loop in a 16-bit DEST with APPROX = false and HiFi4, the
// chain's configuration, except the two SFPU calls inside ttnn.rms_norm: the layernorm op compiles them with
// math_approx_mode = true, so they are written out through their macros with `true` in place of APPROX.
// Each phase is one chain op's LLK sequence with that op's pack: the three shifted FIR taps (exact 0/1 selection
// matmuls), the conv (multiply + three mac), silu, for q/k the rms_norm mirror and the bf16 scale (twice for q), and
// for v the row-mask multiply.
//
// CBs: in 0, sel 1, tap 2, tapfull 3, shift 4, conv 5, x 6, xmm2 7, scaler 8, eps 9, scale 10, ex2 11, ex2pe 12,
// unit 13, out 14, mask 15, maskfull 16.  cb_in is the reader's window: the previous tile row at 0..3, this
// one at 4..7.
// Compile-time args: none (the page maps live in the reader and the writer).  Runtime args: 0 units, then
// (column group, tile row) pairs.

#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/mac.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

using namespace ckernel;

namespace {
constexpr uint32_t cb_in = 0, cb_sel = 1, cb_tap = 2, cb_tapfull = 3, cb_shift = 4, cb_conv = 5, cb_x = 6;
constexpr uint32_t cb_xmm2 = 7, cb_scaler = 8, cb_eps = 9, cb_scale = 10, cb_ex2 = 11, cb_ex2pe = 12;
constexpr uint32_t cb_unit = 13, cb_out = 14, cb_mask = 15, cb_maskfull = 16;

constexpr uint32_t head_tiles = 4;  // 128 / 32: Wt of the q/k norm
constexpr uint32_t conv_taps = 4;
constexpr uint32_t history_rows = 3;  // the three shifted taps
constexpr uint32_t qk_heads = 4, qk_groups = 8;
constexpr uint32_t tap_tiles = conv_taps * head_tiles;
constexpr uint32_t window_tiles = 2 * head_tiles;  // cb_in: the previous tile row at 0..3, this one at 4..7
constexpr uint32_t select_current = 0, select_previous = 1, select_history = 2, select_kinds = 3;
constexpr uint32_t select_tiles = history_rows * select_kinds;  // nine 0/1 tiles, pushed once by the reader
constexpr uint32_t inverse_width_bits = 0x3C000000;             // 1.0f / 128: the layernorm reduce's 1/W scale
constexpr uint32_t dst0 = 0, dst1 = 1, dst2 = 2;

constexpr uint32_t select_index(uint32_t kind, uint32_t shift) { return (shift - 1) * select_kinds + kind; }

// The four tap rows of this column group broadcast to full tiles (unary_bcast<ROW>) once per group instead of once
// per output tile: the same bits, since the broadcast is pure and the bf16 -> bf16 pack is lossless.
ALWI void build_tap_tiles() {
    CircularBuffer(cb_tap).wait_front(tap_tiles);
    CircularBuffer(cb_tapfull).reserve_back(tap_tiles);
    reconfig_data_format(cb_tap, cb_tap);
    pack_reconfig_data_format(cb_tapfull);
    unary_bcast_init<BroadcastType::ROW>(cb_tap);
    for (uint32_t t = 0; t < tap_tiles; ++t) {
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_tap, t, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_tapfull);
        tile_regs_release();
    }
    CircularBuffer(cb_tapfull).push_back(tap_tiles);
    CircularBuffer(cb_tapfull).wait_front(tap_tiles);
}

// taps 0..2 of one column tile: shifted_s = Sel_prev_or_hist(s) @ previous + Sel_cur(s) @ current, one DEST tile
// per tap (tap 0 is the shift by 3, tap 2 the shift by 1), packed bf16.
ALWI void shifted_taps(uint32_t chunk, uint32_t column_tile) {
    reconfig_data_format<SrcOrder::Reverse>(cb_sel, cb_in);
    matmul_init(cb_sel, cb_in);
    pack_reconfig_data_format(cb_shift);
    CircularBuffer(cb_shift).reserve_back(history_rows);
    for (uint32_t tap = 0; tap < history_rows; ++tap) {
        const uint32_t shift = history_rows - tap;
        const uint32_t before = select_index(chunk == 0 ? select_history : select_previous, shift);
        tile_regs_acquire();
        matmul_tiles(cb_sel, cb_in, before, column_tile, dst0);
        matmul_tiles(cb_sel, cb_in, select_index(select_current, shift), head_tiles + column_tile, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_shift);
        tile_regs_release();
    }
    CircularBuffer(cb_shift).push_back(history_rows);
    CircularBuffer(cb_shift).wait_front(history_rows);
}

// conv = silu(mul(tap0, w0) then mac(tap_t, w_t, conv) for t = 1..3) on one column tile; one bf16 pack per op.
ALWI void conv_silu(uint32_t column_tile, uint32_t silu_cb) {
    pack_reconfig_data_format(cb_conv);
    CircularBuffer(cb_conv).reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_shift);
    copy_init(cb_shift);
    copy_tile(cb_shift, 0, dst0);
    reconfig_data_format_srca(cb_tapfull);
    copy_init(cb_tapfull);
    copy_tile(cb_tapfull, head_tiles * column_tile + 0, dst1);
    mul_binary_tile_init();
    mul_binary_tile(dst0, dst1, dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_conv);
    tile_regs_release();
    CircularBuffer(cb_conv).push_back(1);

    for (uint32_t t = 1; t < conv_taps; ++t) {
        CircularBuffer(cb_conv).wait_front(1);
        CircularBuffer(cb_conv).reserve_back(1);
        tile_regs_acquire();
        if (t < history_rows) {
            reconfig_data_format_srca(cb_shift);
            copy_init(cb_shift);
            copy_tile(cb_shift, t, dst0);
        } else {  // tap 3 is the rows themselves: the current tile row, already in cb_in
            reconfig_data_format_srca(cb_in);
            copy_init(cb_in);
            copy_tile(cb_in, head_tiles + column_tile, dst0);
        }
        reconfig_data_format_srca(cb_tapfull);
        copy_init(cb_tapfull);
        copy_tile(cb_tapfull, head_tiles * column_tile + t, dst1);
        reconfig_data_format_srca(cb_conv);
        copy_init(cb_conv);
        copy_tile(cb_conv, 0, dst2);
        // mac_tile is only valid immediately after mac_tile_init (replay slots 0..6): no SFPU op in between.
        mac_tile_init<DataFormat::Float16_b>();
        mac_tile<DataFormat::Float16_b>(0, 1, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_conv);
        tile_regs_release();
        CircularBuffer(cb_conv).push_back(1);
        CircularBuffer(cb_conv).pop_front(1);
    }

    CircularBuffer(cb_conv).wait_front(1);
    pack_reconfig_data_format(silu_cb);
    CircularBuffer(silu_cb).reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_conv);
    copy_init(cb_conv);
    copy_tile(cb_conv, 0, dst0);
    silu_tile_init();
    silu_tile(dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, silu_cb);
    tile_regs_release();
    CircularBuffer(silu_cb).push_back(1);
    CircularBuffer(cb_conv).pop_front(1);
}

// ttnn.rms_norm(x bf16, epsilon = 1e-6 / 128) with no weight: layernorm.cpp's RMSNORM path at Wt = 4, every
// intermediate a bf16 CB in the 16-bit DEST.  cb_x -> cb_unit.
ALWI void rms_norm_unit() {
    CircularBuffer(cb_x).wait_front(head_tiles);

    // x * x -> xmm2 (FPU mul_tiles, one block)
    reconfig_data_format(cb_x, cb_x);
    pack_reconfig_data_format(cb_xmm2);
    mul_init(cb_x, cb_x);
    CircularBuffer(cb_xmm2).reserve_back(head_tiles);
    tile_regs_acquire();
    for (uint32_t i = 0; i < head_tiles; ++i) {
        mul_tiles(cb_x, cb_x, i, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < head_tiles; ++i) {
        pack_tile(i, cb_xmm2);
    }
    tile_regs_release();
    CircularBuffer(cb_xmm2).push_back(head_tiles);

    // mean(x * x): numeric::row_wise_mean<SUM, REDUCE_ROW, false, FullBlockWithPopPolicy>, whose real order is
    // reduce_init -> reduce_tile x Wt -> reduce_uninit -> reconfig -> scale_dest, all inside one DEST acquire.
    CircularBuffer(cb_scaler).wait_front(1);
    reconfig_data_format(cb_xmm2, cb_scaler);
    tile_regs_acquire();
    reconfig_data_format(cb_scaler, cb_xmm2);  // REDUCE_ROW + SUM swaps the operands: scaler in SrcA, data in SrcB
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, cb_ex2);
    CircularBuffer(cb_xmm2).wait_front(head_tiles);
    for (uint32_t j = 0; j < head_tiles; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, j, 0, dst0);
    }
    CircularBuffer(cb_xmm2).pop_front(head_tiles);
    reduce_uninit();
    reconfig_data_format(cb_xmm2, cb_scaler);
    // numeric.h's scale_dest: mul_unary_tile(dst, 1/W). The layernorm op compiles it with math_approx_mode = true, so
    // the call goes through the same macro with `true` in place of APPROX. The 16-bit DEST stores the product by
    // truncation.
    binop_with_scalar_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        false /* is_fp32_dest_acc_en */,
        calculate_binop_with_scalar,
        (true /* APPROX */, MUL_UNARY, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */),
        dst0,
        VectorMode::RC,
        inverse_width_bits));
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_ex2).reserve_back(1);
    pack_reconfig_data_format(cb_ex2);
    pack_tile(dst0, cb_ex2);
    tile_regs_release();
    CircularBuffer(cb_ex2).push_back(1);

    // + eps (FPU add against the reader's TRUNCATED bf16 epsilon tile), then rsqrt
    CircularBuffer(cb_ex2).wait_front(1);
    CircularBuffer(cb_eps).wait_front(1);
    reconfig_data_format(cb_ex2, cb_eps);
    tile_regs_acquire();
    add_init(cb_ex2, cb_eps);
    add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
    // layernorm's rsqrt_tile with the op's APPROX = true written out through the macros: the 10-bit approximate
    // reciprocal square root, not the accurate one APPROX = false would select.
    MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (true /* APPROX */)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        false /* is_fp32_dest_acc_en */,
        calculate_rsqrt,
        (true /* APPROX */, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */, false /* FAST_APPROX */),
        dst0,
        VectorMode::RC));
    tile_regs_commit();
    CircularBuffer(cb_ex2).pop_front(1);
    CircularBuffer(cb_ex2pe).reserve_back(1);
    pack_reconfig_data_format(cb_ex2pe);
    tile_regs_wait();
    pack_tile(dst0, cb_ex2pe);
    tile_regs_release();
    CircularBuffer(cb_ex2pe).push_back(1);

    // x * rsqrt(...) (FPU column broadcast)
    CircularBuffer(cb_ex2pe).wait_front(1);
    reconfig_data_format(cb_x, cb_ex2pe);
    pack_reconfig_data_format(cb_unit);
    CircularBuffer(cb_unit).reserve_back(head_tiles);
    tile_regs_acquire();
    mul_bcast_cols_init(cb_x, cb_ex2pe);
    for (uint32_t i = 0; i < head_tiles; ++i) {
        mul_tiles_bcast_cols(cb_x, cb_ex2pe, i, 0, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < head_tiles; ++i) {
        pack_tile(i, cb_unit);
    }
    tile_regs_release();
    CircularBuffer(cb_unit).push_back(head_tiles);
    CircularBuffer(cb_ex2pe).pop_front(1);
    CircularBuffer(cb_x).pop_front(head_tiles);
}

// ttnn.multiply(v_slice, row_mask_bf16_col) as binary_ng's column-broadcast SFPU kernel. It masks v rows past `rows`
// and its zero clamp turns a SiLU -0.0 into +0.0, so the mirror runs it on every v row, as the chain does.
// cb_x -> cb_out.
ALWI void v_row_mask() {
    CircularBuffer(cb_mask).wait_front(1);
    CircularBuffer(cb_maskfull).reserve_back(1);
    reconfig_data_format(cb_mask, cb_mask);
    pack_reconfig_data_format(cb_maskfull);
    unary_bcast_init<BroadcastType::COL>(cb_mask);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(cb_mask, 0, dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_maskfull);
    tile_regs_release();
    CircularBuffer(cb_maskfull).push_back(1);
    CircularBuffer(cb_mask).pop_front(1);

    CircularBuffer(cb_maskfull).wait_front(1);
    CircularBuffer(cb_x).wait_front(head_tiles);
    pack_reconfig_data_format(cb_out);
    CircularBuffer(cb_out).reserve_back(head_tiles);
    for (uint32_t i = 0; i < head_tiles; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, i, dst0);
        reconfig_data_format_srca(cb_maskfull);
        copy_init(cb_maskfull);
        copy_tile(cb_maskfull, 0, dst1);
        mul_binary_tile_init();
        mul_binary_tile(dst0, dst1, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out);
        tile_regs_release();
    }
    CircularBuffer(cb_out).push_back(head_tiles);
    CircularBuffer(cb_x).pop_front(head_tiles);
    CircularBuffer(cb_maskfull).pop_front(1);
}

// ttnn.multiply(x, 128^-0.5): binary_ng's eltwise_binary_sfpu_scalar.cpp -- the lhs tile and the scalar tile copied
// into a DEST pair, then mul_binary_tile (software RNE, 0 * x = +0), one bf16 pack.
ALWI void scale_pass(uint32_t in_cb, uint32_t out_cb) {
    CircularBuffer(in_cb).wait_front(head_tiles);
    CircularBuffer(cb_scale).wait_front(1);
    pack_reconfig_data_format(out_cb);
    CircularBuffer(out_cb).reserve_back(head_tiles);
    for (uint32_t i = 0; i < head_tiles; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(in_cb);
        copy_init(in_cb);
        copy_tile(in_cb, i, dst0);
        reconfig_data_format_srca(cb_scale);
        copy_init(cb_scale);
        copy_tile(cb_scale, 0, dst1);
        mul_binary_tile_init();
        mul_binary_tile(dst0, dst1, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, out_cb);
        tile_regs_release();
    }
    CircularBuffer(out_cb).push_back(head_tiles);
    CircularBuffer(in_cb).pop_front(head_tiles);
}
}  // namespace

void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    constexpr uint32_t first_pair_arg = 1;  // the first runtime arg of the (column group, tile row) pairs

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_sel, cb_in, cb_shift);
    CircularBuffer(cb_sel).wait_front(select_tiles);

    uint32_t previous_group = 0xFFFFFFFFu;
    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t group = get_arg_val<uint32_t>(first_pair_arg + 2 * unit);
        const uint32_t chunk = get_arg_val<uint32_t>(first_pair_arg + 2 * unit + 1);
        const bool fresh = group != previous_group;
        previous_group = group;
        if (fresh) {
            build_tap_tiles();
        }

        CircularBuffer(cb_in).wait_front(window_tiles);
        for (uint32_t column_tile = 0; column_tile < head_tiles; ++column_tile) {
            shifted_taps(chunk, column_tile);
            conv_silu(column_tile, cb_x);
            CircularBuffer(cb_shift).pop_front(history_rows);
        }
        if (group < qk_groups) {
            rms_norm_unit();
            if (group < qk_heads) {  // q carries the chain's scale and the composite's
                scale_pass(cb_unit, cb_x);
                scale_pass(cb_x, cb_out);
            } else {
                scale_pass(cb_unit, cb_out);
            }
        } else {
            v_row_mask();
        }

        // the whole window goes back at once: cb_in is 2 * window_tiles pages and every cycle of pushes and pops
        // sums to its size, so the window read above is always one contiguous block (cb_api.h's contract)
        CircularBuffer(cb_in).pop_front(window_tiles);
        const bool last = unit + 1 == units;
        const bool next_fresh = last || get_arg_val<uint32_t>(first_pair_arg + 2 * (unit + 1)) != group;
        if (next_fresh) {
            CircularBuffer(cb_tap).pop_front(tap_tiles);
            CircularBuffer(cb_tapfull).pop_front(tap_tiles);
        }
    }
}
