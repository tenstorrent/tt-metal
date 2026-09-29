// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group A compute: one (column group, tile row) unit per loop, in a 16-BIT DEST (fp32_dest_acc_en = false) with
// APPROX = false, HiFi4 -- the chain's own configuration for every op here except the two SFPU calls inside
// ttnn.rms_norm, which the layernorm op compiles with math_approx_mode = true and which are therefore written out
// through their macros with `true` in place of APPROX (see the two comments below).
//
// Per unit, each phase is one chain op's LLK sequence with that op's pack:
//   1. the three shifted FIR taps: two matmul_tiles of a 0/1 selection tile into one DEST tile (exact: one 1.0 term
//      per element), packed bf16 -- the mirror of _shifted_rows_slab's row shifts;
//   2. the conv: ttnn.multiply(tap0, w0) as binary_ng's SFPU row-broadcast kernel (the tap row materialised once per
//      column group by unary_bcast<ROW>, then copy / copy / mul_binary_tile) and three ttnn.mac as the ternary SFPU
//      kernel (copy a / copy the broadcast row / copy the accumulator, then mac_tile<Float16_b> IMMEDIATELY after its
//      init: the init records the replay slots and any SFPU op in between invalidates them); five bf16 packs;
//   3. ttnn.silu as the unary SFPU kernel;
//   4. for q/k: the ttnn.rms_norm mirror (layernorm.cpp's RMSNORM path, no gamma, W = 128 so Wt = 4 in one block)
//      and then the scale ttnn.multiply(x, 128^-0.5) as binary_ng's scalar kernel against the host-rounded bf16
//      scalar tile -- twice for q (the chain's, then the composite's chunk_gated_delta_rule q * scale);
//   5. for v: ttnn.multiply(v_slice, row_mask_bf16_col) as binary_ng's column-broadcast kernel, which carries the
//      rows mask and the bf16 multiply's zero clamp.
//
// CBs: in 0, sel 1, tap 2, tapfull 3, shift 4, conv 5, x 6, xmm2 7, scaler 8, eps 9, scale 10, ex2 11, ex2pe 12,
// unit 13, out 14, mask 15, maskfull 16.  CB_IN is the reader's window: the previous tile row at 0..3, this
// one at 4..7.
// Compile-time args: none (the page maps live in the reader and the writer).  Runtime args: 0 units, then
// (column group, tile row) pairs.

#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
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
#include "../../kernels/zones.h"

using namespace ckernel;

namespace {
constexpr uint32_t CB_IN = 0, CB_SEL = 1, CB_TAP = 2, CB_TAPFULL = 3, CB_SHIFT = 4, CB_CONV = 5, CB_X = 6;
constexpr uint32_t CB_XMM2 = 7, CB_SCALER = 8, CB_EPS = 9, CB_SCALE = 10, CB_EX2 = 11, CB_EX2PE = 12;
constexpr uint32_t CB_UNIT = 13, CB_OUT = 14, CB_MASK = 15, CB_MASKFULL = 16;

constexpr uint32_t HEAD_TILES = 4;  // 128 / 32: Wt of the q/k norm
constexpr uint32_t CONV_KERNEL = 4;
constexpr uint32_t HISTORY_ROWS = 3;  // the three shifted taps
constexpr uint32_t QK_HEADS = 4, QK_GROUPS = 8;
constexpr uint32_t TAP_TILES = CONV_KERNEL * HEAD_TILES;
constexpr uint32_t WINDOW = 2 * HEAD_TILES;  // CB_IN: the previous tile row at 0..3, this one at 4..7
constexpr uint32_t SELECT_CUR = 0, SELECT_PREV = 1, SELECT_HIST = 2, SELECT_KINDS = 3;
constexpr uint32_t SELECT_TILES = HISTORY_ROWS * SELECT_KINDS;  // nine 0/1 tiles, pushed once by the reader
constexpr uint32_t RECIP_W = 0x3C000000;                        // 1.0f / 128: the layernorm reduce's 1/W scale
constexpr uint32_t dst0 = 0, dst1 = 1, dst2 = 2;

constexpr uint32_t select_index(uint32_t kind, uint32_t shift) { return (shift - 1) * SELECT_KINDS + kind; }

// The four tap rows of this column group as full tiles: binary_ng's and the ternary kernel's `unary_bcast<ROW>` of a
// [1, 1, 1, 2560] operand, materialised once per group instead of once per output tile (the same bits: the broadcast
// is a pure function of the tap tile, and the pack that follows it is lossless bf16 -> bf16).
ALWI void build_tap_tiles() {
    FUSED_ZONE("fz_gpr_cq_tapfull");
    cb_wait_front(CB_TAP, TAP_TILES);
    cb_reserve_back(CB_TAPFULL, TAP_TILES);
    reconfig_data_format(CB_TAP, CB_TAP);
    pack_reconfig_data_format(CB_TAPFULL);
    unary_bcast_init<BroadcastType::ROW>(CB_TAP);
    for (uint32_t t = 0; t < TAP_TILES; ++t) {
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(CB_TAP, t, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, CB_TAPFULL);
        tile_regs_release();
    }
    cb_push_back(CB_TAPFULL, TAP_TILES);
    cb_wait_front(CB_TAPFULL, TAP_TILES);
}

// taps 0..2 of column tile d: shifted_s = Sel_prev_or_hist(s) @ previous + Sel_cur(s) @ current, one DEST tile per
// tap (tap 0 is the shift by 3, tap 2 the shift by 1), packed bf16.
ALWI void shifted_taps(uint32_t chunk, uint32_t d) {
    FUSED_ZONE("fz_gpr_cq_fir");
    reconfig_data_format<SrcOrder::Reverse>(CB_SEL, CB_IN);
    matmul_init(CB_SEL, CB_IN);
    pack_reconfig_data_format(CB_SHIFT);
    cb_reserve_back(CB_SHIFT, HISTORY_ROWS);
    for (uint32_t tap = 0; tap < HISTORY_ROWS; ++tap) {
        const uint32_t shift = HISTORY_ROWS - tap;
        const uint32_t before = select_index(chunk == 0 ? SELECT_HIST : SELECT_PREV, shift);
        tile_regs_acquire();
        matmul_tiles(CB_SEL, CB_IN, before, d, dst0);
        matmul_tiles(CB_SEL, CB_IN, select_index(SELECT_CUR, shift), HEAD_TILES + d, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, CB_SHIFT);
        tile_regs_release();
    }
    cb_push_back(CB_SHIFT, HISTORY_ROWS);
    cb_wait_front(CB_SHIFT, HISTORY_ROWS);
}

// conv = silu(mul(tap0, w0) then mac(tap_t, w_t, conv) for t = 1..3) on column tile d; one bf16 pack per op.
ALWI void conv_silu(uint32_t d, uint32_t cb_silu) {
    FUSED_ZONE("fz_gpr_cq_conv_silu");
    pack_reconfig_data_format(CB_CONV);
    cb_reserve_back(CB_CONV, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(CB_SHIFT);
    copy_init(CB_SHIFT);
    copy_tile(CB_SHIFT, 0, dst0);
    reconfig_data_format_srca(CB_TAPFULL);
    copy_init(CB_TAPFULL);
    copy_tile(CB_TAPFULL, HEAD_TILES * d + 0, dst1);
    mul_binary_tile_init();
    mul_binary_tile(dst0, dst1, dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, CB_CONV);
    tile_regs_release();
    cb_push_back(CB_CONV, 1);

    for (uint32_t t = 1; t < CONV_KERNEL; ++t) {
        cb_wait_front(CB_CONV, 1);
        cb_reserve_back(CB_CONV, 1);
        tile_regs_acquire();
        if (t < HISTORY_ROWS) {
            reconfig_data_format_srca(CB_SHIFT);
            copy_init(CB_SHIFT);
            copy_tile(CB_SHIFT, t, dst0);
        } else {  // tap 3 is the rows themselves: the current tile row, already in CB_IN
            reconfig_data_format_srca(CB_IN);
            copy_init(CB_IN);
            copy_tile(CB_IN, HEAD_TILES + d, dst0);
        }
        reconfig_data_format_srca(CB_TAPFULL);
        copy_init(CB_TAPFULL);
        copy_tile(CB_TAPFULL, HEAD_TILES * d + t, dst1);
        reconfig_data_format_srca(CB_CONV);
        copy_init(CB_CONV);
        copy_tile(CB_CONV, 0, dst2);
        // mac_tile is only valid immediately after mac_tile_init (replay slots 0..6): no SFPU op in between.
        mac_tile_init<DataFormat::Float16_b>();
        mac_tile<DataFormat::Float16_b>(0, 1, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, CB_CONV);
        tile_regs_release();
        cb_push_back(CB_CONV, 1);
        cb_pop_front(CB_CONV, 1);
    }

    cb_wait_front(CB_CONV, 1);
    pack_reconfig_data_format(cb_silu);
    cb_reserve_back(cb_silu, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(CB_CONV);
    copy_init(CB_CONV);
    copy_tile(CB_CONV, 0, dst0);
    silu_tile_init();
    silu_tile(dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_silu);
    tile_regs_release();
    cb_push_back(cb_silu, 1);
    cb_pop_front(CB_CONV, 1);
}

// ttnn.rms_norm(x bf16, epsilon = 1e-6 / 128) with no weight: layernorm.cpp's RMSNORM path at Wt = 4, every
// intermediate a bf16 CB in the 16-bit DEST.  CB_X -> CB_UNIT.
ALWI void rms_norm_unit() {
    FUSED_ZONE("fz_gpr_cq_norm");
    cb_wait_front(CB_X, HEAD_TILES);

    // x * x -> xmm2 (FPU mul_tiles, one block)
    reconfig_data_format(CB_X, CB_X);
    pack_reconfig_data_format(CB_XMM2);
    mul_init(CB_X, CB_X);
    cb_reserve_back(CB_XMM2, HEAD_TILES);
    tile_regs_acquire();
    for (uint32_t i = 0; i < HEAD_TILES; ++i) {
        mul_tiles(CB_X, CB_X, i, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < HEAD_TILES; ++i) {
        pack_tile(i, CB_XMM2);
    }
    tile_regs_release();
    cb_push_back(CB_XMM2, HEAD_TILES);

    // mean(x * x): numeric::row_wise_mean<SUM, REDUCE_ROW, false, FullBlockWithPopPolicy>, whose real order is
    // reduce_init -> reduce_tile x Wt -> reduce_uninit -> reconfig -> scale_dest, all inside one DEST acquire.
    cb_wait_front(CB_SCALER, 1);
    reconfig_data_format(CB_XMM2, CB_SCALER);
    tile_regs_acquire();
    reconfig_data_format(CB_SCALER, CB_XMM2);  // REDUCE_ROW + SUM swaps the operands: scaler in SrcA, data in SrcB
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, CB_EX2);
    cb_wait_front(CB_XMM2, HEAD_TILES);
    for (uint32_t j = 0; j < HEAD_TILES; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, j, 0, dst0);
    }
    cb_pop_front(CB_XMM2, HEAD_TILES);
    reduce_uninit();
    reconfig_data_format(CB_XMM2, CB_SCALER);
    // numeric.h detail::scale_dest = binop_with_scalar_tile_init() + mul_unary_tile(dst, 1/W).  The layernorm op is
    // compiled math_approx_mode = true (rmsnorm.cpp:16-20) while this kernel is APPROX = false, so the call is
    // written out through the same macro (blackhole llk_math_eltwise_unary_sfpu_macros.h:41-47) with `true` in place
    // of APPROX; the init (binop_with_scalar.h:145-151) does not depend on APPROX.  The 16-bit DEST stores this
    // product by TRUNCATION (ckernel_sfpu_binop_with_unary.h: only RSUB re-rounds).
    binop_with_scalar_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        false /* is_fp32_dest_acc_en */,
        calculate_binop_with_scalar,
        (true /* APPROX */, MUL_UNARY, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */),
        dst0,
        VectorMode::RC,
        RECIP_W));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_EX2, 1);
    pack_reconfig_data_format(CB_EX2);
    pack_tile(dst0, CB_EX2);
    tile_regs_release();
    cb_push_back(CB_EX2, 1);

    // + eps (FPU add against the reader's TRUNCATED bf16 epsilon tile), then rsqrt
    cb_wait_front(CB_EX2, 1);
    cb_wait_front(CB_EPS, 1);
    reconfig_data_format(CB_EX2, CB_EPS);
    tile_regs_acquire();
    add_init(CB_EX2, CB_EPS);
    add_tiles(CB_EX2, CB_EPS, 0, 0, dst0);
    // rsqrt_tile_init() / rsqrt_tile() of layernorm.cpp, again with the op's APPROX = true written out
    // through the macros (rsqrt.h:18-20 init, :37-43 call; SFPU_UNARY_INIT_FN at macros.h:78-80): the 10-bit
    // approximate reciprocal square root, not the accurate one this kernel's APPROX = false would select.
    MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (true /* APPROX */)));
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        false /* is_fp32_dest_acc_en */,
        calculate_rsqrt,
        (true /* APPROX */, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */, false /* FAST_APPROX */),
        dst0,
        VectorMode::RC));
    tile_regs_commit();
    cb_pop_front(CB_EX2, 1);
    cb_reserve_back(CB_EX2PE, 1);
    pack_reconfig_data_format(CB_EX2PE);
    tile_regs_wait();
    pack_tile(dst0, CB_EX2PE);
    tile_regs_release();
    cb_push_back(CB_EX2PE, 1);

    // x * rsqrt(...) (FPU column broadcast)
    cb_wait_front(CB_EX2PE, 1);
    reconfig_data_format(CB_X, CB_EX2PE);
    pack_reconfig_data_format(CB_UNIT);
    cb_reserve_back(CB_UNIT, HEAD_TILES);
    tile_regs_acquire();
    mul_bcast_cols_init(CB_X, CB_EX2PE);
    for (uint32_t i = 0; i < HEAD_TILES; ++i) {
        mul_tiles_bcast_cols(CB_X, CB_EX2PE, i, 0, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < HEAD_TILES; ++i) {
        pack_tile(i, CB_UNIT);
    }
    tile_regs_release();
    cb_push_back(CB_UNIT, HEAD_TILES);
    cb_pop_front(CB_EX2PE, 1);
    cb_pop_front(CB_X, HEAD_TILES);
}

// ttnn.multiply(v_slice, row_mask_bf16_col): binary_ng's eltwise_binary_sfpu_col_bcast.cpp -- the mask column
// materialised once per tile row by unary_bcast<COL> and packed, then copy lhs / copy the broadcast tile /
// mul_binary_tile per tile.  This is what masks a v row past `rows` (x * 0.0), and its zero clamp is also what
// canonicalises a -0.0 that the SiLU may have produced: the chain runs this op on every v row, so the mirror must
// too.  CB_X -> CB_OUT.
ALWI void v_row_mask() {
    FUSED_ZONE("fz_gpr_cq_vmask");
    cb_wait_front(CB_MASK, 1);
    cb_reserve_back(CB_MASKFULL, 1);
    reconfig_data_format(CB_MASK, CB_MASK);
    pack_reconfig_data_format(CB_MASKFULL);
    unary_bcast_init<BroadcastType::COL>(CB_MASK);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(CB_MASK, 0, dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, CB_MASKFULL);
    tile_regs_release();
    cb_push_back(CB_MASKFULL, 1);
    cb_pop_front(CB_MASK, 1);

    cb_wait_front(CB_MASKFULL, 1);
    cb_wait_front(CB_X, HEAD_TILES);
    pack_reconfig_data_format(CB_OUT);
    cb_reserve_back(CB_OUT, HEAD_TILES);
    for (uint32_t i = 0; i < HEAD_TILES; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_X);
        copy_init(CB_X);
        copy_tile(CB_X, i, dst0);
        reconfig_data_format_srca(CB_MASKFULL);
        copy_init(CB_MASKFULL);
        copy_tile(CB_MASKFULL, 0, dst1);
        mul_binary_tile_init();
        mul_binary_tile(dst0, dst1, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, CB_OUT);
        tile_regs_release();
    }
    cb_push_back(CB_OUT, HEAD_TILES);
    cb_pop_front(CB_X, HEAD_TILES);
    cb_pop_front(CB_MASKFULL, 1);
}

// ttnn.multiply(x, 128^-0.5): binary_ng's eltwise_binary_sfpu_scalar.cpp -- the lhs tile and the scalar tile copied
// into a DEST pair, then mul_binary_tile (software RNE, 0 * x = +0), one bf16 pack.
ALWI void scale_pass(uint32_t cb_in, uint32_t cb_out) {
    FUSED_ZONE("fz_gpr_cq_scale");
    cb_wait_front(cb_in, HEAD_TILES);
    cb_wait_front(CB_SCALE, 1);
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, HEAD_TILES);
    for (uint32_t i = 0; i < HEAD_TILES; ++i) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_in);
        copy_init(cb_in);
        copy_tile(cb_in, i, dst0);
        reconfig_data_format_srca(CB_SCALE);
        copy_init(CB_SCALE);
        copy_tile(CB_SCALE, 0, dst1);
        mul_binary_tile_init();
        mul_binary_tile(dst0, dst1, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, HEAD_TILES);
    cb_pop_front(cb_in, HEAD_TILES);
}
}  // namespace

void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    constexpr uint32_t PAIRS = 1;  // the first runtime arg of the (column group, tile row) pairs

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_SEL, CB_IN, CB_SHIFT);
    cb_wait_front(CB_SEL, SELECT_TILES);

    uint32_t previous_group = 0xFFFFFFFFu;
    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t group = get_arg_val<uint32_t>(PAIRS + 2 * unit);
        const uint32_t chunk = get_arg_val<uint32_t>(PAIRS + 2 * unit + 1);
        const bool fresh = group != previous_group;
        previous_group = group;
        if (fresh) {
            build_tap_tiles();
        }

        cb_wait_front(CB_IN, WINDOW);
        for (uint32_t d = 0; d < HEAD_TILES; ++d) {
            shifted_taps(chunk, d);
            conv_silu(d, CB_X);
            cb_pop_front(CB_SHIFT, HISTORY_ROWS);
        }
        if (group < QK_GROUPS) {
            rms_norm_unit();
            if (group < QK_HEADS) {  // q carries the chain's scale and the composite's
                scale_pass(CB_UNIT, CB_X);
                scale_pass(CB_X, CB_OUT);
            } else {
                scale_pass(CB_UNIT, CB_OUT);
            }
        } else {
            v_row_mask();
        }

        // the whole window goes back at once: CB_IN is 2 * WINDOW pages and every cycle of pushes and pops sums to
        // its size, so the WINDOW-tile read above is always one contiguous block (cb_api.h's contract)
        cb_pop_front(CB_IN, WINDOW);
        const bool last = unit + 1 == units;
        const bool next_fresh = last || get_arg_val<uint32_t>(PAIRS + 2 * (unit + 1)) != group;
        if (next_fresh) {
            cb_pop_front(CB_TAP, TAP_TILES);
            cb_pop_front(CB_TAPFULL, TAP_TILES);
        }
    }
}
