// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// matmul_add_norm compute: minimal_matmul's blocked matmul (one N block per core) with a
// residual-add + RMSNorm epilogue per output block. The residual stream is resident in L1 (CB 4 is
// this core's shard of it):
//   pass 1, per tile-row:  h = acc + residual, written back over the residual; h^2 -> CB 8;
//                          row-sum(h^2) / W over this core's real columns, transposed to row 0
//                          -> CB 9 (partial)
//   (the writer swaps partials with the cores holding the other columns of these rows -> CB 10,
//   one row per core)
//   pass 2, per tile-row:  ms = column-sum(gathered partials); inv = rsqrt(ms + eps)
//                          y = (h * bcast_cols(inv)) * gamma -> CB 17
// Block i's pass 2 runs after block i + 1's matmul (see kernel_main).
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
#ifdef EMBED
#include "api/compute/tilize.h"
#endif
#ifdef ZONES
#include "api/debug/kernel_profiler.hpp"
#endif

namespace {
constexpr uint32_t in0_cb = tt::CBIndex::c_0;
constexpr uint32_t in1_cb = tt::CBIndex::c_1;
constexpr uint32_t acc_cb = tt::CBIndex::c_3;
constexpr uint32_t res_cb = tt::CBIndex::c_4;
constexpr uint32_t gamma_cb = tt::CBIndex::c_5;
constexpr uint32_t inv_w_cb = tt::CBIndex::c_6;
constexpr uint32_t x2_cb = tt::CBIndex::c_8;
constexpr uint32_t sq_cb = tt::CBIndex::c_18;
constexpr uint32_t part_cb = tt::CBIndex::c_9;
constexpr uint32_t gath_cb = tt::CBIndex::c_10;
constexpr uint32_t inv_cb = tt::CBIndex::c_11;
constexpr uint32_t one_cb = tt::CBIndex::c_12;
constexpr uint32_t stage_cb = tt::CBIndex::c_14;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr auto D2B = EltwiseBinaryReuseDestType::DEST_TO_SRCB;
}  // namespace

// The epilogue's multiplies and row sums run at HiFi4 whatever the matmul's fidelity (the compute
// API takes MATH_FIDELITY, the matmul's); these mirror reduce_init / reduce_tile, mul_bcast_cols_init /
// mul_tiles_bcast_cols and mul_reuse_dest_init / mul_reuse_dest_tiles with the fidelity fixed.
constexpr auto EPI_FIDELITY = MathFidelity::HiFi4;
#ifdef ROWSUM_FIDELITY
constexpr auto RS_FID = MathFidelity::ROWSUM_FIDELITY;
#else
constexpr auto RS_FID = MathFidelity::HiFi4;
#endif
// The squares of h (bf16) for the row statistic: HiFi2 left y's PCC unchanged at 16384 rows
// (FF2 1171 -> 1160 us, WO 623 -> 615 us against HiFi4).
#ifdef SQ_FIDELITY
constexpr auto SQ_FID = MathFidelity::SQ_FIDELITY;
#else
constexpr auto SQ_FID = MathFidelity::HiFi2;
#endif
// The y multiplies (h * inv, * gamma) feed a bfp8 output; their fidelity is a define.
#ifdef Y_FIDELITY
constexpr auto Y_FID = MathFidelity::Y_FIDELITY;
#else
constexpr auto Y_FID = MathFidelity::HiFi4;
#endif

// Row sum: the LLK puts the scaler in SrcA and the data in SrcB, so formats go (scaler, data).
ALWI void row_sum_init(uint32_t icb, uint32_t scaler_cb, uint32_t ocb) {
    reconfig_data_format(scaler_cb, icb);
    state_configure(scaler_cb, icb, ocb, __builtin_LINE());
    UNPACK((llk_unpack_AB_reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(icb, scaler_cb)));
    MATH((llk_math_reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE, RS_FID>(icb, scaler_cb)));
}

ALWI void row_sum_tile(uint32_t icb, uint32_t scaler_cb, uint32_t itile, uint32_t idst) {
    MATH((llk_math_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, DST_ACCUM_MODE, RS_FID>(icb, scaler_cb, idst)));
    UNPACK((llk_unpack_AB_reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(icb, scaler_cb, itile, 0)));
}

// Column sum (no operand swap: data in SrcA, scaler in SrcB).
ALWI void col_sum_init(uint32_t icb, uint32_t scaler_cb, uint32_t ocb) {
    reconfig_data_format(icb, scaler_cb);
    state_configure(icb, scaler_cb, ocb, __builtin_LINE());
    UNPACK((llk_unpack_AB_reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(icb, scaler_cb)));
    MATH((llk_math_reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL, DST_ACCUM_MODE, EPI_FIDELITY>(icb, scaler_cb)));
}

ALWI void col_sum_tile(uint32_t icb, uint32_t scaler_cb, uint32_t itile, uint32_t idst) {
    MATH((llk_math_reduce<PoolType::SUM, ReduceDim::REDUCE_COL, DST_ACCUM_MODE, EPI_FIDELITY>(icb, scaler_cb, idst)));
    UNPACK((llk_unpack_AB_reduce<PoolType::SUM, ReduceDim::REDUCE_COL>(icb, scaler_cb, itile, 0)));
}

// After a reduce into DST slot 0: swap rows and columns so the 32 sums move between
// column 0 and row 0. The packer's edge mask is left at its default (whole tile).
ALWI void transpose_slot0() {
    MATH((llk_math_reduce_uninit()));
    transpose_dest_init<DST_ACCUM_MODE, true>();
    transpose_dest<DST_ACCUM_MODE, true>(0);
}

ALWI void mul_cols_init(uint32_t icb0, uint32_t icb1) {
    state_configure(icb0, icb1, __builtin_LINE());
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, Y_FID>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1)));
}

ALWI void mul_cols_tile(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::COL,
          DST_ACCUM_MODE,
          Y_FID,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}

ALWI void mul_dest_init(uint32_t icb) {
    state_configure(icb, __builtin_LINE());
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, D2B>(false, false, icb)));
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, Y_FID, D2B>(icb, icb, false)));
}

ALWI void mul_dest_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, true, D2B>(icb, itile)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_ACCUM_MODE, Y_FID, D2B>(
        icb, icb, idst, true)));
}

// h * h with both operands unpacked from the same tile, added onto DST slot `idst` unless `clear`
// (the sum of squares over a row's tiles accumulates in DST).
ALWI void square_init(uint32_t icb) {
    reconfig_data_format(icb, icb);
    state_configure(icb, icb, __builtin_LINE());
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, SQ_FID>(icb, icb, false)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb, icb, Transpose::None)));
}

ALWI void square_acc_tile(uint32_t icb, uint32_t itile, uint32_t idst, bool clear) {
    UNPACK((llk_unpack_AB(icb, icb, itile, itile)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          SQ_FID,
          EltwiseBinaryReuseDestType::NONE>(icb, icb, idst, clear)));
}

// add_unary_tile / rsqrt_tile over the column faces only (faces 0 and 2, VectorMode::C): the inverse
// RMS lives in column 0, so the other half of the tile is wasted SFPU work.
ALWI void add_scalar_col(uint32_t idst, uint32_t bits) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
        idst,
        VectorMode::C,
        bits));
}

ALWI void rsqrt_col(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 8, DST_ACCUM_MODE, true), idst, VectorMode::C));
}

// From minimal_matmul compute.cpp.
void matmul_blocks(
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;
    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();
            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, false, subblock_w, subblock_h, K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                for (uint32_t w = 0; w < subblock_w; w++) {
                    pack_tile<true>(write_dst_index, acc_cb, (M_start + h) * full_N_block_tiles + N_start + w);
                    write_dst_index++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

// h = acc + residual, written back over the residual (CB 4 is this core's resident shard of the
// residual stream, res_w tiles wide; block `blk` row m is shard row blk * M_block + m). Then each
// row's sum of squares: the FPU multiplies each real h tile by itself, read back from the shard,
// accumulating in one DST slot per row; those tiles (CB 18) are row-summed / W and transposed into
// row 0 of one partial tile per row (the writer's exchange format). The phases run over the whole
// block, so the packer-to-unpacker handoffs (CB 8 for h, CB 18 for the squares) happen once per
// block. Columns past res_w (N block padding) read a real tile and are dropped.
template <uint32_t M_block_tiles, uint32_t N_block_tiles, uint32_t res_w>
void pass1(uint32_t blk, uint32_t n_real) {
    constexpr uint32_t out_tiles = M_block_tiles * N_block_tiles;
    constexpr uint32_t P = 2;  // h tiles per DST session
    constexpr uint32_t R = 4;  // rows per DST session (fp32 DST's 4 half-sync slots)
    CircularBuffer cb_acc(acc_cb), cb_x2(x2_cb), cb_sq(sq_cb), cb_part(part_cb);
    const uint32_t base = blk * M_block_tiles * res_w;
#ifndef NORM_ONLY
    {
#ifdef ZONES
        DeviceZoneScopedN("P1A");
#endif
        cb_acc.wait_front(out_tiles);
        reconfig_data_format(acc_cb, res_cb);
        add_init(acc_cb, res_cb);
        pack_reconfig_data_format(res_cb);
        for (uint32_t m = 0; m < M_block_tiles; ++m) {
            const uint32_t row = base + m * res_w;
            for (uint32_t n = 0; n < N_block_tiles; n += P) {
                const uint32_t t = m * N_block_tiles + n;
                const uint32_t cnt = N_block_tiles - n < P ? N_block_tiles - n : P;
                tile_regs_acquire();
                for (uint32_t j = 0; j < cnt; ++j) {
                    add_tiles(acc_cb, res_cb, t + j, row + (n + j < res_w ? n + j : 0), j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < cnt; ++j) {
                    if (n + j < res_w) {
                        pack_tile<true>(j, res_cb, row + n + j);
                    }
                }
                tile_regs_release();
            }
        }
        cb_acc.pop_front(out_tiles);
        // The unpacker reads h back below only after this push, i.e. after the packs above.
        cb_x2.reserve_back(1);
        cb_x2.push_back(1);
        cb_x2.wait_front(1);
    }
#endif
    {
#ifdef ZONES
        DeviceZoneScopedN("P1B");
#endif
        cb_sq.reserve_back(M_block_tiles);
        pack_reconfig_data_format(sq_cb);
        square_init(res_cb);
        for (uint32_t m0 = 0; m0 < M_block_tiles; m0 += R) {
            const uint32_t cnt = M_block_tiles - m0 < R ? M_block_tiles - m0 : R;
            tile_regs_acquire();
            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t row = base + (m0 + i) * res_w;
                for (uint32_t j = 0; j < n_real; ++j) {
                    square_acc_tile(res_cb, row + j, i, j == 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < cnt; ++i) {
                pack_tile<true>(i, sq_cb, m0 + i);
            }
            tile_regs_release();
        }
        cb_sq.push_back(M_block_tiles);
    }
#ifndef NORM_ONLY
    cb_x2.pop_front(1);
#endif
    cb_sq.wait_front(M_block_tiles);
    cb_part.reserve_back(M_block_tiles);
    pack_reconfig_data_format(part_cb);
    for (uint32_t m0 = 0; m0 < M_block_tiles; m0 += R) {
        const uint32_t cnt = M_block_tiles - m0 < R ? M_block_tiles - m0 : R;
        row_sum_init(sq_cb, inv_w_cb, part_cb);
        tile_regs_acquire();
        for (uint32_t i = 0; i < cnt; ++i) {
            row_sum_tile(sq_cb, inv_w_cb, m0 + i, i);
        }
        // Partial sums: column 0 -> row 0, for the writer's contiguous sends.
        MATH((llk_math_reduce_uninit()));
        transpose_dest_init<DST_ACCUM_MODE, true>();
        for (uint32_t i = 0; i < cnt; ++i) {
            transpose_dest<DST_ACCUM_MODE, true>(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < cnt; ++i) {
            pack_tile<true>(i, part_cb, m0 + i);
        }
        tile_regs_release();
    }
    cb_sq.pop_front(M_block_tiles);
    // Pushed after every h tile is packed: the writer's exchange waits on it, and pass 2 waits on
    // the gathered tiles the exchange produces, so pass 2's reads of h follow these packs.
    cb_part.push_back(M_block_tiles);
}

#ifdef EMBED
// embed_rmsnorm: tilize block `blk`'s tile-rows of gathered embedding rows (CB 14, row-major, from
// embed_gather.cpp) into the residual shard, then hand them to pass 1 (the unpacker waits on the push).
template <uint32_t M_block_tiles, uint32_t res_w>
void embed_block(uint32_t blk) {
    CircularBuffer cb_stage(stage_cb), cb_res(res_cb);
    reconfig_data_format_srca(stage_cb);
    pack_reconfig_data_format(res_cb);
    tilize_init(stage_cb, res_w, res_cb);
    for (uint32_t m = 0; m < M_block_tiles; ++m) {
        cb_stage.wait_front(res_w);
        tilize_block(stage_cb, res_w, res_cb, 0, m * res_w);  // relative to the shard's write pointer
        cb_stage.pop_front(res_w);
    }
    tilize_uninit(stage_cb, res_cb);
    cb_res.push_back(M_block_tiles * res_w);
    cb_res.wait_front((blk + 1) * M_block_tiles * res_w);
}
#endif

// inv = rsqrt(column-sum of the gathered partials + eps) per tile-row; y = h * inv * gamma, with h
// read back from the residual shard.
constexpr uint32_t G = 4;  // tile-rows per pass-2 group (their inverse RMS in one fp32 DST session)

// Pass 2 in groups of G tile-rows, so a group can run between two K blocks of the next block's
// matmul (see kernel_main). pass2_begin / pass2_end bracket the groups of one block.
template <uint32_t M_block_tiles, uint32_t N_block_tiles>
void pass2_begin() {
    CircularBuffer cb_gath(gath_cb), cb_y(y_cb);
    cb_gath.wait_front(M_block_tiles);
    cb_y.reserve_back(M_block_tiles * N_block_tiles);
}

template <uint32_t M_block_tiles, uint32_t N_block_tiles>
void pass2_end() {
    CircularBuffer cb_gath(gath_cb), cb_y(y_cb);
    cb_y.push_back(M_block_tiles * N_block_tiles);
    cb_gath.pop_front(M_block_tiles);
}

// Group m0: inv = rsqrt(column-sum of the gathered partials + eps) for G tile-rows (CB 11), then
// y = h * inv (* gamma) for those rows, h read back from the residual shard.
template <uint32_t M_block_tiles, uint32_t N_block_tiles, uint32_t res_w>
void pass2_group(uint32_t blk, uint32_t m0, uint32_t eps_bits) {
    CircularBuffer cb_inv(inv_cb);
    const uint32_t cnt = M_block_tiles - m0 < G ? M_block_tiles - m0 : G;
    cb_inv.reserve_back(G);
    pack_reconfig_data_format(inv_cb);
    col_sum_init(gath_cb, one_cb, inv_cb);
    tile_regs_acquire();
    for (uint32_t j = 0; j < cnt; ++j) {
        col_sum_tile(gath_cb, one_cb, m0 + j, j);
    }
    MATH((llk_math_reduce_uninit()));
    transpose_dest_init<DST_ACCUM_MODE, true>();
    for (uint32_t j = 0; j < cnt; ++j) {
        transpose_dest<DST_ACCUM_MODE, true>(j);  // mean squares: row 0 -> column 0
    }
    binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < cnt; ++j) {
        add_scalar_col(j, eps_bits);
    }
    rsqrt_tile_init();
    for (uint32_t j = 0; j < cnt; ++j) {
        rsqrt_col(j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < cnt; ++j) {
        pack_tile<true>(j, inv_cb, j);
    }
    tile_regs_release();
    cb_inv.push_back(G);
    cb_inv.wait_front(G);

    reconfig_data_format(res_cb, inv_cb);
    mul_cols_init(res_cb, inv_cb);
    pack_reconfig_data_format(y_cb);
    for (uint32_t mi = 0; mi < cnt; ++mi) {
        const uint32_t m = m0 + mi;
        const uint32_t row = (blk * M_block_tiles + m) * res_w;
        for (uint32_t n = 0; n < N_block_tiles; n += 4) {
            const uint32_t c4 = N_block_tiles - n < 4 ? N_block_tiles - n : 4;
#ifndef NO_GAMMA
            reconfig_data_format(res_cb, inv_cb);
            mul_cols_init(res_cb, inv_cb);
#endif
            tile_regs_acquire();
            for (uint32_t j = 0; j < c4; ++j) {
                mul_cols_tile(res_cb, inv_cb, row + (n + j < res_w ? n + j : 0), mi, j);
            }
#ifndef NO_GAMMA
            reconfig_data_format_srca(gamma_cb);
            mul_dest_init(gamma_cb);
            for (uint32_t j = 0; j < c4; ++j) {
                mul_dest_tile(gamma_cb, n + j, j);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < c4; ++j) {
                pack_tile<true>(j, y_cb, m * N_block_tiles + n + j);
            }
            tile_regs_release();
        }
    }
    cb_inv.pop_front(G);
}

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(7);
    constexpr uint32_t res_w = get_compile_time_arg_val(8);
    constexpr uint32_t res_shard_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t n_groups = (M_block_tiles + G - 1) / G;

    const uint32_t M_start_tile = get_arg_val<uint32_t>(0);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(1);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(2);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(3);
    const uint32_t n_real = get_arg_val<uint32_t>(4);

    CircularBuffer cb_in0(in0_cb), cb_in1(in1_cb), cb_acc(acc_cb), cb_gamma(gamma_cb), cb_inv_w(inv_w_cb);
    CircularBuffer cb_one(one_cb), cb_res(res_cb);

#ifdef NORM_ONLY
    compute_kernel_hw_startup(res_cb, inv_w_cb, x2_cb);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, acc_cb);
    matmul_init(in0_cb, in1_cb);
#endif

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
    const uint32_t current_N_block_tiles = N_end_tile - N_start_tile;
    const uint32_t current_subblock_w = std::min(current_N_block_tiles, subblock_w);

#ifndef NO_GAMMA
    cb_gamma.wait_front(N_block_tiles);
#endif
    cb_inv_w.wait_front(1);
    cb_one.wait_front(1);
#ifdef EMBED
    cb_res.reserve_back(res_shard_tiles);
#else
    cb_res.wait_front(res_shard_tiles);
#endif

#ifdef NORM_ONLY
    // y = rmsnorm(residual) only: block i's first pass, then block i - 1's second pass.
    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
#ifdef EMBED
        embed_block<M_block_tiles, res_w>(m_block_iter);
#endif
        pass1<M_block_tiles, N_block_tiles, res_w>(m_block_iter, n_real);
        if (m_block_iter > 0) {
            pass2_begin<M_block_tiles, N_block_tiles>();
            for (uint32_t m0 = 0; m0 < M_block_tiles; m0 += G) {
                pass2_group<M_block_tiles, N_block_tiles, res_w>(m_block_iter - 1, m0, eps_bits);
            }
            pass2_end<M_block_tiles, N_block_tiles>();
        }
    }
    pass2_begin<M_block_tiles, N_block_tiles>();
    for (uint32_t m0 = 0; m0 < M_block_tiles; m0 += G) {
        pass2_group<M_block_tiles, N_block_tiles, res_w>(M_blocks_per_core - 1, m0, eps_bits);
    }
    pass2_end<M_block_tiles, N_block_tiles>();
    return;
#endif
    // Software pipeline. Block i's second pass needs the other cores' partial sums, which arrive
    // while block i + 1's matmul runs, so it runs after that matmul (block i's first pass needs
    // the finished accumulator). With INTERLEAVE_P2 its groups instead run between K blocks of
    // block i + 1's matmul (it does not touch the accumulator); measured: FF2 1180 -> 1173 us,
    // WO 628 -> 640 us at 16384 rows, i.e. the matmul has little input-wait slack to fill.
    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        const uint32_t current_M_block_tiles = m_tile_end - m_tile;
        const uint32_t current_subblock_h = std::min(current_M_block_tiles, subblock_h);
        const bool has_p2 = m_block_iter > 0;
        uint32_t groups_done = 0;

#ifdef ZONES
        DeviceZoneScopedN("MM");
#endif
        reconfig_data_format(in1_cb, in0_cb);
        pack_reconfig_data_format(acc_cb);
        matmul_block_init(in0_cb, in1_cb, false, current_subblock_w, current_subblock_h, K_block_tiles);
        cb_acc.reserve_back(out_block_num_tiles);
        for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
            cb_in0.wait_front(in0_block_num_tiles);
            cb_in1.wait_front(in1_block_num_tiles);
            matmul_blocks(
                current_M_block_tiles,
                current_N_block_tiles,
                N_block_tiles,
                K_block_tiles,
                current_subblock_h,
                current_subblock_w);
            cb_in0.pop_front(in0_block_num_tiles);
            cb_in1.pop_front(in1_block_num_tiles);
            if (k_block == 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
#ifdef INTERLEAVE_P2
            // Group g runs after K block 2 + g * (K_num_blocks - 2) / n_groups.
            if (has_p2 && groups_done < n_groups && K_num_blocks > 3 && k_block + 1 < K_num_blocks &&
                k_block == 2 + groups_done * (K_num_blocks - 3) / n_groups) {
                PACK((llk_pack_reconfig_l1_acc(0)));
                if (groups_done == 0) {
                    pass2_begin<M_block_tiles, N_block_tiles>();
                }
                pass2_group<M_block_tiles, N_block_tiles, res_w>(m_block_iter - 1, groups_done * G, eps_bits);
                ++groups_done;
                reconfig_data_format(in1_cb, in0_cb);
                pack_reconfig_data_format(acc_cb);
                matmul_block_init(in0_cb, in1_cb, false, current_subblock_w, current_subblock_h, K_block_tiles);
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
#endif
        }
        cb_acc.push_back(out_block_num_tiles);
        PACK((llk_pack_reconfig_l1_acc(0)));

        if (has_p2) {
#ifdef ZONES
            DeviceZoneScopedN("P2");
#endif
            if (groups_done == 0) {
                pass2_begin<M_block_tiles, N_block_tiles>();
            }
            for (; groups_done < n_groups; ++groups_done) {
                pass2_group<M_block_tiles, N_block_tiles, res_w>(m_block_iter - 1, groups_done * G, eps_bits);
            }
            pass2_end<M_block_tiles, N_block_tiles>();
        }
        {
#ifdef ZONES
            DeviceZoneScopedN("P1");
#endif
            pass1<M_block_tiles, N_block_tiles, res_w>(m_block_iter, n_real);
        }
    }
    pass2_begin<M_block_tiles, N_block_tiles>();
    for (uint32_t m0 = 0; m0 < M_block_tiles; m0 += G) {
        pass2_group<M_block_tiles, N_block_tiles, res_w>(M_blocks_per_core - 1, m0, eps_bits);
    }
    pass2_end<M_block_tiles, N_block_tiles>();
}
