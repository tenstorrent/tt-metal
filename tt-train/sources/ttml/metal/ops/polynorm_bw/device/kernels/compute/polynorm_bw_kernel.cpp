// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// PolyNorm3 backward compute kernel
//
// Computes gradients for the polynomial normalization:
//   out = w0·RmsNorm(x³) + w1·RmsNorm(x²) + w2·RmsNorm(x) + b
// where RmsNorm(t) = t / sqrt(mean(t²) + ε).
//
// Given upstream gradient dL/dout (same shape as x), this kernel produces:
//   dL/dx  — full-sized gradient tensor (same shape as x)
//   dL/dw  — packed as 3 scalar tiles (one per weight)
//   dL/db  — packed as 1 scalar tile
//
// Math (per row of length N, summing k=1,2,3 with f_k = w_{3-k}):
//
//   inv_rms_k = (Σ x^(2k)/N + ε)^(-1/2)
//   g_k       = Σ x^k · dout
//
//   dL/df_k = inv_rms_k · g_k                          (per-row scalar)
//   dL/db   = Σ dout                                   (per-row scalar)
//   dL/dx_i = Σ_k  α_k · ( dout_i − γ_k · x_i^k ) · k · x_i^(k-1)    where:
//       α_k = f_k · inv_rms_k                          (per-row scalar)
//       γ_k = inv_rms_k² · g_k · (1/N)                 (per-row scalar)
//
// The α/γ form is algebraically equivalent to the previous coeff form, but was
// experimentally more stable for dL/dx because the shared α factor is applied
// after the cancellation in `dout − γ·x^k`.
//
// Algorithm (per row, executed on each Tensix core):
//
//   Pass 1 — accumulate_all_sums_for_row():
//     Read x and dout tiles once from DRAM. For each tile, compute and
//     L1-accumulate 7 running sums across 2 register cycles:
//       Σx², Σx⁴, Σx⁶, Σ(x·dout), Σ(x²·dout), Σ(x³·dout), Σdout
//
//   Reduce phase:
//     reduce_sum_to_inv_rms: row-reduce each power sum → inv_rms scalar tiles
//     reduce_sum_to_scalar:  row-reduce mixed sums → g_k scalar tiles
//
//   compute_dw_and_gamma: from each g_k and inv_rms_k, produce in one DST cycle
//     dw_k = g_k · inv_rms_k                    (partial weight gradient)
//     γ_k  = inv_rms_k² · g_k · (1/N)           (correction for grad_x)
//
//   prepare_weighted_inv_rms_for_row: fold weights into inv_rms, producing
//     α_k = f_k · inv_rms_k as Float32 tiles — reused across every Pass-2 tile.
//
//   Pass 2 — emit_output_for_row():
//     Re-read x and dout from DRAM. For each tile compute per-element grad_x
//     as the sum of three terms α_k · (dout − γ_k · x^k) · k·x^(k-1).
//
//   Finalize:
//     reduce_sum_to_scalar on accumulated Σdout → dL/db scalar
//     emit_packed_partials_for_row: pack [dw0, dw1, dw2, db] into 4 tiles
// ============================================================================

#include <algorithm>
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

inline uint32_t get_num_inner() {
    return get_arg_val<uint32_t>(0);
}

inline uint32_t get_eps_fp32_bits() {
    return get_arg_val<uint32_t>(1);
}

// --- Circular buffer indices (must match program factory and reader/writer) ---

constexpr auto cb_x = tt::CBIndex::c_0;
constexpr auto cb_dout = tt::CBIndex::c_1;
constexpr auto cb_db_acc = tt::CBIndex::c_2;
constexpr auto cb_scaler = tt::CBIndex::c_3;
constexpr auto cb_matmul_reduce = tt::CBIndex::c_4;
constexpr auto cb_one = tt::CBIndex::c_5;
constexpr auto cb_w0 = tt::CBIndex::c_6;
constexpr auto cb_w1 = tt::CBIndex::c_7;
constexpr auto cb_w2 = tt::CBIndex::c_8;
constexpr auto cb_sum_x2 = tt::CBIndex::c_9;
constexpr auto cb_sum_x4 = tt::CBIndex::c_10;
constexpr auto cb_sum_x6 = tt::CBIndex::c_11;
constexpr auto cb_sum_xdout = tt::CBIndex::c_12;
constexpr auto cb_sum_x2dout = tt::CBIndex::c_13;
constexpr auto cb_sum_x3dout = tt::CBIndex::c_14;
constexpr auto cb_inv_rms_x = tt::CBIndex::c_15;
constexpr auto cb_inv_rms_x2 = tt::CBIndex::c_16;
constexpr auto cb_inv_rms_x3 = tt::CBIndex::c_17;
// γ_k = inv_rms_k² · g_k · (1/N), used inside the Pass-2 cancellation
// `dout − γ_k · x^k`.
constexpr auto cb_gamma_1 = tt::CBIndex::c_18;
constexpr auto cb_gamma_2 = tt::CBIndex::c_19;
constexpr auto cb_gamma_3 = tt::CBIndex::c_20;
constexpr auto cb_output = tt::CBIndex::c_21;
constexpr auto cb_packed_partials_output = tt::CBIndex::c_22;
// Preweighted inv_rms used by Pass-2 emit_output_for_row():
//   c_24 = w2 * inv_rms_x    (linear branch)
//   c_25 = w1 * inv_rms_x2   (quadratic branch)
//   c_26 = w0 * inv_rms_x3   (cubic branch)
//
// Stored as Float32 because these row scalars are reused across every Pass-2 tile.
constexpr auto cb_weighted_inv_rms_x = tt::CBIndex::c_24;
constexpr auto cb_weighted_inv_rms_x2 = tt::CBIndex::c_25;
constexpr auto cb_weighted_inv_rms_x3 = tt::CBIndex::c_26;
constexpr uint32_t packed_partials_tiles_per_row = 4U;

// --- Tile load / broadcast helpers ---

// Broadcast a COL-scalar tile from a CB into a DEST register (init + bcast).
inline void bcast_col_to_reg(const uint32_t cb_src, const uint32_t reg_dst) {
    unary_bcast_init<BroadcastType::COL>(cb_src, cb_src);
    unary_bcast<BroadcastType::COL>(cb_src, 0, reg_dst);
}

// Copy a tile at index tile_idx from a CB into a DEST register (init + copy).
inline void copy_tile_to_reg(const uint32_t cb_src, const uint32_t tile_idx, const uint32_t reg_dst) {
    copy_tile_init(cb_src);
    copy_tile(cb_src, tile_idx, reg_dst);
}

// Copy the first (scalar) tile from a CB into a DEST register.
inline void copy_scalar_tile_to_reg(const uint32_t cb_src, const uint32_t reg_dst) {
    copy_tile_to_reg(cb_src, 0U, reg_dst);
}

inline void row_reduce_sum_to_reg(const uint32_t cb_sum, const uint32_t reg_dst) {
    reconfig_data_format(cb_matmul_reduce, cb_sum);
    matmul_init(cb_sum, cb_matmul_reduce, 0);
    matmul_tiles(cb_sum, cb_matmul_reduce, 0, 0, reg_dst);
}

inline bool use_one_block_precision_path() {
    return get_num_inner() <= block_size;
}

// Load the preweighted-inv_rms tile (α_k = f_k · inv_rms_k, tile-broadcast) into a DST
// register for use across the current Pass-2 tile.
inline void load_alpha_tile(const uint32_t cb_alpha, const uint32_t reg_dst) {
    reconfig_data_format(cb_alpha, cb_alpha);
    copy_tile_to_dst_init_short(cb_alpha);
    copy_tile(cb_alpha, 0U, reg_dst);
}

// --- Reduction helpers ---

// Row-reduce a per-tile sum and convert to 1/rms:
//   result = 1 / sqrt( row_reduce(cb_sum) · scaler + eps )
// eps is applied via add_unary_tile (SFPU scalar add), avoiding the need for an eps tile in L1.
// Consumes cb_sum (pop), pushes one tile to cb_inv_rms.
void reduce_sum_to_inv_rms(const uint32_t cb_sum, const uint32_t cb_inv_rms) {
    cb_wait_front(cb_sum, onetile);
    cb_wait_front(cb_scaler, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_acc = 0U;

    if (use_one_block_precision_path()) {
        cb_wait_front(cb_matmul_reduce, onetile);
        constexpr uint32_t reg_scaler = 1U;
        row_reduce_sum_to_reg(cb_sum, reg_acc);
        copy_scalar_tile_to_reg(cb_scaler, reg_scaler);
        mul_binary_tile_init();
        mul_binary_tile(reg_acc, reg_scaler, reg_acc);
    } else {
        reconfig_data_format(cb_scaler, cb_sum);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, cb_inv_rms);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, 0, 0, reg_acc);
        reduce_uninit();
    }

    binop_with_scalar_tile_init();
    add_unary_tile(reg_acc, get_eps_fp32_bits());

    sqrt_tile_init();
    sqrt_tile(reg_acc);
    recip_tile_init<false>();
    recip_tile<false>(reg_acc);

    tile_regs_commit();
    pack_and_push(reg_acc, cb_inv_rms);
    cb_pop_front(cb_sum, onetile);
}

// Row-reduce a sum tile to a scalar tile (multiplied by the all-ones tile).
// Consumes cb_sum (pop), pushes one tile to cb_scalar.
void reduce_sum_to_scalar(const uint32_t cb_sum, const uint32_t cb_scalar) {
    cb_wait_front(cb_sum, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_acc = 0U;

    if (use_one_block_precision_path()) {
        cb_wait_front(cb_matmul_reduce, onetile);
        row_reduce_sum_to_reg(cb_sum, reg_acc);
    } else {
        cb_wait_front(cb_one, onetile);
        reconfig_data_format(cb_one, cb_sum);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_one, cb_scalar);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_one, 0, 0, reg_acc);
        reduce_uninit();
    }

    tile_regs_commit();
    if (cb_sum == cb_scalar) {
        // In-place reduction: free the only input slot before reserve_back in pack_and_push.
        cb_pop_front(cb_sum, onetile);
        pack_and_push(reg_acc, cb_scalar);
    } else {
        pack_and_push(reg_acc, cb_scalar);
        cb_pop_front(cb_sum, onetile);
    }
}

// --- Per-row scalar computations ---

// Multiply one inv_rms tile by its matching weight scalar and push an fp32 tile that
// has valid data in every column. We broadcast inv_rms's col-0 to all columns, then
// multiply by the uniform weight tile — the result α_k = w_k · inv_rms_k is a per-row
// scalar replicated across the tile.
inline void emit_weighted_inv_rms(const uint32_t cb_inv, const uint32_t cb_w, const uint32_t cb_out) {
    constexpr uint32_t reg_inv = 0U;
    constexpr uint32_t reg_weight = 1U;

    tile_regs_acquire();
    bcast_col_to_reg(cb_inv, reg_inv);
    reconfig_data_format_srca(cb_w);
    copy_scalar_tile_to_reg(cb_w, reg_weight);
    mul_binary_tile_init();
    mul_binary_tile(reg_inv, reg_weight, reg_inv);
    tile_regs_commit();
    pack_and_push(reg_inv, cb_out);
}

// Precompute weighted inv_rms triplet once per row, folding the w_k multiplies out of the
// Pass-2 inner loop.  Writes three fp32 tiles to cb_weighted_inv_rms_{x, x2, x3}.
void prepare_weighted_inv_rms_for_row() {
    cb_wait_front(cb_inv_rms_x, onetile);
    cb_wait_front(cb_inv_rms_x2, onetile);
    cb_wait_front(cb_inv_rms_x3, onetile);

    emit_weighted_inv_rms(cb_inv_rms_x, cb_w2, cb_weighted_inv_rms_x);
    emit_weighted_inv_rms(cb_inv_rms_x2, cb_w1, cb_weighted_inv_rms_x2);
    emit_weighted_inv_rms(cb_inv_rms_x3, cb_w0, cb_weighted_inv_rms_x3);

    cb_pop_front(cb_inv_rms_x, onetile);
    cb_pop_front(cb_inv_rms_x2, onetile);
    cb_pop_front(cb_inv_rms_x3, onetile);
}

// From the reduced row-sum g_k (= Σ x^k · dout) and inv_rms_k, produce in one DST cycle:
//   dw_k = g_k · inv_rms_k                (partial weight gradient for this order)
//   γ_k  = inv_rms_k² · g_k · (1/N)       (correction coefficient for grad_x)
// Only column 0 of each output tile is meaningful; downstream consumers fan out via
// bcast<COL>. cb_g is consumed (pop) and dw is written in-place into the same slot.
// cb_inv and cb_scaler remain available for downstream callers.
//
// This replaces the previous compute_dw_and_ws + compute_coeff_tile pair. The combined
// γ formula has one fewer multiply than coeff (no f_k factor, square instead of cube)
// and avoids materialising the intermediate ws_k = g_k · w_k tile.
void compute_dw_and_gamma(const uint32_t cb_g, const uint32_t cb_inv, const uint32_t cb_gamma) {
    cb_wait_front(cb_g, onetile);
    cb_wait_front(cb_inv, onetile);
    cb_wait_front(cb_scaler, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_inv = 0U;     // bcast inv_rms across the tile (col 0 → all cols)
    constexpr uint32_t reg_g = 1U;       // g_k in col 0
    constexpr uint32_t reg_scaler = 2U;  // 1/N in col 0
    constexpr uint32_t reg_dw = 3U;      // dw_k in col 0

    bcast_col_to_reg(cb_inv, reg_inv);
    reconfig_data_format_srca(cb_g);
    copy_scalar_tile_to_reg(cb_g, reg_g);
    copy_scalar_tile_to_reg(cb_scaler, reg_scaler);

    mul_binary_tile_init();
    mul_binary_tile(reg_g, reg_inv, reg_dw);     // dw = g · inv_rms
    mul_binary_tile(reg_inv, reg_inv, reg_inv);  // inv²
    mul_binary_tile(reg_inv, reg_g, reg_g);      // inv² · g
    mul_binary_tile(reg_g, reg_scaler, reg_g);   // γ = inv² · g · scaler

    tile_regs_commit();
    cb_pop_front(cb_g, onetile);
    pack_and_push_two_tiles(reg_dw, cb_g, reg_g, cb_gamma);
}

// --- Pass 1: single-pass accumulation ---

// Accumulate all 7 sums for one tile of x and dout (2 register cycles).
// Each cycle: acquire → compute → commit → L1-pack-accumulate → release.
//   Cycle A: x, dout → x², x⁴, x⁶, dout          → accumulate Σx², Σx⁴, Σx⁶, Σdout
//   Cycle B: x, dout → x·dout, x²·dout, x³·dout  → accumulate Σ(x·dout), Σ(x²·dout), Σ(x³·dout)
inline void accumulate_all_sums_for_tile(const uint32_t block_idx, const bool first) {
    constexpr uint32_t reg_x_or_x6 = 3U;
    constexpr uint32_t reg_x2_or_x2dout = 1U;
    constexpr uint32_t reg_x4_or_xdout = 2U;
    constexpr uint32_t reg_dout_or_x3dout = 0U;

    // Cycle A: x², x⁴, x⁶, dout
    tile_regs_acquire();
    copy_tile_to_reg(cb_x, block_idx, reg_x_or_x6);  // r3 = x
    mul_binary_tile_init();
    mul_binary_tile(reg_x_or_x6, reg_x_or_x6, reg_x2_or_x2dout);           // r1 = x²
    mul_binary_tile(reg_x2_or_x2dout, reg_x2_or_x2dout, reg_x4_or_xdout);  // r2 = x⁴
    mul_binary_tile(reg_x4_or_xdout, reg_x2_or_x2dout, reg_x_or_x6);       // r3 = x⁶
    copy_tile_to_reg(cb_dout, block_idx, reg_dout_or_x3dout);              // r0 = dout
    tile_regs_commit();
    pack_four_l1_acc_tiles(
        first,
        /*reg_0=*/reg_dout_or_x3dout,
        cb_db_acc,
        /*reg_1=*/reg_x2_or_x2dout,
        cb_sum_x2,
        /*reg_2=*/reg_x4_or_xdout,
        cb_sum_x4,
        /*reg_3=*/reg_x_or_x6,
        cb_sum_x6);

    // Cycle B: x·dout, x²·dout, x³·dout
    tile_regs_acquire();
    copy_tile_to_reg(cb_x, block_idx, reg_dout_or_x3dout);   // r0 = x
    copy_tile_to_reg(cb_dout, block_idx, reg_x2_or_x2dout);  // r1 = dout
    mul_binary_tile_init();
    mul_binary_tile(reg_dout_or_x3dout, reg_x2_or_x2dout, reg_x4_or_xdout);  // r2 = x·dout
    mul_binary_tile(reg_x4_or_xdout, reg_dout_or_x3dout, reg_x_or_x6);       // r3 = x²·dout
    mul_binary_tile(reg_x_or_x6, reg_dout_or_x3dout, reg_dout_or_x3dout);    // r0 = x³·dout
    tile_regs_commit();
    pack_three_l1_acc_tiles(
        first,
        /*reg_0=*/reg_x4_or_xdout,
        cb_sum_xdout,
        /*reg_1=*/reg_x_or_x6,
        cb_sum_x2dout,
        /*reg_2=*/reg_dout_or_x3dout,
        cb_sum_x3dout);
}

// Read x and dout tiles once per row, accumulating all 7 sums via L1-pack.
// After this function, sum CBs each contain one reduced-but-unreduced tile.
void accumulate_all_sums_for_row() {
    const uint32_t num_inner = get_num_inner();

    // Reserve output slots for all 7 accumulators
    cb_reserve_back(cb_sum_x2, onetile);
    cb_reserve_back(cb_sum_x4, onetile);
    cb_reserve_back(cb_sum_x6, onetile);
    cb_reserve_back(cb_sum_xdout, onetile);
    cb_reserve_back(cb_sum_x2dout, onetile);
    cb_reserve_back(cb_sum_x3dout, onetile);
    cb_reserve_back(cb_db_acc, onetile);

    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_x, block_size);
        cb_wait_front(cb_dout, block_size);

        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            const bool first = (col == 0 && block_idx == 0);
            accumulate_all_sums_for_tile(block_idx, first);
        }

        cb_pop_front(cb_x, block_size);
        cb_pop_front(cb_dout, block_size);
    }

    // Push all accumulated sum tiles for downstream consumption
    cb_push_back(cb_sum_x2, onetile);
    cb_push_back(cb_sum_x4, onetile);
    cb_push_back(cb_sum_x6, onetile);
    cb_push_back(cb_sum_xdout, onetile);
    cb_push_back(cb_sum_x2dout, onetile);
    cb_push_back(cb_sum_x3dout, onetile);
    cb_push_back(cb_db_acc, onetile);
}

// --- Output emission ---

// Pack [dw0, dw1, dw2, db] as 4 consecutive tiles into the packed_partials output CB.
// These are later reduced on-device by the host wrapper to produce final dL/dw and dL/db.
void emit_packed_partials_for_row() {
    cb_wait_front(cb_sum_xdout, onetile);
    cb_wait_front(cb_sum_x2dout, onetile);
    cb_wait_front(cb_sum_x3dout, onetile);
    cb_wait_front(cb_inv_rms_x, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg0 = 0U;
    constexpr uint32_t reg1 = 1U;
    constexpr uint32_t reg2 = 2U;
    constexpr uint32_t reg3 = 3U;

    reconfig_data_format_srca(cb_sum_x3dout);
    copy_scalar_tile_to_reg(cb_sum_x3dout, reg0);
    copy_scalar_tile_to_reg(cb_sum_x2dout, reg1);
    copy_scalar_tile_to_reg(cb_sum_xdout, reg2);
    reconfig_data_format_srca(cb_inv_rms_x);
    copy_scalar_tile_to_reg(cb_inv_rms_x, reg3);

    tile_regs_commit();
    pack_and_push_block(cb_packed_partials_output, packed_partials_tiles_per_row);

    cb_pop_front(cb_sum_xdout, onetile);
    cb_pop_front(cb_sum_x2dout, onetile);
    cb_pop_front(cb_sum_x3dout, onetile);
    cb_pop_front(cb_inv_rms_x, onetile);
}

// Compute per-element dL/dx for one row (Pass 2).
//
// For each element, dL/dx is the sum of three order-k terms (k=1,2,3):
//
//   term_k = α_k · ( dout − γ_k · x^k ) · k · x^(k-1)
//
// where α_k = w_k · inv_rms_k  (preloaded once per row in cb_weighted_inv_rms_*)
//   and γ_k = inv_rms_k² · g_k · (1/N)  (preloaded in cb_gamma_*).
//
// The k=1 term has outer factor 1·x⁰ = 1 (no outer multiply).
// The k=2 term multiplies by 2x.
// The k=3 term multiplies by 3x².
//
// Per-order schedule (each order acquires no extra DST register beyond reg_acc + 3 working
// regs): bcast γ → load x → x² (and x³ for k=3) → multiply by γ → load dout → subtract →
// load α (FPU SrcA copy) → multiply by α → multiply by k·x^(k-1) → accumulate into reg_acc.
// All α loads use FPU MVDBGA from Default-mode CBs, so SrcA DVALID flow is clean throughout.
void emit_output_for_row() {
    const uint32_t num_inner = get_num_inner();
    constexpr uint32_t reg_acc = 0U;
    constexpr uint32_t reg0 = 1U;
    constexpr uint32_t reg1 = 2U;
    constexpr uint32_t reg2 = 3U;

    binary_op_init_common(cb_x, cb_x, cb_output);

    cb_wait_front(cb_weighted_inv_rms_x, onetile);
    cb_wait_front(cb_weighted_inv_rms_x2, onetile);
    cb_wait_front(cb_weighted_inv_rms_x3, onetile);
    cb_wait_front(cb_gamma_1, onetile);
    cb_wait_front(cb_gamma_2, onetile);
    cb_wait_front(cb_gamma_3, onetile);

    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_x, block_size);
        cb_wait_front(cb_dout, block_size);
        cb_reserve_back(cb_output, block_size);

        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            tile_regs_acquire();

            // ---- Order 1 (k=1): term_1 = α_1 · (dout − γ_1 · x) ----
            bcast_col_to_reg(cb_gamma_1, reg0);       // γ_1 broadcast
            copy_tile_to_reg(cb_x, block_idx, reg1);  // x
            mul_binary_tile_init();
            mul_binary_tile(reg0, reg1, reg0);           // γ_1 · x
            copy_tile_to_reg(cb_dout, block_idx, reg1);  // dout
            sub_binary_tile_init();
            sub_binary_tile(reg1, reg0, reg0);             // dout − γ_1 · x
            load_alpha_tile(cb_weighted_inv_rms_x, reg1);  // α_1
            mul_binary_tile_init();
            mul_binary_tile(reg1, reg0, reg_acc);  // α_1 · (dout − γ_1·x)

            // ---- Order 2 (k=2): term_2 = α_2 · (dout − γ_2 · x²) · 2x ----
            bcast_col_to_reg(cb_gamma_2, reg0);       // γ_2 broadcast
            copy_tile_to_reg(cb_x, block_idx, reg1);  // x  (kept for outer 2x)
            mul_binary_tile_init();
            mul_binary_tile(reg1, reg1, reg2);           // x²
            mul_binary_tile(reg0, reg2, reg2);           // γ_2 · x²
            copy_tile_to_reg(cb_dout, block_idx, reg0);  // dout
            sub_binary_tile_init();
            sub_binary_tile(reg0, reg2, reg0);              // dout − γ_2 · x²
            load_alpha_tile(cb_weighted_inv_rms_x2, reg2);  // α_2
            mul_binary_tile_init();
            mul_binary_tile(reg2, reg0, reg0);  // α_2 · (...)
            add_binary_tile_init();
            add_binary_tile(reg1, reg1, reg1);  // 2x
            mul_binary_tile_init();
            mul_binary_tile(reg0, reg1, reg0);  // · 2x
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg0, reg_acc);  // accumulate

            // ---- Order 3 (k=3): term_3 = α_3 · (dout − γ_3 · x³) · 3x² ----
            bcast_col_to_reg(cb_gamma_3, reg0);       // γ_3 broadcast
            copy_tile_to_reg(cb_x, block_idx, reg1);  // x
            mul_binary_tile_init();
            mul_binary_tile(reg1, reg1, reg2);           // x²  (kept for outer 3x²)
            mul_binary_tile(reg1, reg2, reg1);           // x³ = x · x²
            mul_binary_tile(reg0, reg1, reg1);           // γ_3 · x³
            copy_tile_to_reg(cb_dout, block_idx, reg0);  // dout
            sub_binary_tile_init();
            sub_binary_tile(reg0, reg1, reg0);              // dout − γ_3 · x³
            load_alpha_tile(cb_weighted_inv_rms_x3, reg1);  // α_3
            mul_binary_tile_init();
            mul_binary_tile(reg1, reg0, reg0);  // α_3 · (...)
            add_binary_tile_init();
            add_binary_tile(reg2, reg2, reg1);  // 2x²
            add_binary_tile(reg2, reg1, reg1);  // 3x²
            mul_binary_tile_init();
            mul_binary_tile(reg0, reg1, reg0);  // · 3x²
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg0, reg_acc);  // accumulate

            tile_regs_commit();
            pack_l1_acc_block(cb_output, true, 1U, block_idx);
        }

        cb_push_back(cb_output, block_size);
        cb_pop_front(cb_x, block_size);
        cb_pop_front(cb_dout, block_size);
    }

    cb_pop_front(cb_weighted_inv_rms_x, onetile);
    cb_pop_front(cb_weighted_inv_rms_x2, onetile);
    cb_pop_front(cb_weighted_inv_rms_x3, onetile);
    cb_pop_front(cb_gamma_1, onetile);
    cb_pop_front(cb_gamma_2, onetile);
    cb_pop_front(cb_gamma_3, onetile);
}

// --- Main entry point ---

void kernel_main() {
    // Wait for constant tiles generated by the reader kernel
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);
    cb_wait_front(cb_one, onetile);
    cb_wait_front(cb_w0, onetile);
    cb_wait_front(cb_w1, onetile);
    cb_wait_front(cb_w2, onetile);

    init_sfpu(cb_x, cb_output);
    binary_op_init_common(cb_x, cb_x, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        (void)row;

        // Pass 1: single read of x + dout, L1-accumulate all 7 sums
        accumulate_all_sums_for_row();

        // Reduce power sums → inv_rms scalars
        reduce_sum_to_inv_rms(cb_sum_x2, cb_inv_rms_x);
        reduce_sum_to_inv_rms(cb_sum_x4, cb_inv_rms_x2);
        reduce_sum_to_inv_rms(cb_sum_x6, cb_inv_rms_x3);

        // Reduce mixed sums → scalar tiles
        reduce_sum_to_scalar(cb_sum_xdout, cb_sum_xdout);
        reduce_sum_to_scalar(cb_sum_x2dout, cb_sum_x2dout);
        reduce_sum_to_scalar(cb_sum_x3dout, cb_sum_x3dout);

        // Partial dw_k = g_k · inv_rms_k (in-place into cb_sum_x*dout slot) and γ_k = inv² · g · (1/N).
        compute_dw_and_gamma(cb_sum_xdout, cb_inv_rms_x, cb_gamma_1);
        compute_dw_and_gamma(cb_sum_x2dout, cb_inv_rms_x2, cb_gamma_2);
        compute_dw_and_gamma(cb_sum_x3dout, cb_inv_rms_x3, cb_gamma_3);

        // Fold w0/w1/w2 into inv_rms once per row so Pass-2 drops 3 muls + 3 CB reads per output tile.
        prepare_weighted_inv_rms_for_row();

        // Pass 2: re-read x + dout, emit per-element grad_x
        emit_output_for_row();

        // Reduce accumulated Σdout → dL/db scalar (reuses cb_inv_rms_x slot)
        reduce_sum_to_scalar(cb_db_acc, cb_inv_rms_x);

        // Pack [dw0, dw1, dw2, db] into 4 tiles for on-device final reduction in host wrapper
        emit_packed_partials_for_row();
    }

    // Release constant tiles
    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_matmul_reduce, onetile);
    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_w0, onetile);
    cb_pop_front(cb_w1, onetile);
    cb_pop_front(cb_w2, onetile);
}
