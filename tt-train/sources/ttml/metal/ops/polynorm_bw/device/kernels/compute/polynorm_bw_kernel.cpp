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
// Algorithm (per row, executed on each Tensix core):
//
//   Pass 1 — accumulate_all_sums_for_row():
//     Read x and dout tiles once from DRAM. For each tile, compute and
//     L1-accumulate 7 running sums across 3 register cycles:
//       Σx², Σx⁴, Σx⁶, Σ(x·dout), Σ(x²·dout), Σ(x³·dout), Σdout
//
//   Reduce phase:
//     reduce_sum_to_inv_rms: row-reduce each power sum → 1/rms scalar tiles
//       Σx²  → inv_rms_x   = 1/√(mean(x²) + ε)
//       Σx⁴  → inv_rms_x2  = 1/√(mean(x⁴) + ε)
//       Σx⁶  → inv_rms_x3  = 1/√(mean(x⁶) + ε)
//     reduce_sum_to_scalar: row-reduce mixed sums → scalar tiles
//       Σ(x·dout) → scalar, Σ(x²·dout) → scalar, Σ(x³·dout) → scalar
//
//   compute_dw_and_ws: from each reduced scalar and inv_rms, produce
//     dw_k = scalar_k · inv_rms_k    (partial weight gradient)
//     ws_k = scalar_k · w_k          (for correction coefficients)
//
//   compute_coeff_tile: correction coefficient for grad_x:
//     coeff_k = inv_rms_k³ · ws_k · (1/N)
//
//   Pass 2 — emit_output_for_row():
//     Re-read x and dout from DRAM. For each tile compute per-element grad_x
//     as the sum of three order terms (see function docstring).
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
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

inline uint32_t get_num_inner() {
    return get_arg_val<uint32_t>(0);
}

// --- Circular buffer indices (must match program factory and reader/writer) ---

constexpr auto cb_x = tt::CBIndex::c_0;
constexpr auto cb_dout = tt::CBIndex::c_1;
constexpr auto cb_db_acc = tt::CBIndex::c_2;
constexpr auto cb_scaler = tt::CBIndex::c_3;
constexpr auto cb_eps = tt::CBIndex::c_4;
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
constexpr auto cb_coeff_1 = tt::CBIndex::c_18;
constexpr auto cb_coeff_2 = tt::CBIndex::c_19;
constexpr auto cb_coeff_3 = tt::CBIndex::c_20;
constexpr auto cb_output = tt::CBIndex::c_21;
constexpr auto cb_packed_partials_output = tt::CBIndex::c_22;

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

// Compute dout · weight · inv_rms into reg_dst.
// Clobbers registers 1 and 3. Leaves math pipeline in mul mode.
inline void weighted_dout_to_reg(
    const uint32_t block_idx, const uint32_t cb_w, const uint32_t cb_inv, const uint32_t reg_dst) {
    constexpr uint32_t r1 = 1U;
    constexpr uint32_t r_tmp = 3U;

    copy_tile_to_reg(cb_dout, block_idx, r1);
    copy_scalar_tile_to_reg(cb_w, r_tmp);
    mul_binary_tile_init();
    mul_binary_tile(r1, r_tmp, r_tmp);
    bcast_col_to_reg(cb_inv, r1);
    mul_binary_tile(r_tmp, r1, reg_dst);
}

// --- Reduction helpers ---

// Row-reduce a per-tile sum and convert to 1/rms:
//   result = 1 / sqrt( row_reduce(cb_sum) · scaler + eps )
// Consumes cb_sum (pop), pushes one tile to cb_inv_rms.
void reduce_sum_to_inv_rms(const uint32_t cb_sum, const uint32_t cb_inv_rms) {
    cb_wait_front(cb_sum, onetile);
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_acc = 0U;
    constexpr uint32_t reg_eps = 1U;

    reconfig_data_format(cb_sum, cb_scaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, cb_inv_rms);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, 0, 0, reg_acc);
    reduce_uninit();

    reconfig_data_format_srca(cb_eps);
    copy_scalar_tile_to_reg(cb_eps, reg_eps);
    add_binary_tile_init();
    add_binary_tile(reg_acc, reg_eps, reg_acc);

    sqrt_tile_init();
    sqrt_tile(reg_acc);
    recip_tile_init();
    recip_tile(reg_acc);

    tile_regs_commit();
    pack_and_push(reg_acc, cb_inv_rms);
    cb_pop_front(cb_sum, onetile);
}

// Row-reduce a sum tile to a scalar tile (multiplied by the all-ones tile).
// Consumes cb_sum (pop), pushes one tile to cb_scalar.
void reduce_sum_to_scalar(const uint32_t cb_sum, const uint32_t cb_scalar) {
    cb_wait_front(cb_sum, onetile);
    cb_wait_front(cb_one, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_acc = 0U;

    reconfig_data_format(cb_sum, cb_one);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_one, cb_scalar);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_one, 0, 0, reg_acc);
    reduce_uninit();

    tile_regs_commit();
    pack_and_push(reg_acc, cb_scalar);
    cb_pop_front(cb_sum, onetile);
}

// --- Per-row scalar computations ---

// Compute the correction coefficient for one order's grad_x contribution:
//   coeff = inv³ · ws · scaler
// where ws = (Σ x^n · dout) · w_k (the weight-scaled sum from compute_dw_and_ws).
// Consumes cb_scale (pop); cb_inv and cb_scaler remain available.
void compute_coeff_tile(const uint32_t cb_inv, const uint32_t cb_scale, const uint32_t cb_coeff) {
    cb_wait_front(cb_inv, onetile);
    cb_wait_front(cb_scale, onetile);
    cb_wait_front(cb_scaler, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_inv_sq = 0U;
    constexpr uint32_t reg_scale = 1U;
    constexpr uint32_t reg_scaler = 2U;
    constexpr uint32_t reg_inv = 3U;

    unary_bcast_init<BroadcastType::COL>(cb_inv, cb_inv);
    unary_bcast<BroadcastType::COL>(cb_inv, 0, reg_inv);
    reconfigure_unary_bcast<BroadcastType::COL, BroadcastType::COL>(cb_inv, cb_scale, cb_inv, cb_inv);
    unary_bcast<BroadcastType::COL>(cb_scale, 0, reg_scale);
    copy_scalar_tile_to_reg(cb_scaler, reg_scaler);

    mul_binary_tile_init();
    mul_binary_tile(reg_inv, reg_inv, reg_inv_sq);      // inv²
    mul_binary_tile(reg_inv_sq, reg_scale, reg_scale);  // inv² · ws
    mul_binary_tile(reg_scale, reg_inv, reg_scale);     // inv³ · ws
    mul_binary_tile(reg_scale, reg_scaler, reg_scale);  // inv³ · ws · scaler

    tile_regs_commit();
    pack_and_push(reg_scale, cb_coeff);
    cb_pop_front(cb_scale, onetile);
}

// From a reduced scalar and inv_rms, produce two outputs:
//   dw  = scalar · inv_rms   (partial weight gradient for this order)
//   ws  = scalar · weight    (weight-scaled sum, used by compute_coeff_tile)
// Consumes cb_scalar (pop); cb_inv and cb_weight remain available.
void compute_dw_and_ws(
    const uint32_t cb_scalar,
    const uint32_t cb_inv,
    const uint32_t cb_weight,
    const uint32_t cb_dw_out,
    const uint32_t cb_ws_out) {
    cb_wait_front(cb_scalar, onetile);
    cb_wait_front(cb_inv, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_s = 0U;
    constexpr uint32_t reg_inv = 1U;
    constexpr uint32_t reg_w = 2U;
    constexpr uint32_t reg_dw = 3U;

    bcast_col_to_reg(cb_inv, reg_inv);
    reconfig_data_format_srca(cb_scalar);
    copy_scalar_tile_to_reg(cb_scalar, reg_s);
    copy_scalar_tile_to_reg(cb_weight, reg_w);

    mul_binary_tile_init();
    mul_binary_tile(reg_s, reg_inv, reg_dw);  // dw = scalar · inv_rms
    mul_binary_tile(reg_s, reg_w, reg_s);     // ws = scalar · weight

    tile_regs_commit();
    cb_pop_front(cb_scalar, onetile);
    pack_and_push_two_tiles(reg_dw, cb_dw_out, reg_s, cb_ws_out);
}

// --- Pass 1: single-pass accumulation ---

// Accumulate all 7 sums for one tile of x and dout (3 register cycles).
// Each cycle: acquire → compute → commit → L1-pack-accumulate → release.
//   Cycle 1: x → x², x⁴                  → accumulate Σx², Σx⁴
//   Cycle 2: x, dout → x·dout, x²·dout   → accumulate Σ(x·dout), Σ(x²·dout), Σdout
//   Cycle 3: x, dout → x⁶, x³·dout       → accumulate Σx⁶, Σ(x³·dout)
inline void accumulate_all_sums_for_tile(const uint32_t block_idx, const bool first) {
    // Cycle 1: x² and x⁴
    tile_regs_acquire();
    copy_tile_to_reg(cb_x, block_idx, 0);
    mul_binary_tile_init();
    mul_binary_tile(0, 0, 1);  // r1 = x²
    mul_binary_tile(1, 1, 2);  // r2 = x⁴
    tile_regs_commit();
    pack_two_l1_acc_tiles(first, /*reg_0=*/1U, cb_sum_x2, /*reg_1=*/2U, cb_sum_x4);

    // Cycle 2: x·dout, x²·dout, dout (for db)
    tile_regs_acquire();
    copy_tile_to_reg(cb_x, block_idx, 0);
    copy_tile_to_reg(cb_dout, block_idx, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 2);  // r2 = x·dout
    mul_binary_tile(0, 0, 0);  // r0 = x²
    mul_binary_tile(0, 1, 3);  // r3 = x²·dout  (r1 = dout unchanged)
    tile_regs_commit();
    pack_three_l1_acc_tiles(
        first,
        /*reg_0=*/2U,
        cb_sum_xdout,
        /*reg_1=*/3U,
        cb_sum_x2dout,
        /*reg_2=*/1U,  // dout for db
        cb_db_acc);

    // Cycle 3: x⁶, x³·dout
    tile_regs_acquire();
    copy_tile_to_reg(cb_x, block_idx, 0);
    mul_binary_tile_init();
    mul_binary_tile(0, 0, 1);                 // r1 = x²
    mul_binary_tile(1, 0, 2);                 // r2 = x³
    mul_binary_tile(2, 2, 3);                 // r3 = x⁶
    copy_tile_to_reg(cb_dout, block_idx, 0);  // r0 = dout (x overwritten)
    mul_binary_tile_init();
    mul_binary_tile(2, 0, 1);  // r1 = x³·dout
    tile_regs_commit();
    pack_two_l1_acc_tiles(first, /*reg_0=*/3U, cb_sum_x6, /*reg_1=*/1U, cb_sum_x3dout);
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
// These are later reduced on the host to produce the final dL/dw and dL/db.
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
    pack_and_push_block(cb_packed_partials_output, block_size);

    cb_pop_front(cb_sum_xdout, onetile);
    cb_pop_front(cb_sum_x2dout, onetile);
    cb_pop_front(cb_sum_x3dout, onetile);
    cb_pop_front(cb_inv_rms_x, onetile);
}

// Compute per-element dL/dx for one row (Pass 2).
//
// For each element, dL/dx is the sum of three order-k terms (k=1,2,3):
//
//   term_k = (dout·w_k·inv_rms_k − x^k·coeff_k) · k·x^(k−1)
//
// where inv_rms_k = 1/√(mean(x^(2k)) + ε)  and  coeff_k = inv_rms_k³ · ws_k · (1/N).
//
// The w2 (order 1) term has derivative factor 1 (no multiply by x^0).
// The w1 (order 2) term multiplies by 2x.
// The w0 (order 3) term multiplies by 3x².
void emit_output_for_row() {
    const uint32_t num_inner = get_num_inner();
    constexpr uint32_t reg_acc = 0U;
    constexpr uint32_t reg1 = 1U;
    constexpr uint32_t reg0 = 2U;
    constexpr uint32_t reg_tmp = 3U;

    binary_op_init_common(cb_x, cb_x, cb_output);

    // Wait for all per-row scalar/coeff tiles computed in the reduce phase
    cb_wait_front(cb_inv_rms_x, onetile);
    cb_wait_front(cb_inv_rms_x2, onetile);
    cb_wait_front(cb_inv_rms_x3, onetile);
    cb_wait_front(cb_coeff_1, onetile);
    cb_wait_front(cb_coeff_2, onetile);
    cb_wait_front(cb_coeff_3, onetile);

    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_x, block_size);
        cb_wait_front(cb_dout, block_size);
        cb_reserve_back(cb_output, block_size);

        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            tile_regs_acquire();

            // w2 term (order 1): dout·w2·inv_rms_x − x·coeff_1
            copy_tile_to_reg(cb_x, block_idx, reg0);
            weighted_dout_to_reg(block_idx, cb_w2, cb_inv_rms_x, reg_acc);
            bcast_col_to_reg(cb_coeff_1, reg1);
            mul_binary_tile(reg0, reg1, reg_tmp);
            sub_binary_tile_init();
            sub_binary_tile(reg_acc, reg_tmp, reg_acc);

            // w1 term (order 2): (dout·w1·inv_rms_x2 − x²·coeff_2) · 2x
            weighted_dout_to_reg(block_idx, cb_w1, cb_inv_rms_x2, reg_tmp);
            copy_tile_to_reg(cb_x, block_idx, reg1);
            mul_binary_tile(reg1, reg1, reg1);  // x²
            bcast_col_to_reg(cb_coeff_2, reg0);
            mul_binary_tile(reg1, reg0, reg0);  // x²·coeff_2
            sub_binary_tile_init();
            sub_binary_tile(reg_tmp, reg0, reg_tmp);  // main − correction
            copy_tile_to_reg(cb_x, block_idx, reg1);
            add_binary_tile_init();
            add_binary_tile(reg1, reg1, reg1);  // 2x
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg1, reg_tmp);  // · 2x
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);  // accumulate

            // w0 term (order 3): (dout·w0·inv_rms_x3 − x³·coeff_3) · 3x²
            weighted_dout_to_reg(block_idx, cb_w0, cb_inv_rms_x3, reg_tmp);
            copy_tile_to_reg(cb_x, block_idx, reg1);
            mul_binary_tile(reg1, reg1, reg1);  // x²
            copy_tile_to_reg(cb_x, block_idx, reg0);
            mul_binary_tile(reg1, reg0, reg0);  // x³
            bcast_col_to_reg(cb_coeff_3, reg1);
            mul_binary_tile(reg0, reg1, reg0);  // x³·coeff_3
            sub_binary_tile_init();
            sub_binary_tile(reg_tmp, reg0, reg_tmp);  // main − correction
            copy_tile_to_reg(cb_x, block_idx, reg1);
            mul_binary_tile_init();
            mul_binary_tile(reg1, reg1, reg1);  // x²
            add_binary_tile_init();
            add_binary_tile(reg1, reg1, reg0);  // 2x²
            add_binary_tile(reg0, reg1, reg1);  // 3x²
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg1, reg_tmp);  // · 3x²
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);  // accumulate

            tile_regs_commit();
            pack_l1_acc_block(cb_output, true, 1U, block_idx);
        }

        cb_push_back(cb_output, block_size);
        cb_pop_front(cb_x, block_size);
        cb_pop_front(cb_dout, block_size);
    }

    cb_pop_front(cb_inv_rms_x, onetile);
    cb_pop_front(cb_inv_rms_x2, onetile);
    cb_pop_front(cb_inv_rms_x3, onetile);
    cb_pop_front(cb_coeff_1, onetile);
    cb_pop_front(cb_coeff_2, onetile);
    cb_pop_front(cb_coeff_3, onetile);
}

// --- Main entry point ---

void kernel_main() {
    // Wait for constant tiles generated by the reader kernel
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);
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

        // Partial weight gradients (dw) and weight-scaled sums (ws)
        compute_dw_and_ws(cb_sum_xdout, cb_inv_rms_x, cb_w2, cb_sum_xdout, cb_sum_x2);
        compute_dw_and_ws(cb_sum_x2dout, cb_inv_rms_x2, cb_w1, cb_sum_x2dout, cb_sum_x4);
        compute_dw_and_ws(cb_sum_x3dout, cb_inv_rms_x3, cb_w0, cb_sum_x3dout, cb_sum_x6);

        // Correction coefficients for grad_x
        compute_coeff_tile(cb_inv_rms_x, cb_sum_x2, cb_coeff_1);
        compute_coeff_tile(cb_inv_rms_x2, cb_sum_x4, cb_coeff_2);
        compute_coeff_tile(cb_inv_rms_x3, cb_sum_x6, cb_coeff_3);

        // Pass 2: re-read x + dout, emit per-element grad_x
        emit_output_for_row();

        // Reduce accumulated Σdout → dL/db scalar (reuses cb_inv_rms_x slot)
        reduce_sum_to_scalar(cb_db_acc, cb_inv_rms_x);

        // Pack [dw0, dw1, dw2, db] into 4 tiles for host-side final reduction
        emit_packed_partials_for_row();
    }

    // Release constant tiles
    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_eps, onetile);
    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_w0, onetile);
    cb_pop_front(cb_w1, onetile);
    cb_pop_front(cb_w2, onetile);
}
