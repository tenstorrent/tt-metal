// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t mask_w = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
constexpr uint32_t num_external_tiles = get_compile_time_arg_val(4);
constexpr uint32_t Ht = get_compile_time_arg_val(5);

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_6;
constexpr uint32_t cb_zero_idx = tt::CBIndex::c_7;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_8;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_9;
// CBs with intermediate computations
constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_10;
constexpr uint32_t cb_scale_idx = tt::CBIndex::c_11;
constexpr uint32_t cb_external_sums_idx = tt::CBIndex::c_12;
constexpr uint32_t cb_mask_h_idx = tt::CBIndex::c_13;

// LocalSum emits the reduced scale itself, through the dL_da output CB.
#ifdef LOCAL_SUM
constexpr uint32_t cb_reduced_scale_idx = cb_dL_da_idx;
#else
constexpr uint32_t cb_reduced_scale_idx = cb_scale_idx;
#endif

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

#ifdef DO_MASK_H
constexpr bool do_mask_h = true;
#else
constexpr bool do_mask_h = false;
#endif

// With fp32 DEST accumulation one DEST half holds four tiles. The block helpers below keep the block's results in
// tiles 0..block_size-1 and use the remaining tiles as scratch, so a block is at most two tiles wide.
static_assert(block_size <= 2U, "block helpers need two scratch DST tiles");
static_assert(DST_ACCUM_MODE, "dL_da and ExternalScale sums rely on the FPU multiply accumulating into fp32 DEST");

// ============================================================================
// Register-only helpers. They work on a block of tiles at a time and issue every
// init once per block instead of once per tile.
// ============================================================================

// Index of the block's block_idx-th tile in the row CBs: the whole row sits in L1, or only the current block.
inline uint32_t row_tile_idx(const uint32_t col, const uint32_t block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
    return col + block_idx;
#else
    return block_idx;
#endif
}

// The FPU multiply accumulates into DEST in fp32 mode, so every register that collects a product starts from an
// explicit zero tile.
inline void zero_registers(const uint32_t first_register, const uint32_t count) {
    reconfig_data_format_srca(cb_zero_idx);
    copy_init(cb_zero_idx);
    for (uint32_t i = 0; i < count; ++i) {
        copy_tile(cb_zero_idx, /* tile_idx */ 0, first_register + i);
    }
}

// Multiplies DEST tiles 0..count-1 by 1/rms_a, which is copied once into recip_register.
inline void scale_registers_by_recip(const uint32_t count, const uint32_t recip_register) {
    reconfig_data_format_srca(cb_recip_rms_a_bcasted_idx);
    copy_init(cb_recip_rms_a_bcasted_idx);
    copy_tile(cb_recip_rms_a_bcasted_idx, /* tile_idx */ 0, recip_register);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < count; ++i) {
        mul_binary_tile(i, recip_register, i);
    }
}

// Adds sum over the block of a * gamma * dL_dout to acc_register. Work registers are acc_register + 1 .. and the
// last DEST tile is scratch.
inline void accumulate_local_sum_block(const uint32_t col, const uint32_t acc_register) {
    const uint32_t first_work_register = acc_register + 1U;
    const uint32_t scratch_register = 3U;

    // work_t = dL_dout_t * gamma_t (gamma broadcast over rows)
    zero_registers(first_work_register, block_size);
    reconfig_data_format(cb_dL_out_idx, cb_gamma_idx);
    mul_bcast_rows_init(cb_dL_out_idx, cb_gamma_idx);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        const uint32_t tile_idx = row_tile_idx(col, block_idx);
        mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, tile_idx, tile_idx, first_work_register + block_idx);
    }

    // work_t *= a_t
    reconfig_data_format_srca(cb_input_idx);
    copy_init(cb_input_idx);
    mul_binary_tile_init();
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        copy_tile(cb_input_idx, row_tile_idx(col, block_idx), scratch_register);
        mul_binary_tile(first_work_register + block_idx, scratch_register, first_work_register + block_idx);
    }

    // Columns past the logical width live in the last tile of the row and must not reach the sum.
    if constexpr (do_mask_w) {
        if (col + block_size == Wt) {
            // mask_tile reads its mask from the register right after the data: the scratch register for a block of
            // two, the unused second work register for a block of one.
            const uint32_t data_register = first_work_register + block_size - 1U;
            const uint32_t mask_register = data_register + 1U;
            copy_init(cb_mask_w_idx);
            copy_tile(cb_mask_w_idx, /* tile_idx */ 0, mask_register);
            mask_tile_init();
            mask_tile(data_register, mask_register);
        }
    }

    add_binary_tile_init();
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        add_binary_tile(acc_register, first_work_register + block_idx, acc_register);
    }
}

#ifndef LOCAL_SUM
// Turns the reduced row sum S in scale_register into the per-row dL_da scale k = -S * recip^n / W. cb_scaler holds
// -1/W and n is the number of 1/rms_a factors still missing: 3 in Full mode (S carries none) and 2 in ExternalScale
// mode (S already carries one). The sign makes compute_dL_da_block subtract the a * k term by accumulating it.
inline void finish_dL_da_scale(const uint32_t scale_register, const uint32_t num_recip_factors) {
    const uint32_t recip_register = scale_register + 1U;
    const uint32_t scaler_register = scale_register + 2U;
    reconfig_data_format_srca(cb_recip_rms_a_bcasted_idx);
    copy_init(cb_recip_rms_a_bcasted_idx);
    copy_tile(cb_recip_rms_a_bcasted_idx, /* tile_idx */ 0, recip_register);
    copy_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, /* tile_idx */ 0, scaler_register);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < num_recip_factors; ++i) {
        mul_binary_tile(scale_register, recip_register, scale_register);
    }
    mul_binary_tile(scale_register, scaler_register, scale_register);
}

// dL_da_t = (dL_dout_t * gamma_t) / rms_a + a_t * k for the block's tiles, in DEST tiles 0..block_size-1.
// Formula: dL_da = gained_dL_dout - (scale * a) / (c * rms_a^2) with gained_dL_dout = (gamma / rms_a) * dL_dout;
// the whole per-row factor -scale / (c * rms_a^2) is the k tile in cb_scale.
inline void compute_dL_da_block(const uint32_t col) {
    const uint32_t recip_register = 3U;

    zero_registers(0, block_size);
    reconfig_data_format(cb_dL_out_idx, cb_gamma_idx);
    mul_bcast_rows_init(cb_dL_out_idx, cb_gamma_idx);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        const uint32_t tile_idx = row_tile_idx(col, block_idx);
        mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, tile_idx, tile_idx, block_idx);
    }
    scale_registers_by_recip(block_size, recip_register);

    // The FPU accumulates a_t * k onto the gained gradient; k is negative.
    reconfig_data_format(cb_input_idx, cb_scale_idx);
    mul_bcast_cols_init(cb_input_idx, cb_scale_idx);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        mul_tiles_bcast_cols(cb_input_idx, cb_scale_idx, row_tile_idx(col, block_idx), /* tile_idx */ 0, block_idx);
    }
}

// dL_dgamma_components_t = (a_t * dL_dout_t) / rms_a for count tiles starting at first_block_idx, in DEST tiles
// 0..count-1. The components are reduced over batch and sequence outside the kernel.
inline void compute_dL_dgamma_block(const uint32_t col, const uint32_t first_block_idx, const uint32_t count) {
    const uint32_t recip_register = 3U;

    zero_registers(0, count);
    reconfig_data_format(cb_input_idx, cb_dL_out_idx);
    mul_init(cb_input_idx, cb_dL_out_idx);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t tile_idx = row_tile_idx(col, first_block_idx + i);
        mul_tiles(cb_input_idx, cb_dL_out_idx, tile_idx, tile_idx, i);
    }
    scale_registers_by_recip(count, recip_register);
}
#endif

// ============================================================================
// Functions below compute final results for a pipeline stage and write them to
// circular buffers (CBs). These functions produce outputs that are consumed by
// later stages.
// ============================================================================

inline void compute_recip_rms_a_bcasted() {
    // Computes reciprocal of RMS activation (1/rms_a) and broadcasts it across columns.
    // The broadcast is the binary zero + bcast_cols(rms_a) rather than a unary bcast, which would need a per-row
    // compute_kernel_hw_startup. That is a call-once API: it resets the MATH<->PACK DEST sync while the packer may
    // still be draining the previous row.
    const uint32_t reg_rms_a = 0;
    tile_regs_acquire();
    zero_registers(reg_rms_a, 1U);
    reconfig_data_format(cb_zero_idx, cb_rms_a_idx);
    add_bcast_cols_init(cb_zero_idx, cb_rms_a_idx);
    add_tiles_bcast_cols(cb_zero_idx, cb_rms_a_idx, /* tile_idx */ 0, /* tile_idx */ 0, reg_rms_a);
    recip_tile_init();
    recip_tile(reg_rms_a);
    tile_regs_commit();
    pack_and_push(reg_rms_a, cb_recip_rms_a_bcasted_idx);
}

inline void compute_scale(const uint32_t row) {
    // Computes the row sum S = sum(a * gamma * dL_dout, dim=C), reduced along the inner dimension.
    // LocalSum emits S itself; Full turns it into the dL_da scale k = -S / (c * rms_a^3).
    const uint32_t scale_register = 0U;
    tile_regs_acquire();
    zero_registers(scale_register, 1U);
#ifdef EVERYTHING_FITS_IN_L1
    // Gamma is constant among all rows so if everything fits in L1, we can read it only once.
    if (row == 0) {
        cb_wait_front(cb_gamma_idx, Wt);
    }
    cb_wait_front(cb_dL_out_idx, Wt);
    cb_wait_front(cb_input_idx, Wt);
    for (uint32_t col = 0; col < Wt; col += block_size) {
        accumulate_local_sum_block(col, scale_register);
    }
#else
    for (uint32_t col = 0; col < Wt; col += block_size) {
        cb_wait_front(cb_gamma_idx, block_size);
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_input_idx, block_size);
        accumulate_local_sum_block(col, scale_register);
        // Pop tiles that are not needed anymore.
        cb_pop_front(cb_gamma_idx, block_size);
        cb_pop_front(cb_dL_out_idx, block_size);
        cb_pop_front(cb_input_idx, block_size);
    }
#endif
    tile_regs_commit();

    pack_and_push(scale_register, cb_scale_idx);

    cb_wait_front(cb_scale_idx, onetile);
    const uint32_t reduced_scale_register = 0U;
    tile_regs_acquire();

    // Reduce scale along the inner dimension (C).

    // NOTE: Currently, there is a bug in reduce_tile that causes precision issues. To avoid this, we use a
    // workaround of matmul with appropriate scale. Once the bug is fixed, we can switch back to reduce_tile.
    // matmul reverses operands (in0 -> SrcB, in1 -> SrcA), so reconfig in reversed (in1, in0) order.
    reconfig_data_format(cb_mat_mul_reduce, cb_scale_idx);
    matmul_init(cb_scale_idx, cb_mat_mul_reduce, 0);
    matmul_tiles(
        cb_scale_idx,
        cb_mat_mul_reduce,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        reduced_scale_register);
#ifndef LOCAL_SUM
    finish_dL_da_scale(reduced_scale_register, /* num_recip_factors */ 3U);
#endif
    tile_regs_commit();

    // Pop the non-reduced scale tile from the CB.
    cb_pop_front(cb_scale_idx, onetile);
    pack_and_push(reduced_scale_register, cb_reduced_scale_idx);
}

#ifdef EXTERNAL_SCALE
inline void compute_external_scale() {
    // Computes the dL_da scale k = -(1/rms_a) * S_total / (c * rms_a^2), where S_total is the sum over shards of
    // sum(a * gamma * dL_dout) and each shard's LocalSum output holds its row sum in column 0 of one tile. Column 0
    // is all that the bcast_cols in compute_dL_da_block reads.
    // The fp32 sums are the srcB operand of a multiply, like cb_scale in compute_dL_da_block, so each shard's sum
    // is truncated to tf32 (19-bit source registers) before the shards are added; the accumulation itself stays in
    // fp32 DEST. Shard sums that nearly cancel lose relative precision here. 1/rms_a is broadcast over
    // the whole tile, so sum_k recip * ext_k equals recip * sum_k ext_k, and since the FPU multiply accumulates
    // into DEST the sum over all shards lands in one register with no SFPU pass and no limit on the number of
    // shards.
    const uint32_t scale_register = 0U;
    cb_wait_front(cb_external_sums_idx, num_external_tiles);
    tile_regs_acquire();
    zero_registers(scale_register, 1U);
    reconfig_data_format(cb_recip_rms_a_bcasted_idx, cb_external_sums_idx);
    mul_init(cb_recip_rms_a_bcasted_idx, cb_external_sums_idx);
    for (uint32_t tile_idx = 0U; tile_idx < num_external_tiles; ++tile_idx) {
        mul_tiles(cb_recip_rms_a_bcasted_idx, cb_external_sums_idx, /* tile_idx */ 0, tile_idx, scale_register);
    }
    finish_dL_da_scale(scale_register, /* num_recip_factors */ 2U);
    tile_regs_commit();
    pack_and_push(scale_register, cb_scale_idx);
    cb_pop_front(cb_external_sums_idx, num_external_tiles);
}
#endif

void kernel_main() {
    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask_w_idx, onetile);
    }
    cb_wait_front(cb_scaler_idx, onetile);
    cb_wait_front(cb_mat_mul_reduce, onetile);
    cb_wait_front(cb_zero_idx, onetile);

    [[maybe_unused]] const uint32_t start_row = get_arg_val<uint32_t>(0);
    if constexpr (do_mask_h) {
        cb_wait_front(cb_mask_h_idx, onetile);
    }
#ifdef DGAMMA_ON_CHIP
    // Packed with L1 accumulation over all rows of this core, then pushed once.
    cb_reserve_back(cb_dL_dgamma_components, Wt);
#endif

    compute_kernel_hw_startup(cb_input_idx, cb_gamma_idx, cb_dL_da_idx);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
#ifdef LOCAL_SUM
        compute_scale(row);
#ifdef EVERYTHING_FITS_IN_L1
        cb_pop_front(cb_dL_out_idx, Wt);
        cb_pop_front(cb_input_idx, Wt);
#endif
        continue;
#else
        cb_wait_front(cb_rms_a_idx, onetile);
        // This value is constant for the whole row, so we can compute it once per row.
        compute_recip_rms_a_bcasted();
        cb_wait_front(cb_recip_rms_a_bcasted_idx, onetile);
#ifdef EXTERNAL_SCALE
        compute_external_scale();
#ifdef EVERYTHING_FITS_IN_L1
        if (row == 0) {
            cb_wait_front(cb_gamma_idx, Wt);
        }
        cb_wait_front(cb_dL_out_idx, Wt);
        cb_wait_front(cb_input_idx, Wt);
#endif
#else
        // To compute scale we must iterate over all inner dimension.
        compute_scale(row);
#endif
        // Wait for the per-row dL_da scale to be ready.
        cb_wait_front(cb_scale_idx, onetile);

        // Rows past the logical sequence length in the last tile row of each sequence must not reach dgamma.
        [[maybe_unused]] const bool mask_padded_rows = do_mask_h && (start_row + row) % Ht == Ht - 1U;
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_reserve_back(cb_dL_da_idx, block_size);
#ifndef DGAMMA_ON_CHIP
            cb_reserve_back(cb_dL_dgamma_components, block_size);
#endif
#ifndef EVERYTHING_FITS_IN_L1
            cb_wait_front(cb_gamma_idx, block_size);
            cb_wait_front(cb_dL_out_idx, block_size);
            cb_wait_front(cb_input_idx, block_size);
#endif
            tile_regs_acquire();
            compute_dL_da_block(col);
            tile_regs_commit();
            pack_and_push_block(cb_dL_da_idx, block_size);

#ifdef DGAMMA_ON_CHIP
            if (mask_padded_rows) {
                // One tile at a time, because mask_tile reads its mask from the register right after the data.
                const uint32_t dL_dg_register = 0U;
                const uint32_t mask_register = 1U;
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                    tile_regs_acquire();
                    compute_dL_dgamma_block(col, block_idx, /* count */ 1U);
                    copy_init(cb_mask_h_idx);
                    copy_tile(cb_mask_h_idx, /* tile_idx */ 0, mask_register);
                    mask_tile_init();
                    mask_tile(dL_dg_register, mask_register);
                    tile_regs_commit();
                    pack_l1_acc_block(cb_dL_dgamma_components, row == 0, onetile, col + block_idx);
                }
            } else {
                tile_regs_acquire();
                compute_dL_dgamma_block(col, /* first_block_idx */ 0U, block_size);
                tile_regs_commit();
                pack_l1_acc_block(cb_dL_dgamma_components, row == 0, block_size, col);
            }
#else
            tile_regs_acquire();
            compute_dL_dgamma_block(col, /* first_block_idx */ 0U, block_size);
            tile_regs_commit();
            pack_and_push_block(cb_dL_dgamma_components, block_size);
#endif
#ifndef EVERYTHING_FITS_IN_L1
            cb_pop_front(cb_gamma_idx, block_size);
            cb_pop_front(cb_dL_out_idx, block_size);
            cb_pop_front(cb_input_idx, block_size);
#endif
        }
#ifdef EVERYTHING_FITS_IN_L1
        cb_pop_front(cb_dL_out_idx, Wt);
        cb_pop_front(cb_input_idx, Wt);
#endif
        cb_pop_front(cb_rms_a_idx, onetile);
        cb_pop_front(cb_recip_rms_a_bcasted_idx, onetile);
        cb_pop_front(cb_scale_idx, onetile);
#endif  // LOCAL_SUM
    }
#ifdef DGAMMA_ON_CHIP
    cb_push_back(cb_dL_dgamma_components, Wt);
#endif
#ifdef EVERYTHING_FITS_IN_L1
    cb_pop_front(cb_gamma_idx, Wt);
#endif
    if constexpr (do_mask_h) {
        cb_pop_front(cb_mask_h_idx, onetile);
    }
    cb_pop_front(cb_scaler_idx, onetile);
    cb_pop_front(cb_mat_mul_reduce, onetile);
    cb_pop_front(cb_zero_idx, onetile);
    if constexpr (do_mask_w) {
        cb_pop_front(cb_mask_w_idx, onetile);
    }
}
