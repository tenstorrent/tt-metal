// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_bcast_scalar.h"
#include "ckernel_sfpu_binary_bcast.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

#ifdef TRISC_MATH
// Range checks of the partial-tile column-broadcast forms, statements under LLK_ASSERT (a no-op in release builds).
inline void _sfpu_bcast_col_mode_check_(VectorMode vector_mode) {
    LLK_ASSERT(
        vector_mode == VectorMode::RC || vector_mode == VectorMode::R,
        "sfpu_*_bcast_col: VectorMode::RC (the whole tile) or VectorMode::R (rows 0..15) only");
}
inline void _sfpu_bcast_col_band_check_(uint32_t data_band, uint32_t bcast_band) {
    LLK_ASSERT(data_band < 4 && bcast_band < 4, "sfpu_*_bcast_col_band: bands are 0..3 (rows 4 * band .. +3)");
}
#endif

// ============================================================================
// BCAST_COL: broadcast column 0 of the bcast tile across all 32 tile columns
// ============================================================================

// clang-format off
/**
 * Initialize SFPU state for {add,sub,mul}-with-column-broadcast. Must be
 * called once (before any acquire_dst / sfpu_*_bcast_col pair) to configure
 * the persistent LREG_MASK and replay-buffer slot used by the broadcast
 * helper. A single init is shared across ADD / SUB / MUL because only the
 * broadcast state (which depends on BCAST_DIM) is persisted; the binop
 * opcode is selected per call.
 *
 * On Quasar the init only checks the broadcast dim: each sfpu_*_bcast_*
 * call records its binop and lane mask before it runs, so nothing persists
 * between calls.
 *
 * Return value: None
 */
// clang-format on
ALWI void sfpu_bcast_col_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::_sfpu_binary_bcast_init_, (ckernel::BroadcastType::COL))));
}

ALWI void sfpu_sub_bcast_col_init() { sfpu_bcast_col_init(); }

ALWI void sfpu_add_bcast_col_init() { sfpu_bcast_col_init(); }

ALWI void sfpu_mul_bcast_col_init() { sfpu_bcast_col_init(); }

// clang-format off
/**
 * Apply a binop between `dst_data_idx` and the column-broadcast of column 0
 * of `dst_col_vec_idx`, entirely within DST registers. Result is written
 * in-place into `dst_data_idx`:
 *
 *     sub:  DST[dst_data_idx][r][c] -= DST[dst_col_vec_idx][r][0]
 *     add:  DST[dst_data_idx][r][c] += DST[dst_col_vec_idx][r][0]
 *     mul:  DST[dst_data_idx][r][c] *= DST[dst_col_vec_idx][r][0]
 *
 * for c in [0..31].
 *
 * Both source registers must contain FP32 data (compile with
 * fp32_dest_acc_en = true). The column-vector register is expected to carry
 * meaningful values only in column 0 of each row (e.g. a row-reduce max /
 * sum output); values in columns 1..31 are ignored. Output is FP32.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * Requires a prior call to the matching `sfpu_*_bcast_col_init` (or the
 * shared `sfpu_bcast_col_init`).
 *
 * `vector_mode` selects the rows: `VectorMode::RC` (the default) is the whole
 * tile, `VectorMode::R` is rows 0..15 only (faces 0 and 1), after which rows
 * 16..31 of the data tile are untouched and rows 16..31 of the column tile are
 * never read. Other modes are rejected (LLK_ASSERT). `VectorMode::R` is not
 * supported on Quasar: a kernel that uses it fails to build.
 *
 * Return value: None
 *
 * | Argument         | Description                                                         | Type       | Valid Range                                           | Required |
 * |------------------|---------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | dst_data_idx     | The index of the data tile in DST (result is written back in-place) | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | dst_col_vec_idx  | The index of the col-vector tile in DST (column 0 is broadcast)     | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | vector_mode      | The rows: VectorMode::RC (all) or VectorMode::R (rows 0..15)        | VectorMode | RC, R                                                 | False    |
 */
// clang-format on
// `VectorMode::None` makes `_llk_math_eltwise_sfpu_apply_vector_mode_` bypass the
// per-face loop and invoke the full-tile (or rows-0..15) bcast helper exactly once,
// with the same `_llk_math_eltwise_sfpu_start_(0)` / `_llk_math_eltwise_sfpu_done_()`
// brackets the standalone helper used to set up by hand, plus the dst-bound
// LLK_ASSERTs in `_sfpu_binary_check_`.
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_sub_bcast_col(uint32_t dst_data_idx, uint32_t dst_col_vec_idx, VectorMode vector_mode = VectorMode::RC) {
    MATH((_sfpu_bcast_col_mode_check_(vector_mode)));
    if (vector_mode == VectorMode::R) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_sfpu_binary_bcast_col_rows_0_15_,
            (ckernel::BinaryOp::SUB),
            dst_data_idx,
            dst_col_vec_idx,
            dst_data_idx,
            VectorMode::None)));
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_sfpu_binary_bcast_full_tile_,
            (ckernel::BinaryOp::SUB, ckernel::BroadcastType::COL),
            dst_data_idx,
            dst_col_vec_idx,
            dst_data_idx,
            VectorMode::None)));
    }
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_add_bcast_col(uint32_t dst_data_idx, uint32_t dst_col_vec_idx, VectorMode vector_mode = VectorMode::RC) {
    MATH((_sfpu_bcast_col_mode_check_(vector_mode)));
    if (vector_mode == VectorMode::R) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_sfpu_binary_bcast_col_rows_0_15_,
            (ckernel::BinaryOp::ADD),
            dst_data_idx,
            dst_col_vec_idx,
            dst_data_idx,
            VectorMode::None)));
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_sfpu_binary_bcast_full_tile_,
            (ckernel::BinaryOp::ADD, ckernel::BroadcastType::COL),
            dst_data_idx,
            dst_col_vec_idx,
            dst_data_idx,
            VectorMode::None)));
    }
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_mul_bcast_col(uint32_t dst_data_idx, uint32_t dst_col_vec_idx, VectorMode vector_mode = VectorMode::RC) {
    MATH((_sfpu_bcast_col_mode_check_(vector_mode)));
    if (vector_mode == VectorMode::R) {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_sfpu_binary_bcast_col_rows_0_15_,
            (ckernel::BinaryOp::MUL),
            dst_data_idx,
            dst_col_vec_idx,
            dst_data_idx,
            VectorMode::None)));
    } else {
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            _calculate_sfpu_binary_bcast_full_tile_,
            (ckernel::BinaryOp::MUL, ckernel::BroadcastType::COL),
            dst_data_idx,
            dst_col_vec_idx,
            dst_data_idx,
            VectorMode::None)));
    }
}

// clang-format off
/**
 * One 4-row band of the column broadcast: rows `4 * data_band .. +3` of
 * `dst_data_idx` (all 32 columns) get the binop with column 0 rows
 * `4 * bcast_band .. +3` of `dst_col_vec_idx`, in place:
 *
 *     sub:  DST[dst_data_idx][4*data_band + i][c] -= DST[dst_col_vec_idx][4*bcast_band + i][0]
 *     add:  DST[dst_data_idx][4*data_band + i][c] += DST[dst_col_vec_idx][4*bcast_band + i][0]
 *     mul:  DST[dst_data_idx][4*data_band + i][c] *= DST[dst_col_vec_idx][4*bcast_band + i][0]
 *
 * for i in [0..3], c in [0..31]. Nothing else of either tile is read or
 * written, so one column tile can hold several 4-row scalars for the same data
 * band (a band per scalar), and a 4x32 tiny tile (rows 0..3 of faces 0 and 1)
 * is scaled with `data_band = 0`. The rows it writes are bit-identical to
 * `sfpu_*_bcast_col` on those rows when the bands match. Both tiles are FP32;
 * the column tile's other columns and rows are never read, and column 0 of the
 * band is masked bitwise before the broadcast, so Inf or NaN elsewhere in the
 * column tile cannot leak in.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * Requires a prior call to `sfpu_bcast_col_init` (or the op-specific init).
 * Not supported on Quasar: a kernel that calls it fails to build.
 *
 * Return value: None
 *
 * | Argument         | Description                                                         | Type     | Valid Range                                           | Required |
 * |------------------|---------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_idx     | The index of the data tile in DST (result is written back in-place) | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_col_vec_idx  | The index of the col-vector tile in DST (column 0 is broadcast)     | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | data_band        | The 4-row band of the data tile: rows 4 * data_band .. +3           | uint32_t | 0 to 3                                                | True     |
 * | bcast_band       | The 4-row band of the column tile: column 0 rows 4 * bcast_band ..  | uint32_t | 0 to 3                                                | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_sub_bcast_col_band(
    uint32_t dst_data_idx, uint32_t dst_col_vec_idx, uint32_t data_band, uint32_t bcast_band) {
    MATH(
        (_sfpu_bcast_col_band_check_(data_band, bcast_band),
         SFPU_BINARY_CALL(
             DST_SYNC_MODE,
             is_fp32_dest_acc_en,
             _calculate_sfpu_binary_bcast_col_band_,
             (ckernel::BinaryOp::SUB),
             dst_data_idx,
             dst_col_vec_idx,
             dst_data_idx,
             VectorMode::None,
             data_band,
             bcast_band)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_add_bcast_col_band(
    uint32_t dst_data_idx, uint32_t dst_col_vec_idx, uint32_t data_band, uint32_t bcast_band) {
    MATH(
        (_sfpu_bcast_col_band_check_(data_band, bcast_band),
         SFPU_BINARY_CALL(
             DST_SYNC_MODE,
             is_fp32_dest_acc_en,
             _calculate_sfpu_binary_bcast_col_band_,
             (ckernel::BinaryOp::ADD),
             dst_data_idx,
             dst_col_vec_idx,
             dst_data_idx,
             VectorMode::None,
             data_band,
             bcast_band)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_mul_bcast_col_band(
    uint32_t dst_data_idx, uint32_t dst_col_vec_idx, uint32_t data_band, uint32_t bcast_band) {
    MATH(
        (_sfpu_bcast_col_band_check_(data_band, bcast_band),
         SFPU_BINARY_CALL(
             DST_SYNC_MODE,
             is_fp32_dest_acc_en,
             _calculate_sfpu_binary_bcast_col_band_,
             (ckernel::BinaryOp::MUL),
             dst_data_idx,
             dst_col_vec_idx,
             dst_data_idx,
             VectorMode::None,
             data_band,
             bcast_band)));
}

// clang-format off
/**
 * Apply `(data - mean) * inv_std` in one SFPU traversal, broadcasting column 0
 * of each statistics tile across all columns. All operands and the in-place
 * output are FP32. DST must be acquired, and `sfpu_bcast_col_init()` must have
 * been called since any scalar-broadcast normalisation.
 *
 * Not supported on Quasar: a kernel that calls it fails to build.
 *
 * Return value: None
 *
 * | Argument            | Description                                              | Type     | Valid Range                                           | Required |
 * |---------------------|----------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_idx        | Data tile and in-place output                            | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_mean_col_idx    | Mean tile; column 0 is broadcast                         | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_inv_std_col_idx | Inverse-standard-deviation tile; column 0 is broadcast   | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sfpu_normalize_bcast_col(uint32_t dst_data_idx, uint32_t dst_mean_col_idx, uint32_t dst_inv_std_col_idx) {
    MATH(
        (::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(
             dst_mean_col_idx, dst_inv_std_col_idx, dst_mean_col_idx, VectorMode::None),
         SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
             DST_SYNC_MODE,
             DST_ACCUM_MODE,
             _calculate_sfpu_normalize_bcast_col_full_tile_,
             dst_data_idx,
             dst_mean_col_idx,
             dst_data_idx,
             VectorMode::None,
             dst_inv_std_col_idx)));
}

// clang-format off
/**
 * Apply `(data - mean) * inv_std` to two in-place data tiles while reusing
 * each column-vector statistics broadcast. All operands and outputs are FP32.
 * DST must be acquired, and `sfpu_bcast_col_init()` must have been called.
 *
 * Not supported on Quasar: a kernel that calls it fails to build.
 *
 * Return value: None
 *
 * | Argument             | Description                                            | Type     | Valid Range                                           | Required |
 * |----------------------|--------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_idx         | First data tile and in-place output                    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_second_data_idx  | Second data tile and in-place output                   | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_mean_col_idx     | Mean tile; column 0 is broadcast                       | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_inv_std_col_idx  | Inverse-standard-deviation tile; column 0 is broadcast | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sfpu_normalize_bcast_col_two_tiles(
    uint32_t dst_data_idx, uint32_t dst_second_data_idx, uint32_t dst_mean_col_idx, uint32_t dst_inv_std_col_idx) {
    MATH(
        (::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(
             dst_data_idx, dst_second_data_idx, dst_data_idx, VectorMode::None),
         ::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(
             dst_mean_col_idx, dst_inv_std_col_idx, dst_mean_col_idx, VectorMode::None),
         SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
             DST_SYNC_MODE,
             DST_ACCUM_MODE,
             _calculate_sfpu_normalize_bcast_col_two_tiles_,
             dst_data_idx,
             dst_second_data_idx,
             dst_data_idx,
             VectorMode::None,
             dst_mean_col_idx,
             dst_inv_std_col_idx)));
}

// clang-format off
/**
 * Apply `(data + residual - mean) * inv_std` in one SFPU traversal,
 * broadcasting column 0 of each statistics tile. All operands and the in-place
 * output are FP32. DST must be acquired, and `sfpu_bcast_col_init()` must have
 * been called since any scalar-broadcast normalisation.
 *
 * Not supported on Quasar: a kernel that calls it fails to build.
 *
 * Return value: None
 *
 * | Argument            | Description                                              | Type     | Valid Range                                           | Required |
 * |---------------------|----------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_idx        | Data tile and in-place output                            | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_residual_idx    | Residual tile added to the data                          | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_mean_col_idx    | Mean tile; column 0 is broadcast                         | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_inv_std_col_idx | Inverse-standard-deviation tile; column 0 is broadcast   | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sfpu_residual_normalize_bcast_col(
    uint32_t dst_data_idx, uint32_t dst_residual_idx, uint32_t dst_mean_col_idx, uint32_t dst_inv_std_col_idx) {
    MATH(
        (::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(
             dst_mean_col_idx, dst_inv_std_col_idx, dst_mean_col_idx, VectorMode::None),
         SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
             DST_SYNC_MODE,
             DST_ACCUM_MODE,
             _calculate_sfpu_residual_normalize_bcast_col_full_tile_,
             dst_data_idx,
             dst_residual_idx,
             dst_data_idx,
             VectorMode::None,
             dst_mean_col_idx,
             dst_inv_std_col_idx)));
}

// clang-format off
/**
 * Initialise SFPU state for scalar-broadcast normalisation. Call once before
 * `sfpu_normalize_bcast_scalar()` and again if an intervening operation changes
 * the shared SFPU configuration or zero-stride address modifier.
 *
 * On Quasar `sfpu_normalize_bcast_scalar()` is not supported, so this init
 * has nothing to set up.
 *
 * Return value: None
 */
// clang-format on
ALWI void sfpu_normalize_bcast_scalar_init() {
    // Scalar broadcast needs the common SFPU config and zero-stride address modifier,
    // but not the persistent column-broadcast mask or replay program.
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::_sfpu_binary_bcast_init_, (ckernel::BroadcastType::ROW))));
}

// clang-format off
/**
 * Apply `(data - mean) * inv_std` in one SFPU traversal, broadcasting element
 * zero of each statistics tile across the complete data tile. All operands and
 * the in-place output are FP32. DST must be acquired, and
 * `sfpu_normalize_bcast_scalar_init()` must have been called.
 *
 * This operation invalidates the persistent column-broadcast mask and clobbers
 * programmable constant register 2. A later column-broadcast or SFPU operation
 * that consumes that constant must run its corresponding init first.
 *
 * Not supported on Quasar: a kernel that calls it fails to build.
 *
 * Return value: None
 *
 * | Argument               | Description                                      | Type     | Valid Range                                           | Required |
 * |------------------------|--------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_idx           | Data tile and in-place output                    | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_mean_scalar_idx    | Mean tile; element zero is broadcast             | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_inv_std_scalar_idx | Inverse-standard-deviation tile; element zero is broadcast | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sfpu_normalize_bcast_scalar(
    uint32_t dst_data_idx, uint32_t dst_mean_scalar_idx, uint32_t dst_inv_std_scalar_idx) {
    MATH(
        (::ckernel::_sfpu_binary_check_<DST_SYNC_MODE>(
             dst_mean_scalar_idx, dst_inv_std_scalar_idx, dst_mean_scalar_idx, VectorMode::None),
         SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
             DST_SYNC_MODE,
             DST_ACCUM_MODE,
             _calculate_sfpu_normalize_bcast_scalar_full_tile_,
             dst_data_idx,
             dst_mean_scalar_idx,
             dst_data_idx,
             VectorMode::None,
             dst_inv_std_scalar_idx)));
}

// ============================================================================
// BCAST_SCALAR: element [0, 0] of a DEST tile parked in an SFPU register, then
// DEST tiles multiplied by it in place
// ============================================================================

// clang-format off
/**
 * Initialise SFPU state for the parked-scalar multiply: the shared SFPU
 * configuration and the zero-stride address modifier. Call once before
 * `sfpu_bcast_scalar_load()`, and again after any init that reprograms them.
 * Neither the column-broadcast mask nor its replay program is touched, so the
 * two families coexist.
 *
 * Not supported on Quasar: a kernel that calls it fails to build.
 *
 * Return value: None
 */
// clang-format on
ALWI void sfpu_bcast_scalar_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::_sfpu_bcast_scalar_init_))); }

// clang-format off
/**
 * Broadcast element [0, 0] of `dst_scalar_idx` to every SFPU lane and park it in
 * programmable constant register 2 (LREG14, sfpi's vConstFloatPrgm2), bit for
 * bit (-0.0, NaN payloads and denormals included). Only element [0, 0] is read;
 * the tile may be overwritten right after the call. The parked value is
 * consumed by `sfpu_mul_bcast_scalar()` and stays valid across any number of
 * tiles and across the `sfpu_*_bcast_*` operations, until an init that
 * programs the constant registers LREG12..14 runs (`exp_tile_init`,
 * `sfpu_normalize_bcast_scalar`, the sqrt / rsqrt inits): after such an init
 * call `sfpu_bcast_scalar_init()` and load again. The load itself runs sfpi
 * code and may use LREG0..7, so it invalidates the persistent column-broadcast
 * lane mask: a column-broadcast op that follows a load runs its init first,
 * which leaves the parked value alone. The scalar tile and DEST must be FP32
 * and DEST acquired.
 *
 * Not supported on Quasar: a kernel that calls it fails to build.
 *
 * Return value: None
 *
 * | Argument        | Description                                           | Type     | Valid Range                                           | Required |
 * |-----------------|-------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_scalar_idx  | The tile in DST whose element [0, 0] is the scalar    | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sfpu_bcast_scalar_load(uint32_t dst_scalar_idx) {
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _sfpu_bcast_scalar_load_,
        dst_scalar_idx,
        dst_scalar_idx,
        dst_scalar_idx,
        VectorMode::None)));
}

// clang-format off
/**
 * Multiply `dst_data_idx` in place by the scalar parked by
 * `sfpu_bcast_scalar_load()`:
 *
 *     DST[dst_data_idx][r][c] *= scalar
 *
 * over the faces `vector_mode` names (`VectorMode::RC` the whole tile,
 * `VectorMode::R` rows 0..15, `VectorMode::C` columns 0..15) and the first
 * `ITERATIONS` SFPU vectors of each face (iteration k is face rows
 * 4 * (k / 2) .. +3, one column parity): 8 is the face, 2 is its rows 0..3, so
 * `<2>(idst, VectorMode::R)` scales rows 0..3 of all 32 columns. Everything
 * else is untouched. The multiply is the column broadcast's own instruction,
 * so the result is bit-identical to `sfpu_mul_bcast_col` with a column tile
 * that holds the scalar in every row. FP32 DEST only; DEST must be acquired.
 *
 * Not supported on Quasar: a kernel that calls it fails to build.
 *
 * Return value: None
 *
 * | Argument       | Description                                                         | Type       | Valid Range                                           | Required |
 * |----------------|---------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | ITERATIONS     | SFPU vectors per face (template): 8 is the face, 2 its rows 0..3    | int        | 1 to 8                                                | False    |
 * | dst_data_idx   | The index of the data tile in DST (result is written back in-place) | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | vector_mode    | The faces: VectorMode::RC (all), R (rows 0..15), C (columns 0..15)  | VectorMode | RC, R, C                                              | False    |
 */
// clang-format on
template <int ITERATIONS = 8, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_mul_bcast_scalar(uint32_t dst_data_idx, VectorMode vector_mode = VectorMode::RC) {
    static_assert(is_fp32_dest_acc_en, "sfpu_mul_bcast_scalar: the parked scalar and the tiles are fp32 DEST only");
    static_assert(ITERATIONS >= 1 && ITERATIONS <= 8, "sfpu_mul_bcast_scalar: ITERATIONS is 1..8 vectors per face");
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_mul_bcast_scalar_,
        (ITERATIONS),
        dst_data_idx,
        dst_data_idx,
        dst_data_idx,
        VectorMode::None,
        vector_mode)));
}

// clang-format off
/**
 * `sfpu_bcast_scalar_load(dst_scalar_idx)` followed by
 * `sfpu_mul_bcast_scalar<ITERATIONS>(dst_data_idx, vector_mode)`: one tile
 * scaled by element [0, 0] of another, the scalar left parked for further
 * tiles. See both for the preconditions.
 *
 * Return value: None
 *
 * | Argument        | Description                                                         | Type       | Valid Range                                           | Required |
 * |-----------------|---------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | ITERATIONS      | SFPU vectors per face (template)                                    | int        | 1 to 8                                                | False    |
 * | dst_data_idx    | The index of the data tile in DST (result is written back in-place) | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | dst_scalar_idx  | The tile in DST whose element [0, 0] is the scalar                  | uint32_t   | Must be less than the size of the DST register buffer | True     |
 * | vector_mode     | The faces, as sfpu_mul_bcast_scalar                                 | VectorMode | RC, R, C                                              | False    |
 */
// clang-format on
template <int ITERATIONS = 8, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_mul_bcast_scalar(
    uint32_t dst_data_idx, uint32_t dst_scalar_idx, VectorMode vector_mode = VectorMode::RC) {
    sfpu_bcast_scalar_load(dst_scalar_idx);
    sfpu_mul_bcast_scalar<ITERATIONS, is_fp32_dest_acc_en>(dst_data_idx, vector_mode);
}

// ============================================================================
// BCAST_ROW: broadcast row 0 of the bcast tile across all 32 tile rows
// ============================================================================

// clang-format off
/**
 * Initialize SFPU state for {add,sub,mul}-with-row-broadcast. Must be called
 * once (before any acquire_dst / sfpu_*_bcast_row pair). As with the column
 * variant a single init is shared across ADD / SUB / MUL.
 *
 * On Quasar the init only checks the broadcast dim: each sfpu_*_bcast_*
 * call records its binop and lane mask before it runs, so nothing persists
 * between calls.
 *
 * Return value: None
 */
// clang-format on
ALWI void sfpu_bcast_row_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::_sfpu_binary_bcast_init_, (ckernel::BroadcastType::ROW))));
}

ALWI void sfpu_sub_bcast_row_init() { sfpu_bcast_row_init(); }

ALWI void sfpu_add_bcast_row_init() { sfpu_bcast_row_init(); }

ALWI void sfpu_mul_bcast_row_init() { sfpu_bcast_row_init(); }

// clang-format off
/**
 * Apply a binop between `dst_data_idx` and the row-broadcast of row 0 of
 * `dst_row_vec_idx`, entirely within DST registers. Result is written
 * in-place into `dst_data_idx`:
 *
 *     sub:  DST[dst_data_idx][r][c] -= DST[dst_row_vec_idx][0][c]
 *     add:  DST[dst_data_idx][r][c] += DST[dst_row_vec_idx][0][c]
 *     mul:  DST[dst_data_idx][r][c] *= DST[dst_row_vec_idx][0][c]
 *
 * for r in [0..31].
 *
 * Both source registers must contain FP32 data (compile with
 * fp32_dest_acc_en = true). The row-vector register is expected to carry
 * meaningful values only in row 0 (e.g. layernorm gamma/beta); values in
 * rows 1..31 are ignored. Output is FP32.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * Requires a prior call to the matching `sfpu_*_bcast_row_init` (or the
 * shared `sfpu_bcast_row_init`).
 *
 * Return value: None
 *
 * | Argument         | Description                                                         | Type     | Valid Range                                           | Required |
 * |------------------|---------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | dst_data_idx     | The index of the data tile in DST (result is written back in-place) | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | dst_row_vec_idx  | The index of the row-vector tile in DST (row 0 is broadcast)        | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_sub_bcast_row(uint32_t dst_data_idx, uint32_t dst_row_vec_idx) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_sfpu_binary_bcast_full_tile_,
        (ckernel::BinaryOp::SUB, ckernel::BroadcastType::ROW),
        dst_data_idx,
        dst_row_vec_idx,
        dst_data_idx,
        VectorMode::None)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_add_bcast_row(uint32_t dst_data_idx, uint32_t dst_row_vec_idx) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_sfpu_binary_bcast_full_tile_,
        (ckernel::BinaryOp::ADD, ckernel::BroadcastType::ROW),
        dst_data_idx,
        dst_row_vec_idx,
        dst_data_idx,
        VectorMode::None)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_mul_bcast_row(uint32_t dst_data_idx, uint32_t dst_row_vec_idx) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_sfpu_binary_bcast_full_tile_,
        (ckernel::BinaryOp::MUL, ckernel::BroadcastType::ROW),
        dst_data_idx,
        dst_row_vec_idx,
        dst_data_idx,
        VectorMode::None)));
}

// ============================================================================
// Unified templated API: select broadcast dim and binop at compile time using
// the shared ckernel enums (BroadcastType / EltwiseBinaryType).
// ============================================================================

// clang-format off
/**
 * Compile-time-dispatched init for SFPU binary broadcast. Picks the column or
 * row variant based on the `Dim` template parameter; the binop opcode is
 * selected per call in `sfpu_bcast`, so a single init covers ADD / SUB / MUL.
 * On Quasar it only checks `Dim` (see `sfpu_bcast_col_init`).
 *
 * Only `BroadcastType::COL` and `BroadcastType::ROW` are supported.
 */
// clang-format on
template <BroadcastType Dim>
ALWI void sfpu_bcast_init() {
    static_assert(
        Dim == BroadcastType::COL || Dim == BroadcastType::ROW,
        "sfpu_bcast_init: only BroadcastType::COL / BroadcastType::ROW are supported");
    if constexpr (Dim == BroadcastType::COL) {
        sfpu_bcast_col_init();
    } else {
        sfpu_bcast_row_init();
    }
}

// clang-format off
/**
 * Compile-time-dispatched SFPU binary broadcast. Equivalent to the matching
 * `sfpu_{add,sub,mul}_bcast_{col,row}` selected by the `Dim` and `Op` template
 * parameters. See those functions for semantics and preconditions.
 *
 * Only `BroadcastType::COL` / `ROW` and `EltwiseBinaryType::ELW{ADD,SUB,MUL}`
 * are supported.
 */
// clang-format on
template <BroadcastType Dim, EltwiseBinaryType Op, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_bcast(uint32_t dst_data_idx, uint32_t dst_bcast_idx, VectorMode vector_mode = VectorMode::RC) {
    static_assert(
        Dim == BroadcastType::COL || Dim == BroadcastType::ROW,
        "sfpu_bcast: only BroadcastType::COL / BroadcastType::ROW are supported");
    static_assert(
        Op == EltwiseBinaryType::ELWADD || Op == EltwiseBinaryType::ELWSUB || Op == EltwiseBinaryType::ELWMUL,
        "sfpu_bcast: only EltwiseBinaryType::ELW{ADD,SUB,MUL} are supported");
    if constexpr (Dim == BroadcastType::COL) {
        if constexpr (Op == EltwiseBinaryType::ELWADD) {
            sfpu_add_bcast_col<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx, vector_mode);
        } else if constexpr (Op == EltwiseBinaryType::ELWSUB) {
            sfpu_sub_bcast_col<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx, vector_mode);
        } else {
            sfpu_mul_bcast_col<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx, vector_mode);
        }
    } else {
        // The row broadcast has no partial form.
        MATH(LLK_ASSERT(vector_mode == VectorMode::RC, "sfpu_bcast<ROW>: VectorMode::RC only"));
        if constexpr (Op == EltwiseBinaryType::ELWADD) {
            sfpu_add_bcast_row<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx);
        } else if constexpr (Op == EltwiseBinaryType::ELWSUB) {
            sfpu_sub_bcast_row<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx);
        } else {
            sfpu_mul_bcast_row<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx);
        }
    }
}

// clang-format off
/**
 * Compile-time-dispatched 4-row band of the column broadcast. Equivalent to the
 * matching `sfpu_{add,sub,mul}_bcast_col_band` selected by the `Op` template
 * parameter. See those functions for semantics and preconditions.
 */
// clang-format on
template <EltwiseBinaryType Op, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_bcast_col_band(uint32_t dst_data_idx, uint32_t dst_bcast_idx, uint32_t data_band, uint32_t bcast_band) {
    static_assert(
        Op == EltwiseBinaryType::ELWADD || Op == EltwiseBinaryType::ELWSUB || Op == EltwiseBinaryType::ELWMUL,
        "sfpu_bcast_col_band: only EltwiseBinaryType::ELW{ADD,SUB,MUL} are supported");
    if constexpr (Op == EltwiseBinaryType::ELWADD) {
        sfpu_add_bcast_col_band<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx, data_band, bcast_band);
    } else if constexpr (Op == EltwiseBinaryType::ELWSUB) {
        sfpu_sub_bcast_col_band<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx, data_band, bcast_band);
    } else {
        sfpu_mul_bcast_col_band<is_fp32_dest_acc_en>(dst_data_idx, dst_bcast_idx, data_band, bcast_band);
    }
}

}  // namespace ckernel
