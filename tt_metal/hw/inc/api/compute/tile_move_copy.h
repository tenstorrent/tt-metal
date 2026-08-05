// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "sanitizer/api.h"
#include "llk_assert.h"

#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif
namespace ckernel {

// clang-format off
/**
 * Paired init function for copy_tile / copy_block_matmul_partials. Must be preceded - exactly once,
 * at the very top of the kernel - by compute_kernel_hw_startup(icb, ocb), which performs the one-time
 * hardware configuration. copy_init() then reconfigures the unpacker/math pipeline for the copy op and
 * is the function to call before copy_tile() (including when switching to copy from another op). It
 * does not reconfigure the unpacker data types. Eltwise-unary / SFPU kernels use this same init:
 * compute_kernel_hw_startup(icb, ocb) once, then copy_init(icb).
 *
 * Return value: None
 *
 * | Argument                    | Description                                       | Type     | Valid Range                                                         | Required |
 * |-----------------------------|---------------------------------------------------|----------|--------------------------------------------------------------------|----------|
 * | cbid                        | The identifier of the input circular buffer (CB)  | uint32_t | 0 to 31                                                            | True     |
 * | transpose                   | Flag to perform transpose on SrcA                 | uint32_t | Any positive value will indicate transpose is set                  | False    |
 * | transpose_within_16x16_face | Flag to perform transpose within 16x16 face       | uint32_t | Any positive value will indicate transpose within 16x16 face is set | False    |
 */
// clang-format on
ALWI void copy_init(
    uint32_t cbid,
    uint32_t transpose = 0,
    uint32_t transpose_within_16x16_face = false,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(cbid, call_line);
#else
    LLK_ASSERT(transpose_within_16x16_face == false, "Transpose within face not supported on Quasar");
    LLK_ASSERT(transpose == 0, "Transpose not supported on Quasar");
#endif
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, transpose_within_16x16_face, cbid)));
    // 4th template arg is arch-divergent (unpack_to_dest on Quasar, is_int_fpu_en on WH/BH); keep it
    // arch-specific so WH/BH don't wrongly enable the integer-FPU datacopy MOP.
#ifndef ARCH_QUASAR
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(cbid)));
    // Eltwise unary / SFPU ops keep the Src zero-substitution flag disabled to preserve bf16 -0.0.
    // Folded into the canonical copy init so it is a drop-in for the eltwise-unary short init; this
    // also preserves the zero flag on the plain-copy path. MATH-only config: it records no
    // format-reconfig diff, so the single-SrcA reconfig tracking is unchanged. Not present on Quasar.
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
#else
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        cbid)));
#endif
}

// clang-format off
/**
 * Copies a single tile from the specified input CB and writes the result to
 * DST at a specified index. The function will employ unpacker to first unpack into SRC
 * registers and then perform move into DST registers, at a specified index.
 * For the in_tile_index to be valid for this call, cb_wait_front(n) had to be
 * previously called to ensure that at least some number n>0 of tiles are available
 * in the input CB. The CB index 0 then references the first tile in the received section of the CB,
 * up to index n-1 (in a FIFO order). The DST register buffer must be in acquired state via
 * acquire_dst call. This call is blocking and is only available on the compute
 * engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Data type | Valid range                                         | required |
 * |----------------|---------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the source circular buffer (CB) | uint32_t  | 0 to 31                                             | True     |
 * | in_tile_index  | The index of the tile to copy from the input CB   | uint32_t  | Must be less than the size of the CB                | True     |
 * | dst_tile_index | The index of the tile in the DST register         | uint32_t  | Must be less than the size of the DST register (16) | True     |
 * */
// clang-format on
ALWI void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
#ifndef ARCH_QUASAR
    LLK_SAN_FUNCTION();
#endif
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        dst_tile_index, in_cb_id)));
}

// clang-format off
/**
 * Copies a block of `ntiles` consecutive tiles from the specified input CB into consecutive DST
 * register slots. This is the uniform block entry point for the copy/datacopy op group. It uses the
 * block unpack/datacopy llk paths and requires the same initialization as `copy_tile`
 * (`copy_tile_init` / `copy_tile_to_dst_init_short`). The DST register buffer must be in acquired
 * state via *acquire_dst* call, and `cb_wait_front(n)` must have made at least
 * `start_in_tile_index + ntiles` tiles available in the input CB. This call is blocking and is only
 * available on the compute engine.
 *
 * NOTE: In the future the blocking must be folded further into a hardware MOP / REPLAY buffer (as
 * is being done for Quasar) inside llk-lib, so the whole block issues as a single packed op, without
 * changing this signature. Tracked under the Compute API Split effort (tt-metal#35739); the per-op
 * push-down lands in tt-metal#47485.
 *
 * Return value: None
 *
 * | Argument             | Description                                                | Data type | Valid range                                         | required |
 * |----------------------|------------------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | in_cb_id             | The identifier of the source circular buffer (CB)          | uint32_t  | 0 to 31                                             | True     |
 * | start_in_tile_index  | The index of the first tile to copy from the input CB      | uint32_t  | Must be less than the size of the CB                | True     |
 * | start_dst_tile_index | The index of the first destination tile in the DST register| uint32_t  | Must be less than the size of the DST register (16) | True     |
 * | ntiles               | The number of consecutive tiles to copy                    | uint32_t  | start_dst_tile_index + ntiles <= DST register size  | True     |
 * */
// clang-format on
ALWI void copy_block(uint32_t in_cb_id, uint32_t start_in_tile_index, uint32_t start_dst_tile_index, uint32_t ntiles) {
#ifndef ARCH_QUASAR
    LLK_SAN_FUNCTION();
#endif
    UNPACK((llk_unpack_A_block<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, start_in_tile_index, ntiles)));
    MATH((llk_math_eltwise_unary_datacopy_block<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        start_dst_tile_index, ntiles, in_cb_id)));
}

// =====================================================================================================================
// Deprecated API
//
// The functions below implement the old copy / eltwise-unary programming model. The new model is:
//   compute_kernel_hw_startup(icb, ocb);  // once at the start of MAIN
//   copy_init(icb);                       // before copy_tile
// Generic data-format reconfiguration is done via reconfig_data_format_srca (from reconfig_data_format.h).
// =====================================================================================================================

// clang-format off
/**
 * Perform a init for the copy tile operation.
 *
 * Return value: None
 *
 * | Argument | Description                                      | Type     | Valid Range | Required |
 * |----------|--------------------------------------------------|----------|-------------|----------|
 * | cbid     | The identifier of the input circular buffer (CB) | uint32_t | 0 to 31     | False    |
 */
// clang-format on
[[deprecated("Renamed to copy_init(). This will be removed after 15-09-2026.")]] ALWI void copy_tile_init(
    uint32_t cbid, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    copy_init(cbid, 0, false, call_line);
}

// clang-format off
/**
 * Perform the init short for copy tile. This does not reconfigure the unpacker data types.
 *
 * Return value: None
 *
 * | Argument                    | Description                                       | Type     | Valid Range                                                         | Required |
 * |-----------------------------|---------------------------------------------------|----------|--------------------------------------------------------------------|----------|
 * | cbid                        | The identifier of the input circular buffer (CB)  | uint32_t | 0 to 31                                                            | False    |
 * | transpose                   | Flag to perform transpose on SrcA                 | uint32_t | Any positive value will indicate transpose is set                  | False    |
 * | transpose_within_16x16_face | Flag to perform transpose within 16x16 face       | uint32_t | Any positive value will indicate transpose within 16x16 face is set | False    |
 */
// clang-format on
[[deprecated("Renamed to copy_init(). This will be removed after 15-09-2026.")]] ALWI void copy_tile_to_dst_init_short(
    uint32_t cbid,
    uint32_t transpose = 0,
    uint32_t transpose_within_16x16_face = false,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    copy_init(cbid, transpose, transpose_within_16x16_face, call_line);
}

// clang-format off
/**
 * Reconfigures SrcA to the new operand's data format and re-inits the copy op.
 *
 * Return value: None
 *
 * | Argument       | Description                                                       | Type     | Valid Range                                       | Required |
 * |----------------|-------------------------------------------------------------------|----------|---------------------------------------------------|----------|
 * | old_cbid       | The identifier of the previous input circular buffer (CB) to SrcA | uint32_t | 0 to 31                                           | True     |
 * | new_cbid       | The identifier of the new input circular buffer (CB) to SrcA      | uint32_t | 0 to 31                                           | True     |
 * | transpose      | Flag to perform transpose on SrcA                                 | uint32_t | Any positive value will indicate transpose is set | False    |
 */
// clang-format on
#ifndef ARCH_QUASAR
[[deprecated(
    "Call reconfig_data_format_srca(old, new) then copy_init(new, transpose). This will be removed after "
    "15-09-2026.")]] ALWI void
copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose = 0) {
    LLK_SAN_FUNCTION();
    // This reconfig call checks if old operand has different data format to
    // new operand idx, otherwise no reconfig call occurs
    reconfig_data_format_srca(old_cbid, new_cbid);
    copy_init(new_cbid, transpose);
}
#endif

// clang-format off
/**
 * @deprecated Use `copy_block()`, which is functionally equivalent (same block unpack/datacopy paths).
 * This forwarding shim is retained only for backwards compatibility and will be removed after
 * August 15th, 2026 (see .github/deprecations.json).
 *
 * Return value: None
 *
 * | Argument             | Description                                                | Data type | Valid range                                         | required |
 * |----------------------|------------------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | in_cb_id             | The identifier of the source circular buffer (CB)          | uint32_t  | 0 to 31                                             | True     |
 * | start_in_tile_index  | The index of the first tile to copy from the input CB      | uint32_t  | Must be less than the size of the CB                | True     |
 * | start_dst_tile_index | The index of the first destination tile in the DST register| uint32_t  | Must be less than the size of the DST register (16) | True     |
 * | ntiles               | The number of consecutive tiles to copy                    | uint32_t  | start_dst_tile_index + ntiles <= DST register size  | True     |
 * */
// clang-format on
[[deprecated("Use copy_block(); copy_block_matmul_partials will be removed after August 15th, 2026.")]] ALWI void
copy_block_matmul_partials(
    uint32_t in_cb_id, uint32_t start_in_tile_index, uint32_t start_dst_tile_index, uint32_t ntiles) {
    copy_block(in_cb_id, start_in_tile_index, start_dst_tile_index, ntiles);
}

}  // namespace ckernel
