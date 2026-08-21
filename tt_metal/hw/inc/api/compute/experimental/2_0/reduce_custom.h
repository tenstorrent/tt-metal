// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef TRISC_MATH
#include "llk_math_reduce_api.h"                      // llk_math_reduce_uninit
#include "experimental/llk_math_reduce_custom_api.h"  // llk_math_reduce_block_max_row(_init)
#endif

#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_AB_reduce_custom_api.h"  // llk_unpack_AB_reduce_block_max_row_init/uninit + _core_
#endif

#ifdef TRISC_PACK
#include "llk_pack_reduce_api.h"  // llk_pack_reduce_mask_config_impl / _clear
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// Id-free (2.0) reduce_custom == the SDPA block MAX-row reduce (legacy `reduce_block_max_row`). Reduces a block
// of `block_ct_dim` consecutive data tiles in the width dimension, taking the per-row MAX across the whole block
// (scaled by the scaler tile, whose F0 holds 1.0), into a SINGLE DST tile -- REDUCE_ROW + PoolType::MAX only.
// Takes one LLKOperand per L1 input (data, scaler) and one for the output. Like the generic 2.0 reduce it is
// FORMAT-FREE at the op level: formats are programmed by compute_kernel_hw_startup(data, scaler, out); every core
// here consumes only geometry (the DATA tile shape / the OUT face_r_dim for the packer edge mask) + the two
// runtime input L1 addresses. It REUSES the existing custom reduce LLK cores (no new LLK header): the init / math
// / uninit wrappers already take no CB id, and the unpack-op resolves the two addresses id-free (base + tile
// stride) and calls the `_llk_unpack_AB_reduce_block_max_row_` core directly -- exactly what the CB-id wrapper
// did minus the CB lookup. block_ct_dim is a compile-time NTTP (the runtime-block_ct_dim SDPA variants are not
// ported); respect_trigger (the SDPA MOP-split semaphore optimization) defaults off.

// clang-format off
/**
 * Reduce-custom (SDPA block MAX-row) init: programs UNPACK (block max-row MOP), MATH (block max-row), and the
 * PACK edge-mask for a REDUCE_ROW reduction. compute_kernel_hw_startup(data, scaler, out) must already have
 * programmed the formats (operand + scaler assumed bfloat16_b, matched exp width).
 *
 * | Param Type | Name            | Description                                                    | Type       | Valid Range | Required |
 * |------------|-----------------|--------------------------------------------------------------|------------|-------------|----------|
 * | Template   | block_ct_dim    | Number of width-dim tiles reduced together as one block      | uint32_t   | >= 1        | True     |
 * | Template   | respect_trigger | SDPA MOP-split semaphore optimization (see legacy notes)      | bool       | {true,false}| False    |
 * | Function   | data            | Input operand A (reduced); drives the tile geometry          | LLKOperand |             | True     |
 * | Function   | out             | Output operand (drives the packer edge mask via face_r_dim)  | LLKOperand |             | True     |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim,
    bool respect_trigger = false,
    DataFormat DF,
    TensorShape DS,
    DataFormat OF,
    TensorShape OS>
ALWI void reduce_block_max_row_init(LLKOperand<DF, DS> /*data*/, LLKOperand<OF, OS> /*out*/) {
    static_assert(is_legal_tile_shape(DS), "reduce_block_max_row_init: illegal data tile shape.");
    static_assert(is_legal_tile_shape(OS), "reduce_block_max_row_init: illegal output tile shape.");
    UNPACK((llk_unpack_AB_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE, respect_trigger>(DS)));
    MATH((llk_math_reduce_block_max_row_init<block_ct_dim, DST_ACCUM_MODE>(DS)));
    PACK((llk_pack_reduce_mask_config_impl<ReduceDim::REDUCE_ROW, PackMode::Default>(OS.face_r_dim)));
}

// clang-format off
/**
 * Reduce-custom (SDPA block MAX-row) execute: reduces the block of `block_ct_dim` data tiles starting at
 * row_start_index (each combined with the scaler tile) into DST[idst], taking the per-row MAX across the block.
 * Pair with reduce_block_max_row_init. DST must be acquired. Input addresses are resolved id-free: the data
 * block base advances by row_start_index * tile_stride_words (one-tile page; exp section included for block
 * floats), the scaler base is used as-is.
 *
 * | Param Type | Name            | Description                                                    | Type       | Valid Range | Required |
 * |------------|-----------------|--------------------------------------------------------------|------------|-------------|----------|
 * | Template   | block_ct_dim    | Number of width-dim tiles reduced together as one block      | uint32_t   | >= 1        | True     |
 * | Template   | respect_trigger | SDPA MOP-split semaphore optimization (see legacy notes)      | bool       | {true,false}| False    |
 * | Function   | data / scaler   | Input operands (data reduced; scaler F0 holds 1.0)           | LLKOperand |             | True     |
 * | Function   | row_start_index | Index of the first data tile of the block within `data`      | uint32_t   |             | True     |
 * | Function   | idst            | DST register index for the reduced result                    | uint32_t   | 0 to 15     | True     |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim,
    bool respect_trigger = false,
    DataFormat DF,
    TensorShape DS,
    DataFormat SF,
    TensorShape SS>
ALWI void reduce_block_max_row(
    LLKOperand<DF, DS> data, LLKOperand<SF, SS> scaler, std::uint32_t row_start_index, std::uint32_t idst) {
    static_assert(is_legal_tile_shape(DS), "reduce_block_max_row: illegal data tile shape.");
    constexpr std::uint32_t data_stride = tile_stride_words(static_cast<std::uint8_t>(DF), DS);
    // Match legacy reduce_block_max_row: UNPACK the whole block (base_a + row_start*stride, base_b), then MATH.
    UNPACK((_llk_unpack_AB_reduce_block_max_row_<respect_trigger>(
        data.l1_address + row_start_index * data_stride, scaler.l1_address)));
    MATH((llk_math_reduce_block_max_row<block_ct_dim, DST_ACCUM_MODE>(idst, DS)));
}

// clang-format off
/**
 * Reduce-custom uninit: reset the MATH reduce state, clear the packer edge mask, and restore the unpacker state
 * saved by reduce_block_max_row_init. respect_trigger must match the value passed to init / execute.
 *
 * | Param Type | Name            | Description                                               | Type | Valid Range  | Required |
 * |------------|-----------------|----------------------------------------------------------|------|--------------|----------|
 * | Template   | respect_trigger | SDPA MOP-split semaphore optimization (see legacy notes)  | bool | {true,false} | False    |
 */
// clang-format on
template <bool respect_trigger = false>
ALWI void reduce_block_max_row_uninit() {
    MATH((llk_math_reduce_uninit()));
    PACK((llk_pack_reduce_mask_clear()));
    UNPACK((llk_unpack_AB_reduce_block_max_row_uninit<respect_trigger>()));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
