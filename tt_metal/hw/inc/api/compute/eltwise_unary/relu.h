// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_relu.h"
#endif

// Approach B keeps the per-arch kernel template lists here; approach A moves them into ckernel_sfpu_relu.h.

namespace ckernel {

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_tile_init() {
    MATH((sfpu::Relu<APPROX, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs element-wise computation of relu(x) = (0 if x is negative else x) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_tile(uint32_t idst) {
    MATH((sfpu::Relu<APPROX, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}
#ifndef ARCH_QUASAR
// clang-format off
/**
 * Performs element-wise computation of relu max (relu(max(x, upper_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | upper_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_max_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::ReluClamp<APPROX, false /*IS_LOWER_BOUND*/, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_max_tile_pack(uint32_t idst, uint32_t param0) {
    PACK((sfpu::ReluClamp<APPROX, false /*IS_LOWER_BOUND*/, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_max_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::ReluClamp<APPROX, false /*IS_LOWER_BOUND*/, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_max_tile_uint32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::ReluClamp<APPROX, false /*IS_LOWER_BOUND*/, DataFormat::UInt32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_max_tile_uint16(uint32_t idst, uint32_t param0) {
    MATH((sfpu::ReluClamp<APPROX, false /*IS_LOWER_BOUND*/, DataFormat::UInt16, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_max_tile_init() {
    MATH((sfpu::ReluClamp<APPROX, false /*IS_LOWER_BOUND*/, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_max_tile_init_pack() {
    PACK((sfpu::ReluClamp<APPROX, false /*IS_LOWER_BOUND*/, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

// clang-format off
/**
 * Performs element-wise computation of relu min (relu(min(x, lower_limit))) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | lower_limit    | Upper limit of relu_min                                                    | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_min_tile(uint32_t idst, uint32_t param0) {
    MATH((sfpu::ReluClamp<APPROX, true /*IS_LOWER_BOUND*/, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_min_tile_int32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::ReluClamp<APPROX, true /*IS_LOWER_BOUND*/, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_min_tile_uint32(uint32_t idst, uint32_t param0) {
    MATH((sfpu::ReluClamp<APPROX, true /*IS_LOWER_BOUND*/, DataFormat::UInt32, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_min_tile_uint16(uint32_t idst, uint32_t param0) {
    MATH((sfpu::ReluClamp<APPROX, true /*IS_LOWER_BOUND*/, DataFormat::UInt16, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              calculate(idst, VectorMode::RC, param0 /*threshold*/)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_min_tile_init() {
    MATH((sfpu::ReluClamp<APPROX, true /*IS_LOWER_BOUND*/, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::
              init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void relu_tile_int32(uint32_t idst) {
    MATH((sfpu::Relu<APPROX, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC)));
}

// clang-format off
/**
 * Performs element-wise computation of leaky relu (relu(x) + slope*-relu(-x)) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope          | slope used in leaky relu - will reinterpret unsigned int to float          | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void leaky_relu_tile(uint32_t idst, uint32_t slope = 0) {
    MATH((sfpu::LeakyRelu<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC, slope)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void leaky_relu_tile_init() {
    MATH((sfpu::LeakyRelu<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}
#endif
}  // namespace ckernel
