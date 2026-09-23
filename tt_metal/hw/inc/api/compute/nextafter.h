// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the element-wise nextafter operation y = nextafter(x0, x1): the representable value
 * adjacent to x0 in the direction of x1, or x1 itself when the two are equal -- which is what makes
 * nextafter(+0, -0) return -0. The step is one ULP at x0's own magnitude, so the float32 and
 * bfloat16 destinations need separate entry points.
 *
 * Two deliberate departures from IEEE 754, both properties of the SFPU rather than of this op:
 * a result in the subnormal range is flushed, so nextafter(0.0f, 1.0f) returns +0.0 rather than the
 * smallest denormal; and on a bfloat16 destination a NaN result is delivered as infinity, since a
 * bfloat16 tile does not carry a NaN through the compute path at all (ttnn.multiply of a bfloat16
 * NaN by 1.0 returns infinity for the same reason). A NaN operand does propagate as NaN on a
 * float32 destination.
 * Output overwrites first operand in DST.
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand   | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output          | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nextafter_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    // One float32 ULP is a step of 1 in the dest register, which sits below bfloat16 precision and
    // is discarded when a 16-bit DEST is packed -- the op would silently return its input. That
    // case needs is_fp32_dest_acc_en, but it is not the whole rule: which entry point to call is
    // decided by the *tile* format, not by the DEST width. A bfloat16 tile with an fp32 DEST is a
    // valid and common configuration -- binary_ng turns fp32 dest accumulation on whenever any
    // operand or the output is fp32 -- and it still needs the bf16 entry point.
    static_assert(
        is_fp32_dest_acc_en,
        "nextafter_binary_tile steps one float32 ULP and requires a float32 DEST; use "
        "nextafter_bf16_binary_tile whenever the tile is bfloat16, regardless of DEST width");
    MATH((sfpu::SfpuBinary<APPROX, BinaryOp::NEXTAFTER, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nextafter_binary_tile_init() {
    MATH((sfpu::SfpuBinary<APPROX, BinaryOp::NEXTAFTER, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * bfloat16 tile variant of nextafter_binary_tile. A bfloat16 tile keeps its mantissa in the
 * top 16 bits of the fp32 dest register, so one of its ULPs is a step of 0x10000 there; stepping by
 * a single fp32 ULP instead would be rounded away when the tile is packed and the value would not
 * move at all.
 *
 * @note Selected by the tile format, not the DEST width -- a bfloat16 tile with an fp32 DEST still
 * belongs here. This cannot be a static_assert for the same reason: !is_fp32_dest_acc_en would
 * reject that valid configuration. Fed genuine float32 data it steps 0x10000 rather than 1, which
 * overshoots the neighbour by 2^16 ULPs with no diagnostic, so the caller owns this precondition.
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nextafter_bf16_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((sfpu::SfpuBinary<APPROX, BinaryOp::NEXTAFTER_BF16, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void nextafter_bf16_binary_tile_init() {
    MATH((sfpu::SfpuBinary<APPROX, BinaryOp::NEXTAFTER_BF16, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
