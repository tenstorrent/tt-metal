// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"

// SFPU op tags: the name of an op and its variants, no code, so this header is included on every TRISC thread.
// The math behind a tag is sfpu::math_for<OP, LAYOUT>, defined next to the op (ckernel_sfpu_<op>.h).

namespace ckernel::op {

/** @brief exp(x). APPROXIMATION_MODE picks the hardware LUT; otherwise fp32-accurate on a 32-bit register. */
template <bool APPROXIMATION_MODE>
struct Exp {};

/** @brief x * x. */
struct Square {};

/**
 * @brief in0 + in1: float on a float layout, Int32 on an I32 layout.
 * @tparam ROUNDING: NearestEven rounds a float result to bf16 on a 16-bit register; Default leaves it to the store.
 */
template <DstRoundingMode ROUNDING = DstRoundingMode::Default>
struct Add {};

}  // namespace ckernel::op
