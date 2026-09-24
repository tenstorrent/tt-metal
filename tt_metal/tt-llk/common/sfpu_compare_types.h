// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel::sfpu
{

/**
 * @brief Relational comparison selected by the SFPU compare kernels.
 *
 * The kernel decides what the left and right operands are: the zero compares test `x OP 0`, the
 * unary compares test `x OP scalar`, and the binary compares test `in0 OP in1`.
 */
enum class CompareOp : std::uint8_t
{
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
};

/**
 * @brief Floating-point class test selected by the SFPU isinf/isnan kernels.
 */
enum class FiniteCheck : std::uint8_t
{
    isinf,
    isposinf,
    isneginf,
    isnan,
    isfinite,
};

} // namespace ckernel::sfpu
