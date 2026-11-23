// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What a reduce computes, as the Compute API's reduce_modifiers name it and as the Plan carries it to llk-lib
// (MathPlan::reduce). Arch-independent.

#pragma once

#include <cstdint>

namespace llk2
{

/// The reduction a reduce applies.
enum class ReduceKind : std::uint8_t
{
    Max, ///< the largest datum (LLK 1.0's PoolType::MAX)
    Sum  ///< the sum of the datums, each times the scaler (LLK 1.0's PoolType::SUM)
};

/// The axis a reduce folds.
enum class ReduceDim : std::uint8_t
{
    Row,   ///< over each row's columns, every tile of a tile row: one datum per row, in column 0 (LLK 1.0's REDUCE_ROW)
    Col,   ///< over each column's rows (LLK 1.0's REDUCE_COL). Not ported: the Resolver refuses it
    Scalar ///< over every datum (LLK 1.0's REDUCE_SCALAR). Not ported: the Resolver refuses it
};

} // namespace llk2
