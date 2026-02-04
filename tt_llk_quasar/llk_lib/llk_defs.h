// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel
{

enum class ReduceDim : std::uint8_t
{
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR,
};

enum class PoolType : std::uint8_t
{
    SUM,
    AVG,
    MAX,
};

enum class DataCopyType : std::uint8_t
{
    A2D,
    B2D,
};

enum class EltwiseBinaryType : std::uint8_t
{
    ELWMUL,
    ELWADD,
    ELWSUB,
};

// Broadcasts only occur on SrcB
enum class BroadcastType : std::uint8_t
{
    NONE,
    COL,
    ROW,
    SCALAR,
};

enum class SfpuType : std::uint32_t
{
    tanh,
    gelu,
    exponential,
    reciprocal,
    sqrt,
    rsqrt,
    relu,
    lrelu,
    relumin,
    relumax,
    stochround,
    typecast,
    add,
    square
};

enum class DstSync : std::uint8_t
{
    SyncHalf,
    SyncFull,
};

enum class MathFidelity : std::uint8_t
{
    LoFi  = 0,
    HiFi2 = 2,
    HiFi3 = 3,
    HiFi4 = 4
};

enum class StochRndType : std::uint8_t
{
    None = 0,
    Fpu  = 1,
    Pack = 2,
    All  = 3,
};

} // namespace ckernel

// Make SfpuType available in global namespace for compatibility with test infrastructure
using SfpuType = ckernel::SfpuType;
