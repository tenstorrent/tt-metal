// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/flags.hpp>

namespace tensor_accessor {

/**
 * @brief Encodes a fundamental configuration of a tensor accessor, which must be available at compile time.
 * Specifies whether the tensor is sharded, stored in DRAM, and which arguments should be passed as compile-time vs
 * runtime arguments.
 */
enum class ArgConfig : uint8_t {
    None = 0,
    Sharded = 1 << 0,
    IsDram = 1 << 1,
    RuntimeRank = 1 << 2,
    RuntimeNumBanks = 1 << 3,
    RuntimeTensorShape = 1 << 4,
    RuntimeShardShape = 1 << 5,
    RuntimeBankCoords = 1 << 6,
    Runtime = RuntimeRank | RuntimeNumBanks | RuntimeTensorShape | RuntimeShardShape | RuntimeBankCoords
};

using ArgsConfig = Flags<ArgConfig>;
inline constexpr ArgsConfig operator|(ArgConfig a, ArgConfig b) noexcept { return ArgsConfig(a) | b; }
inline constexpr ArgsConfig operator|(ArgConfig a, ArgsConfig b) noexcept { return ArgsConfig(a) | b; }

}  // namespace tensor_accessor
