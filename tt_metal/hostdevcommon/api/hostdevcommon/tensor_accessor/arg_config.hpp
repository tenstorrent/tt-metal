// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/flags.hpp>

namespace tensor_accessor {

/**
 * @brief Encodes which arguments are compile-time and which are common runtime.
 */
enum class ArgConfig : uint8_t {
    None = 0,
    Sharded = 1 << 0,
    IsDram = 1 << 1,
    RankCRTA = 1 << 2,
    NumBanksCRTA = 1 << 3,
    TensorShapeCRTA = 1 << 4,
    ShardShapeCRTA = 1 << 5,
    BankCoordsCRTA = 1 << 6,
    CRTA = RankCRTA | NumBanksCRTA | TensorShapeCRTA | ShardShapeCRTA | BankCoordsCRTA
};

using ArgsConfig = Flags<ArgConfig>;
inline constexpr ArgsConfig operator|(ArgConfig a, ArgConfig b) noexcept { return ArgsConfig(a) | b; }
inline constexpr ArgsConfig operator|(ArgConfig a, ArgsConfig b) noexcept { return ArgsConfig(a) | b; }

}  // namespace tensor_accessor
