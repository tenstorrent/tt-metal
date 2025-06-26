// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <hostdevcommon/flags.hpp>

/**
 * @brief Encodes which arguments are compile-time and which are common runtime.
 */
enum class ArgConfig : uint8_t {
    CTA = 0,
    RankCRTA = 1 << 0,
    NumBanksCRTA = 1 << 1,
    TensorShapeCRTA = 1 << 2,
    ShardShapeCRTA = 1 << 3,
    BankCoordsCRTA = 1 << 4,
    CRTA = RankCRTA | NumBanksCRTA | TensorShapeCRTA | ShardShapeCRTA | BankCoordsCRTA
};

using ArgsConfig = Flags<ArgConfig>;
inline constexpr ArgsConfig operator|(ArgConfig a, ArgConfig b) noexcept { return ArgsConfig(a) | b; }
inline constexpr ArgsConfig operator|(ArgConfig a, ArgsConfig b) noexcept { return ArgsConfig(a) | b; }
