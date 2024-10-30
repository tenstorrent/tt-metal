// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

enum ProgrammableCoreType {
    TENSIX     = 0,
    ACTIVE_ETH = 1,
    IDLE_ETH   = 2,
    COUNT      = 3,
};

enum class AddressableCoreType : uint8_t {
    TENSIX    = 0,
    ETH       = 1,
    PCIE      = 2,
    DRAM      = 3,
    HARVESTED = 4,
    UNKNOWN   = 5,
    COUNT     = 6,
};

enum class TensixProcessorTypes : uint8_t {
    DM0    = 0,
    DM1    = 1,
    MATH0  = 2,
    MATH1  = 3,
    MATH2  = 4,
    COUNT  = 5
};

enum class EthProcessorTypes : uint8_t {
    DM0    = 0,
    DM1    = 1,
    COUNT  = 2
};

enum class DramProcessorTypes : uint8_t {
    DM0     = 0,
    COUNT   = 1
};

constexpr uint8_t MaxProcessorsPerCoreType = 5;
constexpr uint8_t NumTensixDispatchClasses = 3;
constexpr uint8_t NumEthDispatchClasses = 1;
constexpr uint8_t NumDramDispatchClasses = 1;
