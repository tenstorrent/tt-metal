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
    BRISC  = 0,
    NCRISC = 1,
    TRISC0 = 2,
    TRISC1 = 3,
    TRISC2 = 4,

    DM0    = 0,
    DM1    = 1,
    MATH0  = 2,
    MATH1  = 3,
    MATH2  = 4,
};

enum class EthProcessorTypes : uint8_t {
    DM0    = 0,
};

constexpr uint8_t MaxProcessorsPerCoreType = 5;
