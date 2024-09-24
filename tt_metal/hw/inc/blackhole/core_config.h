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
