// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

enum ProgrammableCoreType {
    TENSIX     = 0,
    COUNT      = 3, // for now, easier to keep structures shared across arches' the same size
};

enum class AddressableCoreType : uint8_t {
    TENSIX    = 0,
    ETH       = 1, // TODO: Make this accessible through the HAL and remove non-GS entries
    PCIE      = 2,
    DRAM      = 3,
    HARVESTED = 4,
    UNKNOWN   = 5,
    COUNT     = 6,
};
