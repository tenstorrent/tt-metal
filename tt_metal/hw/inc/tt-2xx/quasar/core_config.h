// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

enum ProgrammableCoreType {
    TENSIX = 0,
    ACTIVE_ETH = 1,
    IDLE_ETH = 2,
    COUNT = 3,
};

enum class TensixProcessorTypes : uint8_t {
    DM0 = 0,
    DM1 = 1,
    DM2 = 2,
    DM3 = 3,
    DM4 = 4,
    DM5 = 5,
    DM6 = 6,
    DM7 = 7,
    MATH0 = 8,
    MATH1 = 9,
    MATH2 = 10,
    MATH3 = 11,
    COUNT = 12
};

enum class EthProcessorTypes : uint8_t { DM0 = 0, DM1 = 1, COUNT = 2 };

enum class DramProcessorTypes : uint8_t { DM0 = 0, COUNT = 1 };

constexpr uint8_t MaxProcessorsPerCoreType = 24;
constexpr uint8_t MaxDMProcessorsPerCoreType = 8;
constexpr uint8_t NumTensixDispatchClasses = 3;
constexpr uint8_t NumEthDispatchClasses = 2;
constexpr uint8_t NumDramDispatchClasses = 1;
constexpr uint8_t noc_size_x = 8;
constexpr uint8_t noc_size_y = 4;
constexpr uint8_t tensix_harvest_axis = 0x2;
#define LOG_BASE_2_OF_DRAM_ALIGNMENT 6  // TODO: verify
#define LOG_BASE_2_OF_L1_ALIGNMENT 4
