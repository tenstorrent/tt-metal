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

enum class TensixProcessorTypes : uint8_t { DM0 = 0, MATH0 = 1, MATH1 = 2, MATH2 = 3, MATH3 = 4, COUNT = 5 };

enum class EthProcessorTypes : uint8_t { DM0 = 0, DM1 = 1, COUNT = 2 };

enum class DramProcessorTypes : uint8_t { DM0 = 0, COUNT = 1 };

constexpr uint8_t MaxProcessorsPerCoreType = 8;
constexpr uint8_t MaxDMProcessorsPerCoreType = 8;
constexpr uint8_t NumTensixDispatchClasses = 3;
constexpr uint8_t NumEthDispatchClasses = 2;
constexpr uint8_t NumDramDispatchClasses = 1;
constexpr uint8_t noc_size_x = 8;
constexpr uint8_t noc_size_y = 4;
constexpr uint8_t tensix_harvest_axis = 0x1;
#define LOG_BASE_2_OF_DRAM_ALIGNMENT 6
#define LOG_BASE_2_OF_L1_ALIGNMENT 4
