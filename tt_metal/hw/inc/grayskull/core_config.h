// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

enum ProgrammableCoreType {
    TENSIX = 0,
    COUNT = 3,  // for now, easier to keep structures shared across arches' the same size
};

enum class TensixProcessorTypes : uint8_t { DM0 = 0, DM1 = 1, MATH0 = 2, MATH1 = 3, MATH2 = 4, COUNT = 5 };

constexpr uint8_t MaxProcessorsPerCoreType = 5;
constexpr uint8_t NumTensixDispatchClasses = 3;
constexpr uint8_t noc_size_x = 13;
constexpr uint8_t noc_size_y = 12;
#define LOG_BASE_2_OF_DRAM_ALIGNMENT 5
#define LOG_BASE_2_OF_L1_ALIGNMENT 4
