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
    E0_MATH0 = 8,
    E0_MATH1 = 9,
    E0_MATH2 = 10,
    E0_MATH3 = 11,
    E1_MATH0 = 12,
    E1_MATH1 = 13,
    E1_MATH2 = 14,
    E1_MATH3 = 15,
    COUNT = 16
};

enum class EthProcessorTypes : uint8_t { DM0 = 0, DM1 = 1, COUNT = 2 };

enum class DramProcessorTypes : uint8_t { DM0 = 0, COUNT = 1 };

union subordinate_map_t {
    // Quasar: expanded structure for multiple DM cores
    union {
        struct {
            volatile uint64_t allDMs;
            volatile uint32_t allNeo0;
            volatile uint32_t allNeo1;
            volatile uint32_t allNeo2;
            volatile uint32_t allNeo3;
        };
        struct {
            volatile uint8_t dm1;  // Keep dm1 name for compatibility
            volatile uint8_t dm2;
            volatile uint8_t dm3;
            volatile uint8_t dm4;
            volatile uint8_t dm5;
            volatile uint8_t dm6;
            volatile uint8_t dm7;
            volatile uint8_t padding;
            volatile uint8_t neo0_trisc0;
            volatile uint8_t neo0_trisc1;
            volatile uint8_t neo0_trisc2;
            volatile uint8_t neo0_trisc3;
            volatile uint8_t neo1_trisc0;
            volatile uint8_t neo1_trisc1;
            volatile uint8_t neo1_trisc2;
            volatile uint8_t neo1_trisc3;
            volatile uint8_t neo2_trisc0;
            volatile uint8_t neo2_trisc1;
            volatile uint8_t neo2_trisc2;
            volatile uint8_t neo2_trisc3;
            volatile uint8_t neo3_trisc0;
            volatile uint8_t neo3_trisc1;
            volatile uint8_t neo3_trisc2;
            volatile uint8_t neo3_trisc3;
            uint8_t pad[12];
        };
    } __attribute__((packed));
} __attribute__((packed));

constexpr uint8_t MaxProcessorsPerCoreType = 24;
constexpr uint8_t MaxDMProcessorsPerCoreType = 8;
constexpr uint8_t NumTensixDispatchClasses = 3;
constexpr uint8_t NumEthDispatchClasses = 2;
constexpr uint8_t NumDramDispatchClasses = 1;
constexpr uint8_t noc_size_x = 8;
constexpr uint8_t noc_size_y = 4;
constexpr uint8_t tensix_harvest_axis = 0x2;
constexpr uint8_t subordinate_map_size = sizeof(subordinate_map_t);
#define LOG_BASE_2_OF_DRAM_ALIGNMENT 6  // TODO: verify
#define LOG_BASE_2_OF_L1_ALIGNMENT 4
