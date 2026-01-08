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

union subordinate_map_t {
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
        volatile uint8_t neo0Trisc0;
        volatile uint8_t neo0Trisc1;
        volatile uint8_t neo0Trisc2;
        volatile uint8_t neo0Trisc3;
        volatile uint8_t neo1Trisc0;
        volatile uint8_t neo1Trisc1;
        volatile uint8_t neo1Trisc2;
        volatile uint8_t neo1Trisc3;
        volatile uint8_t neo2Trisc0;
        volatile uint8_t neo2Trisc1;
        volatile uint8_t neo2Trisc2;
        volatile uint8_t neo2Trisc3;
        volatile uint8_t neo3Trisc0;
        volatile uint8_t neo3Trisc1;
        volatile uint8_t neo3Trisc2;
        volatile uint8_t neo3Trisc3;
        uint8_t pad[12];
    };
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
