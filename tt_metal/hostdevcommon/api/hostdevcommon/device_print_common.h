// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 *
 * Type ids shared between the debug print server and on-device debug prints.
 *
 */

#pragma once

#if defined(KERNEL_BUILD) || defined(FW_BUILD) || defined(HAL_BUILD)

#include "core_config.h"

enum class DevicePrintRiscCoreState : uint8_t {
    KernelNotPrinted = 0,
    KernelPrinted = 1,
    PrintingDisabled = 2,
};

struct DevicePrintMemoryLayout {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    static constexpr uint32_t PROCESSOR_COUNT = static_cast<uint32_t>(EthProcessorTypes::COUNT);
#else
    static constexpr uint32_t PROCESSOR_COUNT = static_cast<uint32_t>(TensixProcessorTypes::COUNT);
#endif

    struct Aux {
        // current writer offset in buffer
        uint32_t wpos;
        uint32_t rpos;
        DevicePrintRiscCoreState risc_state[PROCESSOR_COUNT];  // Has kernel printed since starting
    } aux;
    static_assert(
        sizeof(Aux) == 4 + 4 + (PROCESSOR_COUNT * sizeof(DevicePrintRiscCoreState) + 3) / 4 * 4,
        "Aux struct size must be correct");
    static_assert(sizeof(Aux) % 4 == 0, "Aux struct must be a multiple of 4 bytes for proper alignment of data");
    uint8_t data[DPRINT_BUFFER_SIZE * PROCESSOR_COUNT - sizeof(Aux)];
    static_assert(sizeof(data) % 4 == 0, "Data array size must be a multiple of 4 bytes for proper alignment");
};
static_assert(
    sizeof(DevicePrintMemoryLayout) == DPRINT_BUFFER_SIZE * DevicePrintMemoryLayout::PROCESSOR_COUNT,
    "DevicePrintMemoryLayout size must match total buffer size");
static_assert(
    sizeof(DevicePrintMemoryLayout) % 4 == 0,
    "DevicePrintMemoryLayout size must be a multiple of 4 bytes for proper alignment");

#endif
