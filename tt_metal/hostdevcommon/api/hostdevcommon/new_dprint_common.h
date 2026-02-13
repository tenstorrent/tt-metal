// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

struct NewDebugPrintMemLayout {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    static constexpr uint32_t PROCESSOR_COUNT = static_cast<uint32_t>(EthProcessorTypes::COUNT);
#else
    static constexpr uint32_t PROCESSOR_COUNT = static_cast<uint32_t>(TensixProcessorTypes::COUNT);
#endif

    struct Aux {
        // current writer offset in buffer
        uint32_t wpos;
        uint32_t rpos;
        uint8_t kernel_printed[PROCESSOR_COUNT];  // Has kernel printed since starting
    } aux;
    static_assert(sizeof(Aux) % 4 == 0, "Aux struct must be a multiple of 4 bytes for proper alignment of data");
    uint8_t data[DPRINT_BUFFER_SIZE * PROCESSOR_COUNT - sizeof(Aux)];

} __attribute__((packed));

#endif
