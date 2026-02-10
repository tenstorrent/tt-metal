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
    struct Aux {
        // current writer offset in buffer
        uint32_t wpos;
        uint32_t rpos;
    } aux __attribute__((packed));
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    static constexpr uint32_t PROCESSOR_COUNT = static_cast<uint32_t>(EthProcessorTypes::COUNT);
#else
    static constexpr uint32_t PROCESSOR_COUNT = static_cast<uint32_t>(TensixProcessorTypes::COUNT);
#endif
    uint8_t data[DPRINT_BUFFER_SIZE * PROCESSOR_COUNT - sizeof(Aux)];

} __attribute__((packed));

#endif
