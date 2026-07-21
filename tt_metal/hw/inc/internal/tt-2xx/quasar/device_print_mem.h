// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdev/device_print_common.h"

struct DevicePrintMemoryLayout {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    // TODO: This needs to be properly defined
#if !defined(DEVICE_PRINT_BUFFER_SIZE)
    DevicePrintBuffer<408, 2> buffer;
#else
    DevicePrintBuffer<DEVICE_PRINT_BUFFER_SIZE, 2> buffer;
#endif
#elif defined(COMPILE_FOR_DRISC)
    // TODO: This needs to be properly defined
#if !defined(DEVICE_PRINT_BUFFER_SIZE)
    DevicePrintBuffer<204, 1> buffer;
#else
    DevicePrintBuffer<DEVICE_PRINT_BUFFER_SIZE, 1> buffer;
#endif
#else
#if !defined(DEVICE_PRINT_BUFFER_SIZE)
    static constexpr uint32_t buffer_size_triscs = 3264;
#else
    static constexpr uint32_t buffer_size_triscs = DEVICE_PRINT_BUFFER_SIZE;
#endif
#if !defined(DEVICE_PRINT_BUFFER_SIZE2)
    static constexpr uint32_t buffer_size_dms = 1632;
#else
    static constexpr uint32_t buffer_size_dms = DEVICE_PRINT_BUFFER_SIZE2;
#endif
#if defined(COMPILE_FOR_DM)
    DevicePrintBuffer<buffer_size_triscs, 16, 8> buffer_triscs;  // Quasar TRISC 16 processors
    DevicePrintBuffer<buffer_size_dms, 8, 0> buffer;             // Quasar DM 8 processors
#elif defined(COMPILE_FOR_TRISC) || defined(ENV_LLK_INFRA)  // Eventual LLK DM code will need to #define COMPILE_FOR_DM.
    DevicePrintBuffer<buffer_size_triscs, 16, 8> buffer;  // Quasar TRISC 16 processors
    DevicePrintBuffer<buffer_size_dms, 8, 0> buffer_dms;  // Quasar DM 8 processors
#else
    DevicePrintBuffer<buffer_size_triscs, 16, 8> buffer_triscs;  // Quasar TRISC 16 processors
    DevicePrintBuffer<buffer_size_dms, 8, 0> buffer_dms;         // Quasar DM 8 processors
#endif
#endif
};
