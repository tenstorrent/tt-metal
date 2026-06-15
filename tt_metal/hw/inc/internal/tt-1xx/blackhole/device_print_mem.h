// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdev/device_print_common.h"

struct DevicePrintMemoryLayout {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    // BH has 2 processors on ETH
#if !defined(DEVICE_PRINT_BUFFER_SIZE)
    DevicePrintBuffer<408, 2> buffer;
#else
    DevicePrintBuffer<DEVICE_PRINT_BUFFER_SIZE, 2> buffer;
#endif
#elif defined(COMPILE_FOR_DRISC)
    // BH has 1 processor on DRAM
#if !defined(DEVICE_PRINT_BUFFER_SIZE)
    DevicePrintBuffer<204, 1> buffer;
#else
    DevicePrintBuffer<DEVICE_PRINT_BUFFER_SIZE, 1> buffer;
#endif
#else
    // BH has 5 processors on TENSIX
#if !defined(DEVICE_PRINT_BUFFER_SIZE)
    DevicePrintBuffer<1020, 5> buffer;
#else
    DevicePrintBuffer<DEVICE_PRINT_BUFFER_SIZE, 5> buffer;
#endif
#endif
};
