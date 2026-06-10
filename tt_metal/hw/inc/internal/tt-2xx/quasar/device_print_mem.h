// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdev/device_print_common.h"

struct DevicePrintMemoryLayout {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    // TODO: This needs to be properly defined
    DevicePrintBuffer<408, 2> buffer;
#elif defined(COMPILE_FOR_DRISC)
    // TODO: This needs to be properly defined
    DevicePrintBuffer<204, 1> buffer;
#else
#if defined(COMPILE_FOR_DM)
    DevicePrintBuffer<3264, 16, 8> buffer_triscs;  // Quasar TRISC 16 processors
    DevicePrintBuffer<1632, 8, 0> buffer;          // Quasar DM 8 processors
#elif defined(COMPILE_FOR_TRISC)
    DevicePrintBuffer<3264, 16, 8> buffer;     // Quasar TRISC 16 processors
    DevicePrintBuffer<1632, 8, 0> buffer_dms;  // Quasar DM 8 processors
#else
    DevicePrintBuffer<3264, 16, 8> buffer_triscs;  // Quasar TRISC 16 processors
    DevicePrintBuffer<1632, 8, 0> buffer_dms;      // Quasar DM 8 processors
#endif
#endif
};
