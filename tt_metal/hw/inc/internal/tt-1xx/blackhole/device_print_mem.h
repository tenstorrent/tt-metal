// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdev/device_print_common.h"

struct DevicePrintMemoryLayout {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    DevicePrintBuffer<408, 2> buffer;  // BH ETH 2 processors
#elif defined(COMPILE_FOR_DRISC)
    DevicePrintBuffer<204, 1> buffer;  // BH DRAM 1 processor
#else
    DevicePrintBuffer<1020, 5> buffer;  // BH TENSIX 5 processors
#endif
};
