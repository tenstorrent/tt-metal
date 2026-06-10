// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdev/device_print_common.h"

struct DevicePrintMemoryLayout {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    DevicePrintBuffer<204, 1> buffer;  // WH ETH 1 processor
#else
    DevicePrintBuffer<1020, 5> buffer;  // WH TENSIX 5 processors
#endif
};
