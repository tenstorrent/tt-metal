// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdev/device_print_common.h"

struct DevicePrintMemoryLayout {
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    // WH has 1 processor on ETH
#if !defined(DEVICE_PRINT_BUFFER_SIZE)
    // Sized so eth mailboxes_t, including go_messages[] of 8-byte go_msg_t, fits in the packed eth L1.
    DevicePrintBuffer<168, 1> buffer;
#else
    DevicePrintBuffer<DEVICE_PRINT_BUFFER_SIZE, 1> buffer;
#endif
#else
    // WH has 5 processors on TENSIX
#if !defined(DEVICE_PRINT_BUFFER_SIZE)
    DevicePrintBuffer<1020, 5> buffer;
#else
    DevicePrintBuffer<DEVICE_PRINT_BUFFER_SIZE, 5> buffer;
#endif
#endif
};
