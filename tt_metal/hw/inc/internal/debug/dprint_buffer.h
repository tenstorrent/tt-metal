// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENV_LLK_INFRA
#include "hostdev/device_print_common.h"
#else
#include "hostdev/dev_msgs.h"
#endif

#include "device_print_mem.h"

using DevicePrintBufferType = decltype(DevicePrintMemoryLayout::buffer);

inline volatile tt_l1_ptr DevicePrintBufferType* get_device_print_buffer() {
#ifdef ENV_LLK_INFRA
    // LLK has a different memory layout; this must match
    // DEVICE_PRINT_BUFFER_BASE in tests/python_tests/helpers/test_config.py.
    return reinterpret_cast<volatile tt_l1_ptr DevicePrintBufferType*>(LLK_DEVICE_PRINT_BUFFER_BASE);
#else
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.buffer);
#endif
}
