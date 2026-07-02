// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ENV_LLK_INFRA
#include "hostdev/dev_msgs.h"
#endif
#include "device_print_mem.h"
using DevicePrintBufferType = decltype(DevicePrintMemoryLayout::buffer);

inline volatile tt_l1_ptr DevicePrintBufferType* get_device_print_buffer() {
#ifdef ENV_LLK_INFRA
    return &reinterpret_cast<volatile tt_l1_ptr DevicePrintMemoryLayout*>(LLK_DEVICE_PRINT_BUFFER_BASE)->buffer;
#else
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.buffer);
#endif
}
