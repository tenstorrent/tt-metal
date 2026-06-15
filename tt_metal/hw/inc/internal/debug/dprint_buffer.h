// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ENV_LLK_INFRA
#include "hostdev/dev_msgs.h"
#endif
// device_print_mem.h defines the arch's DevicePrintMemoryLayout (single buffer on
// WH/BH, a TRISC + DM split on Quasar). LLK infra reuses the same layout as Metal,
// just anchored at a fixed L1 address instead of the mailbox, so the dual-buffer
// behavior matches the dprint server.
#include "device_print_mem.h"
using DevicePrintBufferType = decltype(DevicePrintMemoryLayout::buffer);

inline volatile tt_l1_ptr DevicePrintBufferType* get_device_print_buffer() {
#ifdef ENV_LLK_INFRA
    // LLK places the layout at a fixed L1 base; must match DEVICE_PRINT_BUFFER_BASE in
    // tests/python_tests/helpers/test_config.py. `buffer` is this core's own buffer
    // (the TRISC buffer is first in the struct, so DM cores resolve to the second one).
    return &reinterpret_cast<volatile tt_l1_ptr DevicePrintMemoryLayout*>(LLK_DEVICE_PRINT_BUFFER_BASE)->buffer;
#else
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.buffer);
#endif
}
