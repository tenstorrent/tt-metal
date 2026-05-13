// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/dprint_common.h"
#if !defined(ENV_LLK_INFRA)
#include "hostdev/dev_msgs.h"
#include "internal/hw_thread.h"

// Returns the buffer address for current thread+core. Differs for NC/BR/ER/TR0-2.
inline volatile tt_l1_ptr DebugPrintMemLayout* get_debug_print_buffer() {
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[internal_::get_hw_thread_idx()]);
}
#endif

inline volatile tt_l1_ptr DevicePrintMemoryLayout* get_device_print_buffer() {
#ifdef ENV_LLK_INFRA
    // LLK has a different memory layout; this must match
    // DEVICE_PRINT_BUFFER_BASE in tests/python_tests/helpers/test_config.py.
    return reinterpret_cast<volatile tt_l1_ptr DevicePrintMemoryLayout*>(LLK_DEVICE_PRINT_BUFFER_BASE);
#else
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.shared_data);
#endif
}
