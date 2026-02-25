// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/dprint_common.h"
#include "hostdev/dev_msgs.h"
#include "internal/hw_thread.h"

#include "hostdevcommon/dprint_common.h"
// Returns the buffer address for current thread+core. Differs for NC/BR/ER/TR0-2.
inline volatile tt_l1_ptr DebugPrintMemLayout* get_debug_print_buffer() {
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[internal_::get_hw_thread_idx()]);
}
