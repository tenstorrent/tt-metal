// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdev/dev_msgs.h"

inline volatile tt_l1_ptr DevicePrintMemoryLayout* get_device_print_buffer() {
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf);
}
