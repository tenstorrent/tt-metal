// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/dprint_common.h"
#include "hostdev/dev_msgs.h"

#include "hostdevcommon/dprint_common.h"
// Returns the buffer address for current thread+core. Differs for NC/BR/ER/TR0-2.
inline volatile tt_l1_ptr DebugPrintMemLayout* get_debug_print_buffer() {
#if defined(ARCH_QUASAR)
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[hartid]);
#else
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[PROCESSOR_INDEX]);
#endif
}
