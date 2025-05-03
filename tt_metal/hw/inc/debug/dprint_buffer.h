// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/dprint_common.h"
#include <dev_msgs.h>

#include "hostdevcommon/dprint_common.h"

// Returns the buffer address for current thread+core. Differs for NC/BR/ER/TR0-2.
inline volatile tt_l1_ptr DebugPrintMemLayout* get_debug_print_buffer() {
#if defined(COMPILE_FOR_NCRISC)
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_NC]);
#elif defined(COMPILE_FOR_BRISC)
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_BR]);
#elif defined(COMPILE_FOR_ERISC)
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_ER]);
#elif (defined(COMPILE_FOR_IDLE_ERISC) && COMPILE_FOR_IDLE_ERISC == 0)
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_ER]);
#elif (defined(COMPILE_FOR_IDLE_ERISC) && COMPILE_FOR_IDLE_ERISC == 1)
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_ER1]);
#elif defined(UCK_CHLKC_UNPACK)
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_TR0]);
#elif defined(UCK_CHLKC_MATH)
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_TR1]);
#elif defined(UCK_CHLKC_PACK)
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_TR2]);
#else
    return 0;
#endif
}
