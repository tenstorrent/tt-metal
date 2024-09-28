// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <dev_msgs.h>

// Returns the buffer address for current thread+core. Differs for NC/BR/ER/TR0-2.
inline uint8_t* get_debug_print_buffer() {
    #if defined(COMPILE_FOR_NCRISC)
        return reinterpret_cast<uint8_t*>(GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_NC]));
    #elif defined(COMPILE_FOR_BRISC)
        return reinterpret_cast<uint8_t*>(GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_BR]));
    #elif defined(COMPILE_FOR_ERISC)
        return reinterpret_cast<uint8_t*>(GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_ER]));
    #elif defined(COMPILE_FOR_IDLE_ERISC)
        return reinterpret_cast<uint8_t*>(GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_ER]));
    #elif defined(UCK_CHLKC_UNPACK)
        return reinterpret_cast<uint8_t*>(GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_TR0]));
    #elif defined(UCK_CHLKC_MATH)
        return reinterpret_cast<uint8_t*>(GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_TR1]));
    #elif defined(UCK_CHLKC_PACK)
        return reinterpret_cast<uint8_t*>(GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[DPRINT_RISCV_INDEX_TR2]));
    #else
        return 0;
    #endif
}
