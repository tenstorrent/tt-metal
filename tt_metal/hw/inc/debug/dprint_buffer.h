// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Returns the buffer address for current thread+core. Differs for NC/BR/ER/TR0-2.
inline uint8_t* get_debug_print_buffer() {
    #if defined(COMPILE_FOR_NCRISC)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_NC);
    #elif defined(COMPILE_FOR_BRISC)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_BR);
    #elif defined(COMPILE_FOR_ERISC)
        return reinterpret_cast<uint8_t*>(eth_l1_mem::address_map::PRINT_BUFFER_ER);
    #elif defined(COMPILE_FOR_IDLE_ERISC)
        return reinterpret_cast<uint8_t*>(eth_l1_mem::address_map::PRINT_BUFFER_ER);
    #elif defined(UCK_CHLKC_UNPACK)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_T0);
    #elif defined(UCK_CHLKC_MATH)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_T1);
    #elif defined(UCK_CHLKC_PACK)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_T2);
    #endif
}
