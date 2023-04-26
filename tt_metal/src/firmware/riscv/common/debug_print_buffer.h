#pragma once

// Returns the buffer address for current thread+core. Differs for NC/BR/TR0-2.
inline uint8_t* get_debug_print_buffer() {
    #if defined(COMPILE_FOR_NCRISC)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_NC);
    #elif defined(COMPILE_FOR_BRISC)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_BR);
    #elif defined(TRISC_UNPACK)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_T0);
    #elif defined(TRISC_MATH)
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_T1);
    #else
        return reinterpret_cast<uint8_t*>(PRINT_BUFFER_T2);
    #endif
}
