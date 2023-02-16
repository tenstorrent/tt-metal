/*
*
* Type ids shared between the debug print server and on-device debug prints.
*
*/

#pragma once

#include <cstddef>

constexpr int DEBUG_PRINT_TYPEID_CSTR    = 0;
constexpr int DEBUG_PRINT_TYPEID_ENDL    = 1; // std::endl
constexpr int DEBUG_PRINT_TYPEID_SETW    = 2; // std::setw
constexpr int DEBUG_PRINT_TYPEID_UINT32  = 3;
constexpr int DEBUG_PRINT_TYPEID_FLOAT32 = 4;
constexpr int DEBUG_PRINT_TYPEID_CHAR    = 5;
constexpr int DEBUG_PRINT_TYPEID_RAISE   = 6;
constexpr int DEBUG_PRINT_TYPEID_WAIT    = 7;
constexpr int DEBUG_PRINT_TYPEID_FLOAT16 = 8;
constexpr int DEBUG_PRINT_TYPEID_SETP    = 9; // std::setprecision
constexpr int DEBUG_PRINT_TYPEID_FIXP    = 10; // std::fixed
constexpr int DEBUG_PRINT_TYPEID_HEX     = 11; // std::hex
constexpr int DEBUG_PRINT_TYPEID_INT32   = 12;
constexpr int DEBUG_PRINT_TYPEID_TIMER   = 13;

// Because there's no mechanism (known to me) to initialize values at fixed mem locations in kernel code,
// in order to initialize the pointers in the buffers we use a trick with print server writing
// a magic value to a fixed location, we look for it on device, and only if it's present we initialize
// the read and write ptrs to 0 in DebugPrinter() constructor. This check is actually done every time
// a DebugPrinter() object is created (which can be many times), but results in resetting the pointers
// only once.
constexpr int DEBUG_PRINT_SERVER_STARTING_MAGIC = 0x12341234; 
constexpr int DEBUG_PRINT_SERVER_DISABLED_MAGIC = 0x23455432; 

// In case a single argument to operator << (such as a string) is larger than the buffer size
// (making it impossible to print) we will instead print this message.
constexpr const char* debug_print_overflow_error_message = "*** INTERNAL DEBUG PRINT BUFFER OVERFLOW ***\n\n";

constexpr int TIMER_PRINT_DATA_SIZE = 12;

struct DebugPrintMemLayout {
    struct Aux {
        // current writer offset in buffer
        uint32_t wpos __attribute__((aligned(4)));
        uint32_t rpos __attribute__((aligned(4)));
    } aux __attribute__((aligned(4)));
    uint8_t data[PRINT_BUFFER_SIZE-sizeof(Aux)];

    static size_t rpos_offs() { return offsetof(DebugPrintMemLayout::Aux, rpos) + offsetof(DebugPrintMemLayout, aux); }

} __attribute__((packed));

struct TimerPrintData {
    uint32_t timestamp_L __attribute__((aligned(4)));
    uint32_t timestamp_H __attribute__((aligned(4)));
    uint32_t id __attribute__((aligned(4)));
} __attribute__((packed));


static_assert(sizeof(TimerPrintData) == TIMER_PRINT_DATA_SIZE);
static_assert(sizeof(DebugPrintMemLayout) == PRINT_BUFFER_SIZE);
