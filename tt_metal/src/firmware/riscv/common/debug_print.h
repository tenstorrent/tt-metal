#pragma once

/*
* Device-side debug print API for device kernels.
* Works on either one of NC/BR/TR threads.
* On device the use is as follows:
*
* DPRINT << SETW(2) << 0 << 0.1f << "string" << ENDL();
*
* This DebugPrinter object can be created multiple times.
*
* On the host it's required to start the print server first, otherwise the behavior will be incorrect.
* This is because the host print server writes a special value that is used in DebugPrinter() constructor
* to initialize the read/write pointers to 0 only once.
* It is also needed to empty the print buffer, otherwise device code will stall waiting on the host to flush it.
*
* Use llrt/tt_debug_print_server.h APIs to start the host-side print server.
*
*/

#include <cstdint>
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
// TODO(AP): this ifdef doesn't seem to make sense given we include risc_common.h
// The issue is some files included inside risc_common.h only apply to NC/BRISCS
// But moving this ifdef inside of the header breaks other code
// So there are some not fully decoupled dependencies in this header.
#include "risc_common.h"
#endif
#include "hostdevcommon/debug_print_common.h"
#include "hostdevcommon/common_runtime_address_map.h"

#include "debug_print_buffer.h"

#define DPRINT DebugPrinter()

struct BF16 { uint16_t val; BF16(uint16_t val) : val(val) {} } ATTR_PACK;
struct F32  { float val; F32(float val) : val(val) {} } ATTR_PACK;
struct U32  { uint32_t val; U32(uint32_t val) : val(val) {} } ATTR_PACK;

struct ENDL { char tmp; } ATTR_PACK; // Analog of cout << std::endl - not making it zero size to avoid special cases
struct SETP { char p; SETP(char pa) : p(pa) {} } ATTR_PACK; // Analog of cout << std::setprecision()
struct FIXP { char tmp; } ATTR_PACK; // Analog of cout << std::fixed
struct HEX  { char tmp; } ATTR_PACK; // Analog of cout << std::hex
struct SETW {
    char w;
    SETW(char wa, bool sticky = true) { w = wa; if (sticky) w |= 0b10000000; }
} ATTR_PACK; // Analog of cout << std::setw(), defaults to sticky TODO(AP): 2 chars didn't work

// These primitives are intended for ordering debug prints
// A possible use here is to synchronize debug print order between cores/harts
// It could be implemented, for instance as code = linearize({x,y})*5 + hart_id
// With another core/hart waiting on that index
struct RAISE { uint32_t code; RAISE(uint32_t val) : code(val) {} } ATTR_PACK; // raise a condition with specified code
struct WAIT { uint32_t code; WAIT(uint32_t val) : code(val) {} } ATTR_PACK; // wait for a condition with specified code

// didn't want to include string.h
inline uint32_t DebugPrintStrLen(const char* val) {
    const char* end = val;
    while (*end) { end++; };
    return uint32_t(end-val)+1;
}

inline uint32_t DebugPrintStrCopy(char* dst, const char* src) {
    uint32_t len = DebugPrintStrLen(src);
    for (uint32_t j = 0; j < len; j++)
        dst[j] = src[j];
    return len;
}

// Extend with new type id here, each new type needs specializations for 1 (or 3) of these functions below:
// This template instantiation maps from type to type id to send over our comm channel
template<typename T> uint8_t DebugPrintTypeToId();
template<typename T> uint32_t DebugPrintTypeToSize(T val) { return sizeof(T); };
template<typename T> const uint8_t* DebugPrintTypeAddr(T* val) { return reinterpret_cast<const uint8_t*>(val); }

template<> uint8_t DebugPrintTypeToId<const char*>()   { return DEBUG_PRINT_TYPEID_CSTR; }
template<> uint8_t DebugPrintTypeToId<char*>()         { return DEBUG_PRINT_TYPEID_CSTR; }
template<> uint8_t DebugPrintTypeToId<ENDL>()          { return DEBUG_PRINT_TYPEID_ENDL; }
template<> uint8_t DebugPrintTypeToId<SETW>()          { return DEBUG_PRINT_TYPEID_SETW; }
template<> uint8_t DebugPrintTypeToId<uint32_t>()      { return DEBUG_PRINT_TYPEID_UINT32; }
template<> uint8_t DebugPrintTypeToId<float>()         { return DEBUG_PRINT_TYPEID_FLOAT32; }
template<> uint8_t DebugPrintTypeToId<char>()          { return DEBUG_PRINT_TYPEID_CHAR; }
template<> uint8_t DebugPrintTypeToId<RAISE>()         { return DEBUG_PRINT_TYPEID_RAISE; }
template<> uint8_t DebugPrintTypeToId<WAIT>()          { return DEBUG_PRINT_TYPEID_WAIT; }
template<> uint8_t DebugPrintTypeToId<BF16>()          { return DEBUG_PRINT_TYPEID_BFLOAT16; }
template<> uint8_t DebugPrintTypeToId<SETP>()          { return DEBUG_PRINT_TYPEID_SETP; }
template<> uint8_t DebugPrintTypeToId<FIXP>()          { return DEBUG_PRINT_TYPEID_FIXP; }
template<> uint8_t DebugPrintTypeToId<HEX>()           { return DEBUG_PRINT_TYPEID_HEX; }
template<> uint8_t DebugPrintTypeToId<F32>()           { return DEBUG_PRINT_TYPEID_FLOAT32; }
template<> uint8_t DebugPrintTypeToId<U32>()           { return DEBUG_PRINT_TYPEID_UINT32; }
template<> uint8_t DebugPrintTypeToId<int>()           { return DEBUG_PRINT_TYPEID_INT32; }
template<> uint8_t DebugPrintTypeToId<uint64_t>()      { return DEBUG_PRINT_TYPEID_UINT64; }
static_assert(sizeof(int) == 4);

// Specializations for const char* (string literals), typically you will not need these for other types
template<> uint32_t       DebugPrintTypeToSize<const char*>(const char* val) { return DebugPrintStrLen(val); } // also copy the terminating zero
template<> const uint8_t* DebugPrintTypeAddr<const char*>(const char** val)  { return reinterpret_cast<const uint8_t*>(*val); }
template<> uint32_t       DebugPrintTypeToSize<char*>(char* val)             { return DebugPrintStrLen(val); } // also copy the terminating zero
template<> const uint8_t* DebugPrintTypeAddr<char*>(char** val)              { return reinterpret_cast<const uint8_t*>(*val); }


struct DebugPrinter {
    volatile tt_l1_ptr uint32_t* wpos() {
        auto printbuf = get_debug_print_buffer();
        return &reinterpret_cast<DebugPrintMemLayout*>(printbuf)->aux.wpos;
    }
    volatile tt_l1_ptr uint32_t* rpos() {
        auto printbuf = get_debug_print_buffer();
        return &reinterpret_cast<DebugPrintMemLayout*>(printbuf)->aux.rpos;
    }
    uint8_t* buf() { return get_debug_print_buffer(); }
    uint8_t* data() { return reinterpret_cast<DebugPrintMemLayout*>(buf())->data; }
    uint8_t* bufend() { return buf() + PRINT_BUFFER_SIZE; }

    DebugPrinter() {
#ifndef PROFILE_KERNEL
        if (*wpos() == DEBUG_PRINT_SERVER_STARTING_MAGIC) {
            // Host debug print server writes this value
            // we don't want to reset wpos/rpos to 0 unless this is the first time
            // DebugPrinter() is created (even across multiple kernel calls)
            *wpos() = 0;
            *rpos() = 0;
        }
#endif //PROFILE_KERNEL
    }
};

template<typename T>
__attribute__((__noinline__))
DebugPrinter operator <<(DebugPrinter dp, T val) {

#ifndef PROFILE_KERNEL
    if (*dp.wpos() == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
        // skip all prints if this hart+core was not specifically enabled on the host
        return dp;
    }

    uint32_t payload_sz = DebugPrintTypeToSize<T>(val); // includes terminating 0 for char*
    uint8_t typecode = DebugPrintTypeToId<T>();

    constexpr int code_sz = 1; // size of type code
    constexpr int sz_sz = 1; // size of serialized size
    uint32_t wpos = *dp.wpos(); // copy wpos into local storage
    auto sum_sz = payload_sz + code_sz + sz_sz;
    if (dp.data() + wpos + sum_sz >= dp.bufend()) {
        // buffer is full - wait for the host reader to flush+update rpos
        while (*dp.rpos() < *dp.wpos())
            ; // wait for host to catch up to wpos with it's rpos
        *dp.wpos() = 0;
        // TODO(AP): are these writes guaranteed to be ordered?
        *dp.rpos() = 0;
        wpos = 0;
        if (payload_sz >= sizeof(DebugPrintMemLayout::data)-2) {
            // Handle a special case - this value cannot be printed because it cannot fit in the buffer.
            // -2 is for code_sz and sz_sz.
            // Note that the outer if is definitely also true if we got to this inner if.
            // In this case we replace the input value with debug error message.
            // We cannot recursively call operator << from here because it hasn't been defined yet
            // so there's a bit of code duplication here for this special case
            // Another possibility is to wait for the device to flush and print the string piecemeal.
            // As a negative side effect,
            // unfortunately this special case increases the code size generated for each instance of <<.
            uint8_t* printbuf = dp.data();
            payload_sz = DebugPrintStrCopy(
                reinterpret_cast<char*>(printbuf+code_sz+sz_sz),
                debug_print_overflow_error_message);
            printbuf[0] = DEBUG_PRINT_TYPEID_CSTR;
            printbuf[code_sz] = payload_sz;
            wpos = payload_sz + sz_sz + code_sz;
            *dp.wpos() = wpos;
            return dp;
        }
    }

    uint8_t* printbuf = dp.data();
    // no need for a circular buffer since perf is not critical
    printbuf[wpos] = typecode;
    wpos += code_sz;
    printbuf[wpos] = payload_sz;
    wpos += sz_sz;
    const uint8_t* valaddr = DebugPrintTypeAddr<T>(&val);
    for (uint32_t j = 0; j < payload_sz; j++)
        printbuf[wpos+j] = valaddr[j];
    wpos += payload_sz;

    // our message needs to be atomic w.r.t code, size and payload
    // so we only update wpos in the end
    *dp.wpos() = wpos;
#endif //PROFILE_KERNEL
    return dp;
}

// explicit instantiations of operator<<
template DebugPrinter operator<< <const char*>(DebugPrinter dp, const char* val);
template DebugPrinter operator<< <ENDL>(DebugPrinter, ENDL val);
template DebugPrinter operator<< <SETW>(DebugPrinter, SETW val);
template DebugPrinter operator<< <uint32_t>(DebugPrinter, uint32_t val);
template DebugPrinter operator<< <float>(DebugPrinter, float val);
template DebugPrinter operator<< <char>(DebugPrinter, char val);
template DebugPrinter operator<< <RAISE>(DebugPrinter, RAISE val);
template DebugPrinter operator<< <WAIT>(DebugPrinter, WAIT val);
template DebugPrinter operator<< <FIXP>(DebugPrinter, FIXP val);
template DebugPrinter operator<< <HEX>(DebugPrinter, HEX val);
template DebugPrinter operator<< <SETP>(DebugPrinter, SETP val);
template DebugPrinter operator<< <BF16>(DebugPrinter, BF16 val);
template DebugPrinter operator<< <F32>(DebugPrinter, F32 val);
template DebugPrinter operator<< <U32>(DebugPrinter, U32 val);

#include "debug_print_tile.h"
#include "debug_print_core_xy.h"
