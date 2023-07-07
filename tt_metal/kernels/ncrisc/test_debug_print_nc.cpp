#include <cstdint>
#include "dataflow_kernel_api.h"
#include "debug_print.h"

using u32 = std::uint32_t;

// This kernel is a part of the test for device debug prints.
void kernel_main() {

    u32 dram_buffer_src_addr  = *((volatile u32*)(L1_ARG_BASE));
    u32 dram_src_noc_x        = *((volatile u32*)(L1_ARG_BASE+4));
    u32 dram_src_noc_y        = *((volatile u32*)(L1_ARG_BASE+8));
    u32 ARG0                  = *((volatile u32*)(L1_ARG_BASE+12));
    u32 ARG1                  = *((volatile u32*)(L1_ARG_BASE+16));
    u32 ARG2                  = *((volatile u32*)(L1_ARG_BASE+20));
    u32 ARG3                  = *((volatile u32*)(L1_ARG_BASE+24));
    u32 x                     = *((volatile u32*)(L1_ARG_BASE+28));
    u32 y                     = *((volatile u32*)(L1_ARG_BASE+32));
    u32 X                     = *((volatile u32*)(L1_ARG_BASE+36));

    if (ARG3 == 0xFFFF) {
        // special value for testing buffer overflow error message
        auto sz = sizeof(DebugPrintMemLayout::data);
        char overflow_buf[sz];
        for (unsigned j = 0; j < sz-1; j++)
            overflow_buf[j] = 'a';
        overflow_buf[sz-1] = 0;
        overflow_buf[sizeof(overflow_buf)-1] = 0;
        DPRINT << overflow_buf;
    } else {
        // This bit of code ensures that we print in the expected order (wrap around in x, incrementing y)
        // in the multi-core test
        if (ARG2 > 0 && x+y != 2) { // exclude {1,1} since it doesn't need to wait
            u32 wait_x = x-1;
            u32 wait_y = y;
            if (wait_x == 0) {
                // wrap around
                wait_x = X-1;
                wait_y = y-1;
            }
            DPRINT << WAIT{wait_x + wait_y*20 + 20000}; // wait on previous core's NC hart to raise a signal
        }

        DPRINT << SETW(ARG0, false) << ARG1;
        DPRINT << ENDL();
        // Using a string of size 17 here which is non-multiple of 4 (there was a bug with .rodata alignment for non-multiple of 4 strings)
        //         0123456789abcdef
        DPRINT << "TestConstCharStr";
        DPRINT << 'N' << 'C' << '{' << x << ',' << y << '}' << ENDL();
        DPRINT << SETP(4) << F32(0.123456f) << ENDL();
        DPRINT << FIXP() << F32(0.12f) << ENDL();
        DPRINT << BF16(0x3dfb) << ENDL(); // 0.12255859375
        for (u32 a = 0; a < ARG3; a++)
            DPRINT << '_';
        DPRINT << ENDL();
    }

    DPRINT << RAISE{x*5 + y*1000}; // in _br kernel we wait for this signal to sync debug print order on the host


    // TODO(AP): investigate this failure case - prints don't get through to debug server if kernel is stuck in an infinite loop
    // for(;;);

    // still need to produce one tile for the TR/reader
    constexpr u32 onetile = 1;
    constexpr u32 operand0 = 0;
    cb_reserve_back(operand0, onetile);
    u32 dest_tr0_l1 = get_write_ptr(operand0);
    cb_push_back(operand0, onetile);
}
