#pragma once

#include <climits>

#include "risc_common.h"

#define MAIN_FUNCT_MARKER   (1U << 0)
#define KERNEL_FUNCT_MARKER (1U << 1)

#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER
//#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER | MAIN_FUNCT_MARKER

//TODO: Add mechanism for selecting PROFILER markers

#define CC_MAIN_START          1U
#define CC_KERNEL_MAIN_START   2U
#define CC_KERNEL_MAIN_END     3U
#define CC_MAIN_END            4U

namespace kernel_profiler{

    volatile uint32_t *buffer;
    uint32_t w_index;
    void init_profiler()
    {
        buffer = reinterpret_cast<uint32_t*>(get_debug_print_buffer());
        w_index = 1;
        buffer [0] = w_index;
    }

    inline void mark_time(uint32_t timer_id)
    {
#ifdef PROFILE_KERNEL

        uint32_t time_L = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);

	if (w_index + 3 > PRINT_BUFFER_SIZE) { // wpos is *volatile uint32 ptr
	    return;
	} else {
	    buffer[w_index++] = timer_id;
	    buffer[w_index++] = time_L;
	    buffer[w_index++] = time_H;
	}
        buffer[0] = w_index;

#endif //PROFILE_KERNEL
    }

    //TODO: Due to the issue with linker placing rodata and uninitialized
    // vars in far away addresses, making the hex to be 4GB, using
    // static variables causes the hex dispatch to hang
    // one_time bool should become a static to this function after
    // the fix and the function call and static one_time variable
    // will be turned into a macro call
    inline void mark_time_once(uint32_t timer_id, bool * one_time)
    {
#ifdef PROFILE_KERNEL
        if (*one_time)
        {
            mark_time(timer_id);
        }
        *one_time = false;
#endif //PROFILE_KERNEL
    }
}
