#pragma once

#define MAIN_FUNCT_MARKER   (1U << 0)
#define KERNEL_FUNCT_MARKER (1U << 1)

#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER | MAIN_FUNCT_MARKER

//TODO: Add mechanism for selecting PROFILER markers

#include <climits>

#include "debug_print.h"

#define CC_MAIN_START          1U
#define CC_KERNEL_MAIN_START   2U
#define CC_KERNEL_MAIN_END     3U
#define CC_MAIN_END            4U

namespace kernel_profiler{

    inline void mark_time(uint32_t timer_id)
    {
#ifdef PROFILE_KERNEL
        TimerPrintData timer_data;

#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif


        timer_data.timestamp_L = time_L;
        timer_data.timestamp_H = time_H;
        timer_data.id = timer_id;

        DPRINT << timer_data;
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
