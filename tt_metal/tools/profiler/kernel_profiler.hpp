#pragma once


#include <climits>

#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
#include "risc_common.h"
#else
#include "ckernel.h"
#endif

#ifdef PROFILE_KERNEL
#include "debug_print_buffer.h" // only needed because the address is shared
#endif

#include "hostdevcommon/profiler_common.h"
#include "src/firmware/riscv/common/risc_attribs.h"


//#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER
#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER | MAIN_FUNCT_MARKER

//TODO: Add mechanism for selecting PROFILER markers

namespace kernel_profiler{

    /*
     * For TRISCs the kernel_profiler will be in multiple translation units
     * if custom kernel markers are used. Namespace variables are inline'd
     * so that they don't confuse the linker.
     * https://pabloariasal.github.io/2019/02/28/cpp-inlining/
     *
     * */
    extern uint32_t wIndex;

    inline __attribute__((always_inline)) void init_profiler()
    {
#if defined(PROFILE_KERNEL)
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_debug_print_buffer());
        wIndex = MARKER_DATA_START;
        buffer [BUFFER_END_INDEX] = wIndex;
        buffer [DROPPED_MARKER_COUNTER] = 0;
#endif //PROFILE_KERNEL
    }

    /*
     *  Profiler init function only for BRISC core.
     *  BRISC will always and this init guarantees that other profiler regions
     *  are initialized with proper values when read by the host even if that
     *  risc does not execute any FW.
     * */
    inline __attribute__((always_inline)) void init_BR_profiler()
    {
#if defined(PROFILE_KERNEL) && defined(COMPILE_FOR_BRISC)
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_debug_print_buffer());
        buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_NC);
        buffer [BUFFER_END_INDEX] = MARKER_DATA_START;
        buffer [DROPPED_MARKER_COUNTER] = 0;
        buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_T0);
        buffer [BUFFER_END_INDEX] = MARKER_DATA_START;
        buffer [DROPPED_MARKER_COUNTER] = 0;
        buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_T1);
        buffer [BUFFER_END_INDEX] = MARKER_DATA_START;
        buffer [DROPPED_MARKER_COUNTER] = 0;
        buffer = reinterpret_cast<uint32_t*>(PRINT_BUFFER_T2);
        buffer [BUFFER_END_INDEX] = MARKER_DATA_START;
        buffer [DROPPED_MARKER_COUNTER] = 0;

        init_profiler();
#endif
    }

    inline __attribute__((always_inline)) void mark_time(uint32_t timer_id)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif

        // Either buffer has room for more markers or the end of FW marker is place on the last marker spot
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_debug_print_buffer());
	if (((wIndex + (2*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
            ((timer_id == CC_MAIN_END) && !((wIndex + TIMER_DATA_UINT32_SIZE) > (PRINT_BUFFER_SIZE/sizeof(uint32_t))))) {
	    buffer[wIndex+TIMER_ID] = timer_id;
	    buffer[wIndex+TIMER_VAL_L] = time_L;
	    buffer[wIndex+TIMER_VAL_H] = time_H;
            wIndex += TIMER_DATA_UINT32_SIZE;
            buffer [BUFFER_END_INDEX] = wIndex;
	} else {
            buffer [DROPPED_MARKER_COUNTER]++;
	}
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time_once(uint32_t timer_id, bool * one_time)
    {
#if defined(PROFILE_KERNEL)
        if (*one_time)
        {
            mark_time(timer_id);
        }
        *one_time = false;
#endif //PROFILE_KERNEL
    }
}
