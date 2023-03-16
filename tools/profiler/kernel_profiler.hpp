#pragma once

#include <climits>

#include "risc_common.h"
#include "hostdevcommon/profiler_common.h"

//#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER
#define PROFILER_OPTIONS KERNEL_FUNCT_MARKER | MAIN_FUNCT_MARKER

//TODO: Add mechanism for selecting PROFILER markers

namespace kernel_profiler{

    volatile uint32_t *buffer;
    uint32_t wIndex;

    void init_profiler()
    {
#ifdef PROFILE_KERNEL
        buffer = reinterpret_cast<uint32_t*>(get_debug_print_buffer());
        wIndex = MARKER_DATA_START;
        buffer [BUFFER_END_INDEX] = wIndex;
        buffer [DROPPED_MARKER_COUNTER] = 0;
#endif //PROFILE_KERNEL
    }

    inline void mark_time(uint32_t timer_id)
    {
#ifdef PROFILE_KERNEL
        uint32_t time_L = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read_barrier(RISCV_DEBUG_REG_WALL_CLOCK_H);

	if (wIndex + TIMER_DATA_UINT32_SIZE > PRINT_BUFFER_SIZE) {
            buffer [DROPPED_MARKER_COUNTER]++;
	    return;
	} else {
	    buffer[wIndex+TIMER_ID] = timer_id;
	    buffer[wIndex+TIMER_VAL_L] = time_L;
	    buffer[wIndex+TIMER_VAL_H] = time_H;
	}
        wIndex += TIMER_DATA_UINT32_SIZE;
        buffer [BUFFER_END_INDEX] = wIndex;
#endif //PROFILE_KERNEL
    }

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
