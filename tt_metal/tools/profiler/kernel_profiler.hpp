// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <climits>

#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
#include "risc_common.h"
#else
#include "ckernel.h"
#endif

#ifdef PROFILE_KERNEL
#include "debug/dprint_buffer.h" // only needed because the address is shared
#endif

#include "hostdevcommon/profiler_common.h"
#include "risc_attribs.h"

namespace kernel_profiler{

    extern uint32_t wIndex;
    extern uint32_t device_function_sums[GLOBAL_SUM_COUNT];
    extern uint64_t device_function_starts[GLOBAL_SUM_COUNT];

    inline __attribute__((always_inline)) void init_profiler()
    {
#if defined(PROFILE_KERNEL)
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_debug_print_buffer());
        wIndex = MARKER_DATA_START;
        buffer [BUFFER_END_INDEX] = wIndex;
        buffer [DROPPED_MARKER_COUNTER] = 0;
        for (uint32_t i=0; i< GLOBAL_SUM_COUNT; i++)
        {
            device_function_sums[i] = 0;
            device_function_starts[i] = 0;
        }
#if defined(COMPILE_FOR_BRISC)
        buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PRINT_BUFFER_NC);
        buffer [BUFFER_END_INDEX] = wIndex;
        buffer [DROPPED_MARKER_COUNTER] = 0;

        buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PRINT_BUFFER_T0);
        buffer [BUFFER_END_INDEX] = wIndex;
        buffer [DROPPED_MARKER_COUNTER] = 0;

        buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PRINT_BUFFER_T1);
        buffer [BUFFER_END_INDEX] = wIndex;
        buffer [DROPPED_MARKER_COUNTER] = 0;

        buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PRINT_BUFFER_T2);
        buffer [BUFFER_END_INDEX] = wIndex;
        buffer [DROPPED_MARKER_COUNTER] = 0;
#elif defined(COMPILE_FOR_ERISC)
        buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_l1_mem::address_map::PRINT_BUFFER_ER);
        buffer[BUFFER_END_INDEX] = wIndex;
        buffer[DROPPED_MARKER_COUNTER] = 0;
#endif
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_function_sum_start(uint32_t function_id)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif

        device_function_starts[function_id] = ((uint64_t) time_H) << 32 | time_L;
#endif
    }

    inline __attribute__((always_inline)) void mark_function_sum_end(uint32_t function_id)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif

        device_function_sums[function_id] += (((uint64_t) time_H) << 32 | time_L) - device_function_starts[function_id];
#endif
    }

    inline __attribute__((always_inline)) void store_function_sums()
    {
#if defined(PROFILE_KERNEL)
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_debug_print_buffer());
        for (uint32_t i = 0; i < GLOBAL_SUM_COUNT; i++)
        {
            if (device_function_sums[i] > 0)
            {
                if ((wIndex + (3*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))){
                    buffer[wIndex+TIMER_ID] = i + GLOBAL_SUM_MARKER;
                    buffer[wIndex+TIMER_VAL_L] = device_function_sums[i];
                    buffer[wIndex+TIMER_VAL_H] = 0;
                    wIndex += TIMER_DATA_UINT32_SIZE;
                    buffer [BUFFER_END_INDEX] = wIndex;
                } else {
                    buffer [DROPPED_MARKER_COUNTER]++;
                }
            }
        }
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time(uint32_t timer_id)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif

        // Either buffer has room for more markers or the end of FW marker is place on the last marker spot
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_debug_print_buffer());
	if (((wIndex + (3*TIMER_DATA_UINT32_SIZE)) < (PRINT_BUFFER_SIZE/sizeof(uint32_t))) ||\
            (((timer_id == CC_MAIN_END) || (timer_id == CC_KERNEL_MAIN_END)) &&\
             !((wIndex + TIMER_DATA_UINT32_SIZE) > (PRINT_BUFFER_SIZE/sizeof(uint32_t))))) {
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
