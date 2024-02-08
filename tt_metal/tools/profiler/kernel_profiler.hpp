// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//#define PROFILER_KERNEL_FORCE_INLINE

#include <climits>

#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
#include "risc_common.h"
#include "dataflow_api.h"
#else
#include "ckernel.h"
#endif

#ifdef PROFILER_KERNEL_FORCE_INLINE
#define PROFILER_INLINE inline __attribute__((always_inline))
#else
#define PROFILER_INLINE
#endif

#ifdef PROFILE_KERNEL
#include "debug/dprint_buffer.h" // only needed because the address is shared
#endif

#include "hostdevcommon/profiler_common.h"
#include "risc_attribs.h"

namespace kernel_profiler{

    extern uint32_t wIndex;
    extern uint32_t stackSize;

#if defined(COMPILE_FOR_BRISC)
    const uint32_t profilerBuffer = PROFILER_L1_BUFFER_BR;
    const uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_BR;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#elif defined(COMPILE_FOR_ERISC)
    const uint32_t profilerBuffer = eth_l1_mem::address_map::PROFILER_L1_BUFFER_ER;
    const uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_ER;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eth_l1_mem::address_map::PROFILER_L1_BUFFER_CONTROL);
#elif defined(COMPILE_FOR_NCRISC)
    const uint32_t profilerBuffer = PROFILER_L1_BUFFER_NC;
    const uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_NC;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#elif COMPILE_FOR_TRISC == 0
    const uint32_t profilerBuffer = PROFILER_L1_BUFFER_T0;
    const uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T0;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#elif COMPILE_FOR_TRISC == 1
    const uint32_t profilerBuffer = PROFILER_L1_BUFFER_T1;
    const uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T1;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#elif COMPILE_FOR_TRISC == 2
    const uint32_t profilerBuffer = PROFILER_L1_BUFFER_T2;
    const uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T2;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#endif

    void init_profiler(uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0)
    {
#if defined(PROFILE_KERNEL)
        wIndex = CUSTOM_MARKERS;

#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_BRISC)
        uint32_t noc_x = my_x[0];
        uint32_t noc_y = my_y[0];
        uint32_t runCounter = profiler_control_buffer[RUN_COUNTER];

        uint16_t core_flat_id = noc_xy_to_profiler_flat_id[noc_x][noc_y];

#if defined(COMPILE_FOR_ERISC)
        volatile tt_l1_ptr uint32_t *eriscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eth_l1_mem::address_map::PROFILER_L1_BUFFER_ER);

#pragma GCC unroll 65534
        for (int i = ID_HH; i < GUARANTEED_MARKER_1_H; i ++)
        {
            eriscBuffer[i] = 0;
        }

#pragma GCC unroll 65534
        for (int i = GUARANTEED_MARKER_1_H; i < CUSTOM_MARKERS; i ++)
        {
        //TODO(MO): Clean up magic numbers
            eriscBuffer[i] = 0x80000000;
        }

        //TODO(MO): Clean up magic numbers
        eriscBuffer [ID_LL] = runCounter;

        eriscBuffer [ID_LH] = ((core_flat_id & 0xFF) << 3) | 0;
#endif //ERISC_INIT

#if defined(COMPILE_FOR_BRISC)
        volatile tt_l1_ptr uint32_t *briscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_BR);
        volatile tt_l1_ptr uint32_t *ncriscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_NC);
        volatile tt_l1_ptr uint32_t *trisc0Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T0);
        volatile tt_l1_ptr uint32_t *trisc1Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T1);
        volatile tt_l1_ptr uint32_t *trisc2Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T2);

#pragma GCC unroll 65534
        for (int i = ID_HH; i < GUARANTEED_MARKER_1_H; i ++)
        {
            briscBuffer[i] = 0;
            ncriscBuffer[i] = 0;
            trisc0Buffer[i] = 0;
            trisc1Buffer[i] = 0;
            trisc2Buffer[i] = 0;
        }

#pragma GCC unroll 65534
        for (int i = GUARANTEED_MARKER_1_H; i < CUSTOM_MARKERS; i ++)
        {
        //TODO(MO): Clean up magic numbers
            briscBuffer[i] = 0x80000000;
            ncriscBuffer[i] = 0x80000000;
            trisc0Buffer[i] = 0x80000000;
            trisc1Buffer[i] = 0x80000000;
            trisc2Buffer[i] = 0x80000000;
        }

        //TODO(MO): Clean up magic numbers
        briscBuffer [ID_LL] = runCounter;
        ncriscBuffer[ID_LL] = runCounter;
        trisc0Buffer[ID_LL] = runCounter;
        trisc1Buffer[ID_LL] = runCounter;
        trisc2Buffer[ID_LL] = runCounter;

        briscBuffer [ID_LH] = ((core_flat_id & 0xFF) << 3) | 0;
        ncriscBuffer[ID_LH] = ((core_flat_id & 0xFF) << 3) | 1;
        trisc0Buffer[ID_LH] = ((core_flat_id & 0xFF) << 3) | 2;
        trisc1Buffer[ID_LH] = ((core_flat_id & 0xFF) << 3) | 3;
        trisc2Buffer[ID_LH] = ((core_flat_id & 0xFF) << 3) | 4;
#endif //BRISC_INIT
#endif

#endif //PROFILE_KERNEL
    }


    PROFILER_INLINE void mark_time_at_index(uint32_t index, uint32_t timer_id)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(profilerBuffer);

        //TODO(MO): Clean up magic numbers
        buffer[index] = 0x80000000 | ((timer_id & 0x7FFFF) << 12) | (time_H & 0xFFF);
        buffer[index+1] = time_L;
#endif //PROFILE_KERNEL
    }

    PROFILER_INLINE void mark_time(uint32_t timer_id)
    {
#if defined(PROFILE_KERNEL)
        if (wIndex < PROFILER_L1_VECTOR_SIZE)
        {
            mark_time_at_index(wIndex, timer_id);
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
        }
#endif //PROFILE_KERNEL
    }

    PROFILER_INLINE void mark_time_once(uint32_t timer_id, bool * one_time)
    {
#if defined(PROFILE_KERNEL)
        if (*one_time)
        {
            mark_time(timer_id);
        }
        *one_time = false;
#endif //PROFILE_KERNEL
    }

    PROFILER_INLINE void mark_BR_fw_first_start()
    {
#if defined(PROFILE_KERNEL) & defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

        profiler_control_buffer[FW_RESET_L] = time_L;
        profiler_control_buffer[FW_RESET_H] = time_H;
#endif //PROFILE_KERNEL
    }

    PROFILER_INLINE void risc_finished_profiling()
    {
#if defined(PROFILE_KERNEL)
        for (uint32_t i = 0; i < (wIndex % NOC_ALIGNMENT_FACTOR); i++)
        {
            mark_time(PADDING_MARKER);
        }
        profiler_control_buffer[kernel_profiler::deviceBufferEndIndex] = wIndex;
#endif //PROFILE_KERNEL
    }

    void finish_profiler()
    {
#if defined(PROFILE_KERNEL)
        risc_finished_profiling();
#if (defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_BRISC))
        uint32_t noc_x = my_x[0];
        uint32_t noc_y = my_y[0];
        uint16_t core_flat_id = noc_xy_to_profiler_flat_id[noc_x][noc_y];

        profiler_control_buffer[NOC_X] = noc_x;
        profiler_control_buffer[NOC_Y] = noc_y;
        profiler_control_buffer[FLAT_ID] = core_flat_id;

        uint32_t dram_profiler_address = profiler_control_buffer[DRAM_PROFILER_ADDRESS];

        uint32_t pageSize =
            PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * PROFILER_RISC_COUNT * profiler_core_count_per_dram;


#if defined(COMPILE_FOR_ERISC)
        int hostIndex = HOST_BUFFER_END_INDEX_ER;
        int deviceIndex = DEVICE_BUFFER_END_INDEX_ER;
        uint32_t currEndIndex =
            profiler_control_buffer[deviceIndex] +
            profiler_control_buffer[hostIndex];

        uint32_t dram_offset =
            (core_flat_id % profiler_core_count_per_dram) * PROFILER_RISC_COUNT * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
            profiler_control_buffer[hostIndex] * sizeof(uint32_t);

        const InterleavedAddrGen<true> s = {
            .bank_base_address = dram_profiler_address,
            .page_size = pageSize
        };

        if ( currEndIndex < PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC)
        {
            uint64_t dram_bank_dst_noc_addr = s.get_noc_addr(core_flat_id / profiler_core_count_per_dram, dram_offset);

            noc_async_write(
                    eth_l1_mem::address_map::PROFILER_L1_BUFFER_ER,
                    dram_bank_dst_noc_addr,
                    profiler_control_buffer[deviceIndex] * sizeof(uint32_t));

            profiler_control_buffer[hostIndex] = currEndIndex;
        }
        else
        {
            profiler_control_buffer[hostIndex] = PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC+1;
        }
        eth_noc_async_write_barrier();
#endif
#if defined(COMPILE_FOR_BRISC)
        int hostIndex;
        int deviceIndex;
        for (hostIndex = kernel_profiler::HOST_BUFFER_END_INDEX_BR, deviceIndex = kernel_profiler::DEVICE_BUFFER_END_INDEX_BR;
                (hostIndex <= kernel_profiler::HOST_BUFFER_END_INDEX_T2) && (deviceIndex <= kernel_profiler::DEVICE_BUFFER_END_INDEX_T2);
                hostIndex++, deviceIndex++)
        {
            uint32_t currEndIndex =
                profiler_control_buffer[deviceIndex] +
                profiler_control_buffer[hostIndex];

            uint32_t dram_offset =
                (core_flat_id % profiler_core_count_per_dram) * PROFILER_RISC_COUNT * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                hostIndex * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                profiler_control_buffer[hostIndex] * sizeof(uint32_t);

            const InterleavedAddrGen<true> s = {
                .bank_base_address = dram_profiler_address,
                .page_size = pageSize
            };

            if ( currEndIndex <= PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC)
            {
                uint64_t dram_bank_dst_noc_addr = s.get_noc_addr(core_flat_id / profiler_core_count_per_dram, dram_offset);

                noc_async_write(
                        PROFILER_L1_BUFFER_BR + hostIndex * PROFILER_L1_BUFFER_SIZE,
                        dram_bank_dst_noc_addr,
                        profiler_control_buffer[deviceIndex] * sizeof(uint32_t));

                profiler_control_buffer[hostIndex] = currEndIndex;
            }
            else
            {
                profiler_control_buffer[hostIndex] = PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC+1;
            }
        }
        noc_async_write_barrier();
#endif
        profiler_control_buffer[RUN_COUNTER] ++;
#endif
#endif //PROFILE_KERNEL
    }

    PROFILER_INLINE void flush_profiler()
    {
        stackSize = 0;
    }

    class profileScope
    {
        private:
            uint32_t timer_id;
            bool start_marked;
        public:
            PROFILER_INLINE profileScope (uint32_t timer_id_arg) : timer_id(timer_id_arg)
            {
#if defined(PROFILE_KERNEL)
                if (!stackSize)
                {
                    init_profiler();
                }
                if (wIndex < (PROFILER_L1_VECTOR_SIZE - stackSize))
                {
                    mark_time(timer_id);
                    stackSize += PROFILER_L1_MARKER_UINT32_SIZE;
                    start_marked = true;
                }
#endif
            }

            PROFILER_INLINE ~profileScope ()
            {
#if defined(PROFILE_KERNEL)
                if (start_marked)
                {
                    stackSize -= PROFILER_L1_MARKER_UINT32_SIZE;
                    mark_time((timer_id & 0xFFFF) | (1<<16));
                    start_marked = false;
                    if (!stackSize)
                    {
                        finish_profiler();
                    }
                }
#endif
            }
    };
}

constexpr uint32_t Hash32_CT( const char * str, size_t n, uint32_t basis = UINT32_C( 2166136261 ) ) {
    return n == 0 ? basis : Hash32_CT( str + 1, n - 1, ( basis ^ str[ 0 ] ) * UINT32_C( 16777619 ) );
}

template< size_t N >
constexpr uint32_t Hash16_CT( const char ( &s )[ N ] ) {
    auto res = Hash32_CT( s, N - 1 );
    return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
}

#define DO_PRAGMA(x) _Pragma (#x)

#define Stringize( L )     #L
#define MakeString( M, L ) M(L)
#define $Line MakeString( Stringize, __LINE__ )

#define PROFILER_MSG __FILE__ "," $Line ",KERNEL_PROFILER"
#define PROFILER_MSG_NAME( name )  name "," PROFILER_MSG

#ifdef PROFILE_KERNEL

#define DeviceZoneScoped DO_PRAGMA(message(PROFILER_MSG_NAME("no-name"))); kernel_profiler::profileScope zone = kernel_profiler::profileScope(Hash16_CT(PROFILER_MSG_NAME("no-name")));

#define DeviceZoneScopedN( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); kernel_profiler::profileScope zone = kernel_profiler::profileScope(Hash16_CT(PROFILER_MSG_NAME(name)));

#define DeviceProfilerFlush kernel_profiler::flush_profiler();

#else

#define DeviceZoneScoped

#define DeviceZoneScopedN( name )

#define DeviceProfilerFlush

#endif
