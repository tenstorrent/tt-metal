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

#include "hostdevcommon/profiler_common.h"
#include "risc_attribs.h"

#ifdef PROFILER_KERNEL_FORCE_INLINE
#define PROFILER_INLINE inline __attribute__((always_inline))
#else
#define PROFILER_INLINE __attribute__((noinline))
#endif

#define DO_PRAGMA(x) _Pragma (#x)

#define Stringize( L )     #L
#define MakeString( M, L ) M(L)
#define $Line MakeString( Stringize, __LINE__ )

#define PROFILER_MSG __FILE__ "," $Line ",KERNEL_PROFILER"
#define PROFILER_MSG_NAME( name )  name "," PROFILER_MSG

#ifdef PROFILE_KERNEL
namespace kernel_profiler{

    extern uint32_t wIndex;
    extern uint32_t stackSize;

    extern uint32_t sums[SUM_COUNT];
    extern uint32_t sumIDs[SUM_COUNT];

#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t profilerBuffer = PROFILER_L1_BUFFER_BR;
    constexpr uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_BR;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
    uint16_t core_flat_id;
#elif defined(COMPILE_FOR_ERISC)
    constexpr uint32_t profilerBuffer = eth_l1_mem::address_map::PROFILER_L1_BUFFER_ER;
    constexpr uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_ER;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eth_l1_mem::address_map::PROFILER_L1_BUFFER_CONTROL);
    uint16_t core_flat_id;
#elif defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t profilerBuffer = PROFILER_L1_BUFFER_NC;
    constexpr uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_NC;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#elif COMPILE_FOR_TRISC == 0
    constexpr uint32_t profilerBuffer = PROFILER_L1_BUFFER_T0;
    constexpr uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T0;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#elif COMPILE_FOR_TRISC == 1
    constexpr uint32_t profilerBuffer = PROFILER_L1_BUFFER_T1;
    constexpr uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T1;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#elif COMPILE_FOR_TRISC == 2
    constexpr uint32_t profilerBuffer = PROFILER_L1_BUFFER_T2;
    constexpr uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T2;
    volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
#endif

    inline __attribute__((always_inline)) void init_profiler(uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0)
    {
        wIndex = CUSTOM_MARKERS;
        stackSize = 0;

        for (int i = 0; i < SUM_COUNT; i ++)
        {
            sumIDs[i] = 0;
            sums[i] = 0;
        }

#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_BRISC)
        uint32_t runCounter = profiler_control_buffer[RUN_COUNTER];

#if defined(COMPILE_FOR_ERISC)
        volatile tt_l1_ptr uint32_t *eriscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eth_l1_mem::address_map::PROFILER_L1_BUFFER_ER);

        if (runCounter == 0)
        {
            core_flat_id = noc_xy_to_profiler_flat_id[my_x[0]][my_y[0]];

#pragma GCC unroll 65534
            for (int i = ID_HH; i < GUARANTEED_MARKER_1_H; i ++)
            {
                eriscBuffer[i] = 0;
            }

            eriscBuffer [ID_LH] = ((core_flat_id & 0xFF) << 3) | 0;

            profiler_control_buffer[NOC_X] = my_x[0];
            profiler_control_buffer[NOC_Y] = my_y[0];
            profiler_control_buffer[FLAT_ID] = core_flat_id;
        }

#pragma GCC unroll 65534
        for (int i = GUARANTEED_MARKER_1_H; i < CUSTOM_MARKERS; i ++)
        {
        //TODO(MO): Clean up magic numbers
            eriscBuffer[i] = 0x80000000;
        }

        eriscBuffer [ID_LL] = runCounter;

#endif //ERISC_INIT
#if defined(COMPILE_FOR_BRISC)
        volatile tt_l1_ptr uint32_t *briscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_BR);
        volatile tt_l1_ptr uint32_t *ncriscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_NC);
        volatile tt_l1_ptr uint32_t *trisc0Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T0);
        volatile tt_l1_ptr uint32_t *trisc1Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T1);
        volatile tt_l1_ptr uint32_t *trisc2Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T2);

        if (runCounter == 0)
        {
            core_flat_id = noc_xy_to_profiler_flat_id[my_x[0]][my_y[0]];

#pragma GCC unroll 65534
            for (int i = ID_HH; i < GUARANTEED_MARKER_1_H; i ++)
            {
                briscBuffer[i] = 0;
                ncriscBuffer[i] = 0;
                trisc0Buffer[i] = 0;
                trisc1Buffer[i] = 0;
                trisc2Buffer[i] = 0;
            }

            briscBuffer [ID_LH] = ((core_flat_id & 0xFF) << 3) | 0;
            ncriscBuffer[ID_LH] = ((core_flat_id & 0xFF) << 3) | 1;
            trisc0Buffer[ID_LH] = ((core_flat_id & 0xFF) << 3) | 2;
            trisc1Buffer[ID_LH] = ((core_flat_id & 0xFF) << 3) | 3;
            trisc2Buffer[ID_LH] = ((core_flat_id & 0xFF) << 3) | 4;

            profiler_control_buffer[NOC_X] = my_x[0];
            profiler_control_buffer[NOC_Y] = my_y[0];
            profiler_control_buffer[FLAT_ID] = core_flat_id;
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

#endif //BRISC_INIT
#endif
    }

    constexpr uint32_t get_end_timer_id (uint32_t timer_id)
    {
        return ((timer_id & 0xFFFF) | ((1<<16) & 0x7FFFF));
    }

    uint32_t get_sum_id (uint32_t sum_id)
    {
        return ((sum_id & 0xFFFF) | ((1<<17) & 0x7FFFF));
    }

    inline __attribute__((always_inline)) void mark_time_at_index_inlined(uint32_t index, uint32_t timer_id)
    {
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_profiler::profilerBuffer);
        volatile tt_reg_ptr uint32_t *p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t *> (RISCV_DEBUG_REG_WALL_CLOCK_L);
        buffer[index] = 0x80000000 | ((timer_id & 0x7FFFF) << 12) | (p_reg[1] & 0xFFF);
        buffer[index+1] = p_reg[0];
    }

    PROFILER_INLINE void mark_padding()
    {
        if (wIndex < PROFILER_L1_VECTOR_SIZE)
        {
            volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_profiler::profilerBuffer);
            buffer[wIndex] = 0x80000000;
            buffer[wIndex+1] = 0;
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }

    PROFILER_INLINE void mark_BR_fw_first_start()
    {
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

        profiler_control_buffer[FW_RESET_L] = time_L;
        profiler_control_buffer[FW_RESET_H] = time_H;
    }

    inline __attribute__((always_inline)) void risc_finished_profiling()
    {
        for (int i = 0; i < SUM_COUNT; i ++)
        {
            if (sums[i] > 0)
            {
                if (wIndex < PROFILER_L1_VECTOR_SIZE)
                {
                    volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_profiler::profilerBuffer);
                    buffer[wIndex] = 0x80000000 | ((get_sum_id(sumIDs[i]) & 0x7FFFF) << 12);
                    buffer[wIndex + 1] = sums[i];
                    wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
                }
            }
        }

        for (uint32_t i = 0; i < (wIndex % NOC_ALIGNMENT_FACTOR); i++)
        {
            mark_padding();
        }
        profiler_control_buffer[kernel_profiler::deviceBufferEndIndex] = wIndex;

    }

    inline __attribute__((always_inline)) void finish_profiler()
    {
        risc_finished_profiling();
#if (defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_BRISC))

        uint32_t pageSize =
            PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * PROFILER_RISC_COUNT * profiler_core_count_per_dram;

        uint32_t dram_profiler_address = profiler_control_buffer[DRAM_PROFILER_ADDRESS];

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
#endif
#if defined(COMPILE_FOR_BRISC)
        int hostIndex;
        int deviceIndex;
        for (hostIndex = kernel_profiler::HOST_BUFFER_END_INDEX_BR, deviceIndex = kernel_profiler::DEVICE_BUFFER_END_INDEX_BR;
                (hostIndex <= kernel_profiler::HOST_BUFFER_END_INDEX_T2) && (deviceIndex <= kernel_profiler::DEVICE_BUFFER_END_INDEX_T2);
                hostIndex++, deviceIndex++)
        {
            if (profiler_control_buffer[deviceIndex])
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

                profiler_control_buffer[deviceIndex] = 0;
            }
        }
#endif
        noc_async_write_barrier();
        profiler_control_buffer[RUN_COUNTER] ++;
#endif
    }

    constexpr uint32_t Hash32_CT( const char * str, size_t n, uint32_t basis = UINT32_C( 2166136261 ) ) {
        return n == 0 ? basis : Hash32_CT( str + 1, n - 1, ( basis ^ str[ 0 ] ) * UINT32_C( 16777619 ) );
    }

    template< size_t N >
    constexpr uint32_t Hash16_CT( const char ( &s )[ N ] ) {
        auto res = Hash32_CT( s, N - 1 );
        return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
    }

    template<uint32_t timer_id>
    struct profileScope
    {
        bool start_marked = false;
        PROFILER_INLINE profileScope ()
        {
            if (wIndex < (PROFILER_L1_VECTOR_SIZE - stackSize))
            {
                stackSize += PROFILER_L1_MARKER_UINT32_SIZE;
                start_marked = true;
                mark_time_at_index_inlined(wIndex, timer_id);
                wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
            }
        }

        PROFILER_INLINE ~profileScope ()
        {
            if (start_marked)
            {
                mark_time_at_index_inlined(wIndex, get_end_timer_id(timer_id));
                wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
                start_marked = false;
                stackSize -= PROFILER_L1_MARKER_UINT32_SIZE;
            }
        }
    };

    template<uint32_t timer_id, uint32_t index>
    struct profileScopeGuaranteed
    {
        static constexpr uint32_t start_index = (2 * index * PROFILER_L1_MARKER_UINT32_SIZE) + GUARANTEED_MARKER_1_H;
        static constexpr uint32_t  end_index = (2 * index * PROFILER_L1_MARKER_UINT32_SIZE) + GUARANTEED_MARKER_2_H;

        static_assert (start_index < CUSTOM_MARKERS);
        static_assert (end_index < CUSTOM_MARKERS);

        inline __attribute__((always_inline)) profileScopeGuaranteed ()
        {
            if constexpr  (index == 0)
            {
                init_profiler();
            }
            mark_time_at_index_inlined(start_index, timer_id);
        }
        inline __attribute__((always_inline))  ~profileScopeGuaranteed ()
        {
            mark_time_at_index_inlined(end_index, get_end_timer_id(timer_id));
            if constexpr  (index == 0)
            {
                finish_profiler();
            }
        }
    };

    template<uint32_t timer_id, uint32_t index>
    struct profileScopeAccumulate
    {
        uint64_t start_time = 0;
        volatile tt_reg_ptr uint32_t *p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t *> (RISCV_DEBUG_REG_WALL_CLOCK_L);

        inline __attribute__((always_inline)) profileScopeAccumulate ()
        {
            start_time = ((uint64_t)p_reg[1] << 32) | p_reg[0];
        }
        inline __attribute__((always_inline))  ~profileScopeAccumulate ()
        {
            sumIDs[index] = timer_id;
            sums[index] += (((uint64_t)p_reg[1] << 32) | p_reg[0]) - start_time;
        }
    };
}


#define DeviceZoneScopedN( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

#define DeviceZoneScopedMainN( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScopeGuaranteed<hash, 0> zone = kernel_profiler::profileScopeGuaranteed<hash, 0>();

#define DeviceZoneScopedMainChildN( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));kernel_profiler::profileScopeGuaranteed<hash, 1> zone = kernel_profiler::profileScopeGuaranteed<hash, 1>();

#define DeviceZoneScopedSumN1( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScopeAccumulate<hash, 0> zone = kernel_profiler::profileScopeAccumulate<hash, 0>();

#define DeviceZoneScopedSumN2( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScopeAccumulate<hash, 1> zone = kernel_profiler::profileScopeAccumulate<hash, 1>();

#else

#define DeviceZoneScopedMainN( name )

#define DeviceZoneScopedMainChildN( name )

#define DeviceZoneScopedN( name )

#define DeviceZoneScopedSumN1( name )

#define DeviceZoneScopedSumN2( name )

#endif
