// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <climits>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC)
#include "risc_common.h"
#include "dataflow_api.h"
#else
#include "ckernel.h"
#endif

#include "hostdevcommon/profiler_common.h"
#include "risc_attribs.h"

#include "dev_msgs.h"

#define DO_PRAGMA(x) _Pragma (#x)

#define Stringize( L )     #L
#define MakeString( M, L ) M(L)
#define $Line MakeString( Stringize, __LINE__ )

#define PROFILER_MSG __FILE__ "," $Line ",KERNEL_PROFILER"
#define PROFILER_MSG_NAME( name )  name "," PROFILER_MSG

#define SrcLocNameToHash( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME( name ));

#if  defined(PROFILE_KERNEL) && ( !defined(DISPATCH_KERNEL) || (defined(DISPATCH_KERNEL) && defined(COMPILE_FOR_NCRISC) && (PROFILE_KERNEL == PROFILER_OPT_DO_DISPATCH_CORES)))
namespace kernel_profiler{

    extern uint32_t wIndex;
    extern uint32_t stackSize;

    extern uint32_t sums[SUM_COUNT];
    extern uint32_t sumIDs[SUM_COUNT];

    constexpr uint32_t QUICK_PUSH_MARKER_COUNT = 2;
    constexpr int WALL_CLOCK_HIGH_INDEX = 1;
    constexpr int WALL_CLOCK_LOW_INDEX = 0;

    volatile tt_l1_ptr uint32_t *profiler_control_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));

    volatile tt_l1_ptr uint32_t (*profiler_data_buffer)[kernel_profiler::PROFILER_L1_VECTOR_SIZE] =
        reinterpret_cast<volatile tt_l1_ptr uint32_t (*)[kernel_profiler::PROFILER_L1_VECTOR_SIZE]>(GET_MAILBOX_ADDRESS_DEV(profiler.buffer));

#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t myRiscID = 0;
#elif defined(COMPILE_FOR_ERISC)
    constexpr uint32_t myRiscID = 0;
#elif defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t myRiscID = 1;
#elif COMPILE_FOR_TRISC == 0
    constexpr uint32_t myRiscID = 2;
#elif COMPILE_FOR_TRISC == 1
    constexpr uint32_t myRiscID = 3;
#elif COMPILE_FOR_TRISC == 2
    constexpr uint32_t myRiscID = 4;
#endif

    constexpr uint32_t Hash32_CT( const char * str, size_t n, uint32_t basis = UINT32_C( 2166136261 ) ) {
        return n == 0 ? basis : Hash32_CT( str + 1, n - 1, ( basis ^ str[ 0 ] ) * UINT32_C( 16777619 ) );
    }

    template< size_t N >
    constexpr uint32_t Hash16_CT( const char ( &s )[ N ] ) {
        auto res = Hash32_CT( s, N - 1 );
        return ((res & 0xFFFF) ^ ((res & 0xFFFF0000) >> 16)) & 0xFFFF;
    }

    __attribute__((noinline)) void init_profiler(uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0)
    {
        wIndex = CUSTOM_MARKERS;
        stackSize = 0;

        for (int i = 0; i < SUM_COUNT; i ++)
        {
            sumIDs[i] = 0;
            sums[i] = 0;
        }

#if defined(COMPILE_FOR_ERISC) ||  defined(COMPILE_FOR_BRISC)
        uint32_t runCounter = profiler_control_buffer[RUN_COUNTER];
        profiler_control_buffer[PROFILER_DONE] = 0;

        if (runCounter == 0)
        {
            for (uint32_t riscID = 0; riscID < PROFILER_RISC_COUNT; riscID ++)
            {
                for (uint32_t i = ID_HH; i < GUARANTEED_MARKER_1_H; i ++)
                {
                    profiler_data_buffer[riscID][i] = 0;
                }
            }

            profiler_control_buffer[NOC_X] = my_x[0];
            profiler_control_buffer[NOC_Y] = my_y[0];
        }

        for (uint32_t riscID = 0; riscID < PROFILER_RISC_COUNT; riscID ++)
        {
            for (uint32_t i = GUARANTEED_MARKER_1_H; i < CUSTOM_MARKERS; i ++)
            {
                //TODO(MO): Clean up magic numbers
                profiler_data_buffer[riscID][i] = 0x80000000;
            }
            profiler_data_buffer[riscID][ID_LL] =  (runCounter & 0xFFFF) | (profiler_data_buffer[riscID][ID_LL] & 0xFFFF0000);
        }
#endif
    }

    constexpr uint32_t get_end_timer_id (uint32_t timer_id)
    {
        return ((timer_id & 0xFFFF) | ((1<<16) & 0x7FFFF));
    }

    inline __attribute__((always_inline)) uint32_t get_sum_id (uint32_t sum_id)
    {
        return ((sum_id & 0xFFFF) | ((1<<17) & 0x7FFFF));
    }

    inline __attribute__((always_inline)) void mark_time_at_index_inlined(uint32_t index, uint32_t timer_id)
    {
        volatile tt_reg_ptr uint32_t *p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t *> (RISCV_DEBUG_REG_WALL_CLOCK_L);
        profiler_data_buffer[myRiscID][index] = 0x80000000 | ((timer_id & 0x7FFFF) << 12) | (p_reg[WALL_CLOCK_HIGH_INDEX] & 0xFFF);
        profiler_data_buffer[myRiscID][index+1] = p_reg[WALL_CLOCK_LOW_INDEX];
    }

    inline __attribute__((always_inline)) void mark_padding()
    {
        if (wIndex < PROFILER_L1_VECTOR_SIZE)
        {
            profiler_data_buffer[myRiscID][wIndex] = 0x80000000;
            profiler_data_buffer[myRiscID][wIndex+1] = 0;
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }

    inline __attribute__((always_inline)) void mark_dropped_timestamps(uint32_t index)
    {
        uint32_t curr = profiler_control_buffer[DROPPED_ZONES];
        profiler_control_buffer[DROPPED_ZONES] = (1 << index) | curr;
    }

    inline __attribute__((always_inline)) void set_host_counter(uint32_t counterValue)
    {
        for (uint32_t riscID = 0; riscID < PROFILER_RISC_COUNT; riscID ++)
        {
            profiler_data_buffer[riscID][ID_LL] = (counterValue << 16) | (profiler_data_buffer[riscID][ID_LL] & 0xFFFF);
        }
    }

    inline __attribute__((always_inline)) void set_profiler_zone_valid(bool condition) {
        profiler_control_buffer[PROFILER_DONE] = !condition;
    }

    inline __attribute__((always_inline)) void risc_finished_profiling()
    {
        for (int i = 0; i < SUM_COUNT; i ++)
        {
            if (sums[i] > 0)
            {
                if (wIndex < PROFILER_L1_VECTOR_SIZE)
                {
                    profiler_data_buffer[myRiscID][wIndex] = 0x80000000 | ((get_sum_id(sumIDs[i]) & 0x7FFFF) << 12);
                    profiler_data_buffer[myRiscID][wIndex + 1] = sums[i];
                    wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
                }
            }
        }

        for (uint32_t i = 0; i < (wIndex % NOC_ALIGNMENT_FACTOR); i++)
        {
            mark_padding();
        }
        profiler_control_buffer[kernel_profiler::DEVICE_BUFFER_END_INDEX_BR_ER + myRiscID] = wIndex;
    }

    __attribute__((noinline)) void finish_profiler()
    {
        risc_finished_profiling();
#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_BRISC)
        if (profiler_control_buffer[PROFILER_DONE] == 1){
            return;
        }
        while (!profiler_control_buffer[DRAM_PROFILER_ADDRESS]);
        uint32_t core_flat_id = profiler_control_buffer[FLAT_ID];
        uint32_t profiler_core_count_per_dram = profiler_control_buffer[CORE_COUNT_PER_DRAM];

        uint32_t pageSize =
            PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MAX_RISCV_PER_CORE * profiler_core_count_per_dram;

        for (uint32_t riscID = 0; riscID < PROFILER_RISC_COUNT; riscID ++)
	{
            profiler_data_buffer[riscID][ID_LH] = ((core_flat_id & 0xFF) << 3) | riscID;
            int hostIndex = riscID;
            int deviceIndex = kernel_profiler::DEVICE_BUFFER_END_INDEX_BR_ER + riscID;
	    if (profiler_control_buffer[deviceIndex])
	    {
		uint32_t currEndIndex =
		    profiler_control_buffer[deviceIndex] +
		    profiler_control_buffer[hostIndex];

                bool do_noc = false;
                uint32_t dram_offset = 0 ;
                uint32_t send_size = 0;
		if (currEndIndex <= PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC)
		{
		    dram_offset =
			(core_flat_id % profiler_core_count_per_dram) * MAX_RISCV_PER_CORE * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
			hostIndex * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
			profiler_control_buffer[hostIndex] * sizeof(uint32_t);

                    send_size = profiler_control_buffer[deviceIndex] * sizeof(uint32_t);

                    do_noc = true;
		    profiler_control_buffer[hostIndex] = currEndIndex;
		}
		else if (profiler_control_buffer[RUN_COUNTER] < 1)
		{
                    dram_offset =
                        (core_flat_id % profiler_core_count_per_dram) *
                        MAX_RISCV_PER_CORE * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                        hostIndex * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC;

                    send_size = CUSTOM_MARKERS * sizeof(uint32_t);

                    do_noc = true;
                    mark_dropped_timestamps(hostIndex);
		}
                else{
                    mark_dropped_timestamps(hostIndex);
                }

                if (do_noc){
		    const InterleavedAddrGen<true> s = {
			.bank_base_address = profiler_control_buffer[DRAM_PROFILER_ADDRESS],
			.page_size = pageSize
		    };

		    uint64_t dram_bank_dst_noc_addr = s.get_noc_addr(core_flat_id / profiler_core_count_per_dram, dram_offset);

                    noc_async_write(
                            reinterpret_cast<uint32_t>(profiler_data_buffer[hostIndex]),
                            dram_bank_dst_noc_addr,
                            send_size);
                }
		profiler_control_buffer[deviceIndex] = 0;
	    }
	}

        noc_async_write_barrier();
        profiler_control_buffer[RUN_COUNTER] ++;
        profiler_control_buffer[PROFILER_DONE] = 1;
#endif
    }

    inline __attribute__((always_inline)) void quick_push ()
    {
#if defined(DISPATCH_KERNEL) && defined(COMPILE_FOR_NCRISC) && (PROFILE_KERNEL == PROFILER_OPT_DO_DISPATCH_CORES)
        SrcLocNameToHash("PROFILER-NOC-QUICK-SEND");
        mark_time_at_index_inlined(wIndex, hash);
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;

        while (!profiler_control_buffer[DRAM_PROFILER_ADDRESS]);
        uint32_t core_flat_id = profiler_control_buffer[FLAT_ID];
        uint32_t profiler_core_count_per_dram = profiler_control_buffer[CORE_COUNT_PER_DRAM];

        profiler_data_buffer[myRiscID][ID_LH] = ((core_flat_id & 0xFF) << 3) | myRiscID;

        uint32_t dram_offset =
            (core_flat_id % profiler_core_count_per_dram) * MAX_RISCV_PER_CORE * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
            (HOST_BUFFER_END_INDEX_BR_ER + myRiscID) * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
            profiler_control_buffer[HOST_BUFFER_END_INDEX_BR_ER + myRiscID] * sizeof(uint32_t);

        const InterleavedAddrGen<true> s = {
            .bank_base_address = profiler_control_buffer[DRAM_PROFILER_ADDRESS],
            .page_size = PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * MAX_RISCV_PER_CORE * profiler_core_count_per_dram
        };

        uint64_t dram_bank_dst_noc_addr = s.get_noc_addr(core_flat_id / profiler_core_count_per_dram, dram_offset);

        mark_time_at_index_inlined(wIndex, get_end_timer_id(hash));
        wIndex += PROFILER_L1_MARKER_UINT32_SIZE;

        uint32_t currEndIndex = profiler_control_buffer[HOST_BUFFER_END_INDEX_BR_ER + myRiscID] + wIndex;

        if ( currEndIndex <= PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC)
        {
            noc_async_write(
                    reinterpret_cast<uint32_t>(profiler_data_buffer[myRiscID]),
                    dram_bank_dst_noc_addr,
                    wIndex * sizeof(uint32_t));

            profiler_control_buffer[HOST_BUFFER_END_INDEX_BR_ER + myRiscID] = currEndIndex;

        }
        else
        {
            mark_dropped_timestamps(HOST_BUFFER_END_INDEX_BR_ER + myRiscID);
        }

        wIndex = CUSTOM_MARKERS;

#endif
    }


    template<uint32_t timer_id, bool dispatch = false>
    struct profileScope
    {
        bool start_marked = false;
        inline __attribute__((always_inline)) profileScope ()
        {
            bool bufferHasRoom = false;
            if constexpr (dispatch)
            {
                bufferHasRoom = wIndex < (PROFILER_L1_VECTOR_SIZE - stackSize - (QUICK_PUSH_MARKER_COUNT * PROFILER_L1_MARKER_UINT32_SIZE));
            }
            else
            {
                bufferHasRoom = wIndex < (PROFILER_L1_VECTOR_SIZE - stackSize);
            }

            if (bufferHasRoom)
            {
                stackSize += PROFILER_L1_MARKER_UINT32_SIZE;
                start_marked = true;
                mark_time_at_index_inlined(wIndex, timer_id);
                wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
            }
        }

        inline __attribute__((always_inline)) ~profileScope ()
        {
            if (start_marked)
            {
                mark_time_at_index_inlined(wIndex, get_end_timer_id(timer_id));
                wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
                start_marked = false;
                stackSize -= PROFILER_L1_MARKER_UINT32_SIZE;
            }

            if constexpr (dispatch)
            {
                if (wIndex >= (PROFILER_L1_VECTOR_SIZE - (QUICK_PUSH_MARKER_COUNT * PROFILER_L1_MARKER_UINT32_SIZE)))
                {
                    quick_push();
                }
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
            start_time = ((uint64_t)p_reg[WALL_CLOCK_HIGH_INDEX] << 32) | p_reg[WALL_CLOCK_LOW_INDEX];
        }
        inline __attribute__((always_inline))  ~profileScopeAccumulate ()
        {
            sumIDs[index] = timer_id;
            sums[index] += (((uint64_t)p_reg[WALL_CLOCK_HIGH_INDEX] << 32) | p_reg[WALL_CLOCK_LOW_INDEX]) - start_time;
        }
    };
}



#define DeviceZoneScopedN( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScope<hash> zone = kernel_profiler::profileScope<hash>();

#if (defined(DISPATCH_KERNEL) && defined(COMPILE_FOR_NCRISC) && (PROFILE_KERNEL == PROFILER_OPT_DO_DISPATCH_CORES))

#define DeviceZoneScopedND( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScope<hash,true> zone = kernel_profiler::profileScope<hash,true>();

#else

#define DeviceZoneScopedND( name )

#endif

#define DeviceValidateProfiler( condition ) kernel_profiler::set_profiler_zone_valid(condition);

#define DeviceZoneScopedMainN( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScopeGuaranteed<hash, 0> zone = kernel_profiler::profileScopeGuaranteed<hash, 0>();

#define DeviceZoneScopedMainChildN( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name));kernel_profiler::profileScopeGuaranteed<hash, 1> zone = kernel_profiler::profileScopeGuaranteed<hash, 1>();

#define DeviceZoneScopedSumN1( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScopeAccumulate<hash, 0> zone = kernel_profiler::profileScopeAccumulate<hash, 0>();

#define DeviceZoneScopedSumN2( name ) DO_PRAGMA(message(PROFILER_MSG_NAME(name))); auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); kernel_profiler::profileScopeAccumulate<hash, 1> zone = kernel_profiler::profileScopeAccumulate<hash, 1>();

#define DeviceZoneSetCounter( counter ) kernel_profiler::set_host_counter(counter);

#else

#define DeviceValidateProfiler( condition )

#define DeviceZoneScopedMainN( name )

#define DeviceZoneScopedMainChildN( name )

#define DeviceZoneScopedN( name )

#define DeviceZoneScopedSumN1( name )

#define DeviceZoneScopedSumN2( name )

#define DeviceZoneScopedND( name )

#define DeviceZoneSetCounter( counter )

#endif
