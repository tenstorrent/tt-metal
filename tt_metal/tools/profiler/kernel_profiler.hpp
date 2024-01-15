// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <climits>

#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
#include "risc_common.h"
#include "dataflow_api.h"
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

#if defined(COMPILE_FOR_BRISC)
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_BR;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_BR;
    uint16_t runCounter = 0;
#elif defined(COMPILE_FOR_NCRISC)
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_NC;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_NC;
#elif COMPILE_FOR_TRISC == 0
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T0;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T0;
#elif COMPILE_FOR_TRISC == 1
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T1;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T1;
#elif COMPILE_FOR_TRISC == 2
    uint32_t profilerBuffer = PROFILER_L1_BUFFER_T2;
    uint32_t deviceBufferEndIndex = DEVICE_BUFFER_END_INDEX_T2;
#endif

    inline __attribute__((always_inline)) void init_profiler(uint16_t briscKernelID = 0, uint16_t ncriscKernelID = 0, uint16_t triscsKernelID = 0)
    {
#if defined(PROFILE_KERNEL)
        volatile tt_l1_ptr uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
        profiler_control_buffer[deviceBufferEndIndex] = 0;
        wIndex = CUSTOM_MARKERS;

#if defined(COMPILE_FOR_BRISC)
        volatile tt_l1_ptr uint32_t *briscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_BR);
        volatile tt_l1_ptr uint32_t *ncriscBuffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_NC);
        volatile tt_l1_ptr uint32_t *trisc0Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T0);
        volatile tt_l1_ptr uint32_t *trisc1Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T1);
        volatile tt_l1_ptr uint32_t *trisc2Buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PROFILER_L1_BUFFER_T2);

        for (int i = ID_HH; i < FW_START; i ++)
        {
            briscBuffer[i] = 0;
            ncriscBuffer[i] = 0;
            trisc0Buffer[i] = 0;
            trisc1Buffer[i] = 0;
            trisc2Buffer[i] = 0;
        }

        for (int i = FW_START; i < CUSTOM_MARKERS; i ++)
        {
        //TODO(MO): Clean up magic numbers
            briscBuffer[i] = 0x80000000;
            ncriscBuffer[i] = 0x80000000;
            trisc0Buffer[i] = 0x80000000;
            trisc1Buffer[i] = 0x80000000;
            trisc2Buffer[i] = 0x80000000;
        }

        const uint32_t NOC_ID_MASK = (1 << NOC_ADDR_NODE_ID_BITS) - 1;
        uint32_t noc_id = noc_local_node_id() & 0xFFF;
        uint32_t noc_x = noc_id & NOC_ID_MASK;
        uint32_t noc_y = (noc_id >> NOC_ADDR_NODE_ID_BITS) & NOC_ID_MASK;

        uint16_t core_flat_id = get_flat_id(noc_x, noc_y);

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
        //briscBuffer[ID_LH] =  ((core_flat_id & 0xFF) << 24) | ((deviceBufferEndIndex & 0xFF) << 16) | briscKernelID;
        //ncriscBuffer[ID_LH] = ((core_flat_id & 0xFF) << 24) | ((deviceBufferEndIndex & 0xFF) << 16) | ncriscKernelID;
        //trisc0Buffer[ID_LH] = ((core_flat_id & 0xFF) << 24) | ((deviceBufferEndIndex & 0xFF) << 16) | triscsKernelID;
        //trisc1Buffer[ID_LH] = ((core_flat_id & 0xFF) << 24) | ((deviceBufferEndIndex & 0xFF) << 16) | triscsKernelID;
        //trisc2Buffer[ID_LH] = ((core_flat_id & 0xFF) << 24) | ((deviceBufferEndIndex & 0xFF) << 16) | triscsKernelID;

#endif //BRISC_INIT

#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time(uint32_t timer_id)
    {
#if defined(PROFILE_KERNEL)
        //TODO(MO): Add drop counter to control register
        if (wIndex < PROFILER_L1_VECTOR_SIZE)
        {
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
            uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
            uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif
            volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(profilerBuffer);
            uint32_t index = wIndex;

            //TODO(MO): Clean up magic numbers
            buffer[index] = 0x80000000 | ((buffer[ID_LH] & 0x7FF) << 20) | ((timer_id & 0xFF) << 12) | (time_H & 0xFFF);
            buffer[index+1] = time_L;
            wIndex += PROFILER_L1_MARKER_UINT32_SIZE;
        }
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_time_guaranteed_event(uint32_t index)
    {
#if defined(PROFILE_KERNEL)
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
        uint32_t time_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif
        volatile tt_l1_ptr uint32_t *buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(profilerBuffer);

        //TODO(MO): Clean up magic numbers
        buffer[index] = 0x80000000 | ((buffer[ID_LH] & 0x7FF) << 20) | ((((index - FW_START + 2) >> 1) & 0xFF) << 12) | (time_H & 0xFFF);
        buffer[index+1] = time_L;
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

    inline __attribute__((always_inline)) void mark_BR_fw_first_start()
    {
#if defined(PROFILE_KERNEL) & defined(COMPILE_FOR_BRISC)
        uint32_t time_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t time_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);

        volatile uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);

        profiler_control_buffer[FW_RESET_L] = time_L;
        profiler_control_buffer[FW_RESET_H] = time_H;
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_fw_start()
    {
#if defined(PROFILE_KERNEL)
        mark_time_guaranteed_event(FW_START);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_fw_end()
    {
#if defined(PROFILE_KERNEL)
        mark_time_guaranteed_event(FW_END);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_kernel_start()
    {
#if defined(PROFILE_KERNEL)
        mark_time_guaranteed_event(KERNEL_START);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void mark_kernel_end()
    {
#if defined(PROFILE_KERNEL)
        mark_time_guaranteed_event(KERNEL_END);
#endif //PROFILE_KERNEL
    }

    inline __attribute__((always_inline)) void finish()
    {
#if defined(PROFILE_KERNEL)
        for (uint32_t i = 0; i < (wIndex % NOC_ALIGNMENT_FACTOR); i++)
        {
            mark_time(PADDING_MARKER);
        }
        volatile uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);
        profiler_control_buffer[kernel_profiler::deviceBufferEndIndex] = wIndex;
#endif //PROFILE_KERNEL
    }
    inline __attribute__((always_inline)) void send_profiler_data_to_dram()
    {
#if defined(PROFILE_KERNEL) && defined(COMPILE_FOR_BRISC)
        volatile uint32_t *profiler_control_buffer = reinterpret_cast<uint32_t*>(PROFILER_L1_BUFFER_CONTROL);

        const uint32_t NOC_ID_MASK = (1 << NOC_ADDR_NODE_ID_BITS) - 1;
        uint32_t noc_id = noc_local_node_id() & 0xFFF;
        uint32_t noc_x = noc_id & NOC_ID_MASK;
        uint32_t noc_y = (noc_id >> NOC_ADDR_NODE_ID_BITS) & NOC_ID_MASK;

        uint16_t core_flat_id = get_flat_id(noc_x, noc_y);
        uint32_t dram_profiler_address = profiler_control_buffer[DRAM_PROFILER_ADDRESS];

        //TODO(MO): WORMHOLE SUPPORT :Hardcoded for GS need to make it universal and no magic numbers
        uint32_t dram_noc_x = (core_flat_id / 30) * 3 + 1;
        uint32_t dram_noc_y = noc_y > 6 ? 6 : 0;

        finish();
        int hostIndex;
        int deviceIndex;
        for (hostIndex = kernel_profiler::HOST_BUFFER_END_INDEX_BR, deviceIndex = kernel_profiler::DEVICE_BUFFER_END_INDEX_BR;
                (hostIndex <= kernel_profiler::HOST_BUFFER_END_INDEX_T2) && (deviceIndex <= kernel_profiler::DEVICE_BUFFER_END_INDEX_T2);
                hostIndex++, deviceIndex++)
        {
            uint32_t currEndIndex =
                profiler_control_buffer[deviceIndex] +
                profiler_control_buffer[hostIndex];

            uint32_t dram_address =
                dram_profiler_address +
                //TODO(MO): WORMHOLE SUPPORT : 15 is only for GS
                (core_flat_id % 15) * PROFILER_RISC_COUNT * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                hostIndex * PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC +
                profiler_control_buffer[hostIndex] * sizeof(uint32_t);

            if ( currEndIndex < PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC)
            {
                uint64_t dram_bank_dst_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, dram_address);
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
        runCounter ++;
#endif //PROFILE_KERNEL
    }
}
