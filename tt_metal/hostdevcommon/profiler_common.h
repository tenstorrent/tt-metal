// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

inline __attribute__((always_inline)) uint32_t get_flat_id(uint32_t coreX, uint32_t coreY)
{
    constexpr uint32_t DRAM_ROW = 6;
    uint32_t coreFlatID = 0;

    if (coreY > DRAM_ROW){
        coreX --;
        coreY --;
        coreY --;
    }
    else{
        coreX --;
        coreY --;
    }
    coreFlatID = coreY * 3 + (coreX / 3) * 30 + (coreX % 3) ;
    return coreFlatID;
}

namespace kernel_profiler{

    constexpr static uint32_t PADDING_MARKER = ((1<<16) - 1);
    constexpr static uint32_t NOC_ALIGNMENT_FACTOR = 4;

    enum BufferIndex {
        ID_HH, ID_HL,
        ID_LH, ID_LL,
        FW_START, FW_START_L,
        KERNEL_START, KERNEL_START_L,
        KERNEL_END, KERNEL_END_L,
        FW_END, FW_END_L,
        CUSTOM_MARKERS};

    enum ControlBuffer
    {
        HOST_BUFFER_END_INDEX_BR,
        HOST_BUFFER_END_INDEX_NC,
        HOST_BUFFER_END_INDEX_T0,
        HOST_BUFFER_END_INDEX_T1,
        HOST_BUFFER_END_INDEX_T2,
        DEVICE_BUFFER_END_INDEX_BR,
        DEVICE_BUFFER_END_INDEX_NC,
        DEVICE_BUFFER_END_INDEX_T0,
        DEVICE_BUFFER_END_INDEX_T1,
        DEVICE_BUFFER_END_INDEX_T2,
        FW_RESET_H,
        FW_RESET_L,
        DRAM_PROFILER_ADDRESS,
        CONTROL_BUFFER_SIZE
    };



}
