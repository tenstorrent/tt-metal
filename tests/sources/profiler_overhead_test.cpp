// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Globals
uint32_t unp_cfg_context        = 0;
uint32_t pack_sync_tile_dst_ptr = 0;

#ifdef LLK_TRISC_UNPACK

#include "profiler.h"

void run_kernel()
{
    // measure length of zones of different sizes

    // start with i = 8 because for i < 8, overhead is not consistent
    for (uint32_t i = 8; i < 40; i++)
    {
        uint32_t cnt = i;
        {
            // The body of the loop without the zone should take i*10 cycles
            // however the ZONE_SCOPED macro will add around 36 cycles of overhead
            // so the total time should be around i*10 + 36 cycles
            ZONE_SCOPED("OVERHEAD");
        loop:
            asm volatile("addi %0, %1, -1" : "=r"(cnt) : "r"(cnt));
            asm volatile("nop");
            asm volatile("nop");
            asm volatile("nop");
            asm volatile("nop");
            asm volatile("nop");
            asm volatile("nop");
            asm volatile("nop");
            asm volatile("nop");
            asm volatile goto("bgtu %0, zero, %l1" : : "r"(cnt) : : loop);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel()
{
    // Only unpack kernel is measuring profiler overhead
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel()
{
    // Only unpack kernel is measuring profiler overhead
}

#endif
