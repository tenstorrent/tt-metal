// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "sfpu_stub.h"

constexpr std::uint32_t GATHER_ELEM_CNT = 32;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // gather source: GATHER_ELEM_CNT contiguous elements
    const Operand& buffer_B         = params.buffer_B;   // gather indices: GATHER_ELEM_CNT bytes
    const Operand& buffer_Res       = params.buffer_Res; // gather result: GATHER_ELEM_CNT contiguous elements
#endif
    volatile const std::uint16_t* src;
    volatile const std::uint8_t* idx;
    volatile std::uint16_t* dst;

    {
        ZONE_SCOPED("INIT")
        src = reinterpret_cast<volatile const std::uint16_t*>(buffer_A[0]);
        idx = reinterpret_cast<volatile const std::uint8_t*>(buffer_B[0]);
        dst = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t w = 0; w < GATHER_ELEM_CNT; w++)
            {
                dst[w] = src[idx[w]];
            }
        }
        asm volatile("fence" ::: "memory");
    }
}

#endif

#if defined(LLK_TRISC_MATH) || defined(LLK_TRISC_PACK)

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params; // TRISC0-only kernel
}

#endif
