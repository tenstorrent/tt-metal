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

constexpr std::uint32_t COMPRESS_ELEM_CNT = 32;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // compress source: COMPRESS_ELEM_CNT contiguous elements
    const Operand& buffer_B         = params.buffer_B;   // compress mask: bit w (LSB-first) keeps source element w
    const Operand& buffer_Res       = params.buffer_Res; // compress result: COMPRESS_ELEM_CNT contiguous elements
#endif
    volatile const std::uint16_t* src;
    volatile const std::uint8_t* mask;
    volatile std::uint16_t* dst;

    {
        ZONE_SCOPED("INIT")
        src  = reinterpret_cast<volatile const std::uint16_t*>(buffer_A[0]);
        mask = reinterpret_cast<volatile const std::uint8_t*>(buffer_B[0]);
        dst  = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            std::uint32_t out = 0;
            for (std::uint32_t w = 0; w < COMPRESS_ELEM_CNT; w++)
            {
                if ((mask[w >> 3] >> (w & 7)) & 1)
                {
                    dst[out++] = src[w];
                }
            }
            // Positions past the selected count read as 0, matching the RVV variant.
            for (; out < COMPRESS_ELEM_CNT; out++)
            {
                dst[out] = 0;
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
