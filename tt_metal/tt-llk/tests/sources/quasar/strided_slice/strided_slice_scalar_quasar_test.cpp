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

constexpr std::uint32_t ROW_ELEM_CNT   = 32;
constexpr std::uint32_t TILE_ROW_COUNT = 32;
constexpr std::uint32_t ELEMENT_SIZE   = 2; // Float16_b

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t SLICE_START = params.SLICE_START;
    const std::uint32_t SLICE_END   = params.SLICE_END;
    const std::uint32_t SLICE_STEP  = params.SLICE_STEP;
    const Operand& buffer_A         = params.buffer_A;   // slice source: row-major 32x32 tile, sliced row by row
    const Operand& buffer_Res       = params.buffer_Res; // slice result: kept columns packed at the start of each 32-wide row
#endif
    volatile const std::uint8_t* src;
    volatile std::uint8_t* dst;

    {
        ZONE_SCOPED("INIT")
        src = reinterpret_cast<volatile const std::uint8_t*>(buffer_A[0]);
        dst = reinterpret_cast<volatile std::uint8_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t row = 0; row < TILE_ROW_COUNT; row++)
            {
                volatile const std::uint8_t* in_stick = src + row * ROW_ELEM_CNT * ELEMENT_SIZE;
                volatile std::uint8_t* out_stick      = dst + row * ROW_ELEM_CNT * ELEMENT_SIZE;
                std::uint32_t out_col                 = 0;
                for (std::uint32_t l = SLICE_START; l < SLICE_END; l += SLICE_STEP)
                {
                    const std::uint32_t src_offset = l * ELEMENT_SIZE;
                    const std::uint32_t dst_offset = out_col * ELEMENT_SIZE;
                    for (std::uint32_t byte = 0; byte < ELEMENT_SIZE; byte++)
                    {
                        out_stick[dst_offset + byte] = in_stick[src_offset + byte];
                    }
                    out_col++;
                }
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
