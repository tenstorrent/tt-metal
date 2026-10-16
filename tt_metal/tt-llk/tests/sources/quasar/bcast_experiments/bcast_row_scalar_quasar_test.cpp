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

constexpr std::uint32_t FACE_U32_CNT     = 128; // 16x16 16-bit elements per face
constexpr std::uint32_t FACE_ROW_U32_CNT = 8;   // 16 16-bit elements per face row

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // scalar: first u32 holds the double-packed 16-bit value
    const Operand& buffer_Res       = params.buffer_Res; // bcast result: 32x32 16-bit tile, only row 0 is written
#endif
    std::uint32_t scalar;
    volatile std::uint32_t* dst;

    {
        ZONE_SCOPED("INIT")
        scalar = *reinterpret_cast<volatile const std::uint32_t*>(buffer_A[0]);
        dst    = reinterpret_cast<volatile std::uint32_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t k = 0; k < 2; ++k)
            {
                const std::uint32_t idx = k * FACE_U32_CNT;
                for (std::uint32_t j = 0; j < FACE_ROW_U32_CNT; ++j)
                {
                    dst[idx + j] = scalar;
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
