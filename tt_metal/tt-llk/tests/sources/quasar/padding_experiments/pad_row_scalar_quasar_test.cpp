// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "pad_row_section.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

template <std::uint32_t FACE>
inline void scalar_pad_face_row(const std::uint16_t scalar, volatile std::uint16_t* dst)
{
    if constexpr (face_padded(FACE))
    {
        for (std::uint32_t j = seg_elem_offset(FACE); j < seg_elem_offset(FACE) + seg_len(FACE); ++j)
        {
            dst[j] = scalar;
        }
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // scalar: low 16 bits of the first u32
    const Operand& buffer_Res       = params.buffer_Res; // pad result: 32x32 16-bit tile, only the row-0 section is written
#endif
    std::uint16_t scalar;
    volatile std::uint16_t* dst;

    {
        ZONE_SCOPED("INIT")
        scalar = static_cast<std::uint16_t>(*reinterpret_cast<volatile const std::uint32_t*>(buffer_A[0]));
        dst    = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            scalar_pad_face_row<0>(scalar, dst);
            scalar_pad_face_row<1>(scalar, dst);
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
