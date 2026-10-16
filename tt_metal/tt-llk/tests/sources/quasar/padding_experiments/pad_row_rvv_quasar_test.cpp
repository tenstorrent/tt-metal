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

static_assert(FACE_COLS == 16, "e16/m2 below assumes one face row fits a 2-register group at VLEN=128");

template <std::uint32_t FACE>
inline void rvv_pad_face_row(std::uint16_t* dst)
{
    if constexpr (face_padded(FACE))
    {
        asm volatile(
            "vsetvli zero, %[n], e16, m2, ta, ma\n" // vl = segment length; v8 keeps the splat from rvv_splat_scalar
            "vse16.v v8, (%[seg])\n"
            :
            : [n] "r"(seg_len(FACE)), [seg] "r"(dst + seg_elem_offset(FACE))
            : "memory");
    }
}

inline void rvv_splat_scalar(const std::uint32_t scalar)
{
    asm volatile(
        "vsetvli zero, %[n], e16, m2, ta, ma\n" // splat a full face row so any segment length can be stored
        "vmv.v.x v8, %[scalar]\n"               // e16 takes the low 16 bits of the scalar
        :
        : [n] "r"(FACE_COLS), [scalar] "r"(scalar)
        : "v8", "v9");
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // scalar: low 16 bits of the first u32
    const Operand& buffer_Res       = params.buffer_Res; // pad result: 32x32 16-bit tile, only the row-0 section is written
#endif
    std::uint32_t scalar;
    std::uint16_t* dst;

    {
        ZONE_SCOPED("INIT")
        scalar = *reinterpret_cast<volatile const std::uint32_t*>(buffer_A[0]);
        dst    = reinterpret_cast<std::uint16_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            rvv_splat_scalar(scalar);
            rvv_pad_face_row<0>(dst);
            rvv_pad_face_row<1>(dst);
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
