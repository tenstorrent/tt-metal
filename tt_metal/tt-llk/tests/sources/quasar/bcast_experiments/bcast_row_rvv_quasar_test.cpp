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

inline void rvv_bcast_row(const std::uint32_t scalar, std::uint32_t* dst)
{
    static_assert(FACE_ROW_U32_CNT == 8, "register grouping below assumes 8 u32 per face row at VLEN=128");

    asm volatile(
        "vsetvli zero, %[n], e32, m2, ta, ma\n" // 32-bit elements, group 2 registers so one operand covers a face row
        "vmv.v.x v8, %[scalar]\n"               // splat the double-packed scalar across the row
        "vse32.v v8, (%[face0])\n"              // face 0, row 0
        "vse32.v v8, (%[face1])\n"              // face 1, row 0
        :
        : [n] "r"(FACE_ROW_U32_CNT), [scalar] "r"(scalar), [face0] "r"(dst), [face1] "r"(dst + FACE_U32_CNT)
        : "memory", "v8", "v9");
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // scalar: first u32 holds the double-packed 16-bit value
    const Operand& buffer_Res       = params.buffer_Res; // bcast result: 32x32 16-bit tile, only row 0 is written
#endif
    std::uint32_t scalar;
    std::uint32_t* dst;

    {
        ZONE_SCOPED("INIT")
        scalar = *reinterpret_cast<volatile const std::uint32_t*>(buffer_A[0]);
        dst    = reinterpret_cast<std::uint32_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            rvv_bcast_row(scalar, dst);
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
