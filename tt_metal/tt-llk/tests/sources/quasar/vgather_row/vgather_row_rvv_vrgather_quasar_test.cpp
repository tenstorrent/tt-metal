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

inline void rvv_vrgather(const std::uint16_t* src, const std::uint8_t* idx, std::uint16_t* dst)
{
    static_assert(GATHER_ELEM_CNT == 32, "register grouping below assumes 32 elements at VLEN=128");

    asm volatile(
        "vsetvli zero, %[n], e8, m2, ta, ma\n"  // set 8-bit elements and group 2 registers as one operand (32 elements of 8 bits)
        "vle8.v v8, (%[idx])\n"                 // load indices
        "vsetvli zero, %[n], e16, m4, ta, ma\n" // set 16-bit elements and group 4 registers as one operand (32 elements of 16 bits)
        "vle16.v v20, (%[src])\n"               // load data
        "vzext.vf2 v12, v8\n"                   // zero-extend indices
        "vrgather.vv v16, v20, v12\n"           // perform rvv-native gather
        "vse16.v v16, (%[dst])\n"               // load back into L1
        :
        : [n] "r"(GATHER_ELEM_CNT), [idx] "r"(idx), [src] "r"(src), [dst] "r"(dst)
        : "memory", "v8", "v9", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // gather source: GATHER_ELEM_CNT contiguous elements
    const Operand& buffer_B         = params.buffer_B;   // gather indices: GATHER_ELEM_CNT bytes
    const Operand& buffer_Res       = params.buffer_Res; // gather result: GATHER_ELEM_CNT contiguous elements
#endif
    const std::uint16_t* src;
    const std::uint8_t* idx;
    std::uint16_t* dst;

    {
        ZONE_SCOPED("INIT")
        src = reinterpret_cast<const std::uint16_t*>(buffer_A[0]);
        idx = reinterpret_cast<const std::uint8_t*>(buffer_B[0]);
        dst = reinterpret_cast<std::uint16_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            rvv_vrgather(src, idx, dst);
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
