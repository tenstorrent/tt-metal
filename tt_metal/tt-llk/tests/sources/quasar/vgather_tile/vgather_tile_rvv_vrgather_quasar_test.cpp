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
constexpr std::uint32_t TILE_ROW_COUNT  = 32;

static_assert(GATHER_ELEM_CNT == 32, "register grouping below assumes 32 elements at VLEN=128");

#ifdef LLK_TRISC_UNPACK

inline void rvv_load_indices(const std::uint8_t* idx)
{
    asm volatile(
        "vsetvli zero, %[n], e8, m2, ta, ma\n"
        "vle8.v v8, (%[idx])\n"
        "vsetvli zero, %[n], e16, m4, ta, ma\n"
        "vzext.vf2 v12, v8\n"
        :
        : [n] "r"(GATHER_ELEM_CNT), [idx] "r"(idx)
        : "memory", "v8", "v9", "v12", "v13", "v14", "v15");
}

inline void rvv_vrgather_row(const std::uint16_t* src, std::uint16_t* dst)
{
    // vrgather.vv's destination group (v16) must not overlap its source (v20) or index (v12) groups.
    asm volatile(
        "vle16.v v20, (%[src])\n"
        "vrgather.vv v16, v20, v12\n"
        "vse16.v v16, (%[dst])\n"
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // gather source: row-major 32x32 tile, gathered row by row
    const Operand& buffer_B         = params.buffer_B;   // shared gather indices: GATHER_ELEM_CNT bytes
    const Operand& buffer_Res       = params.buffer_Res; // gather result: row-major 32x32 tile
#endif
    const std::uint16_t* src;
    std::uint16_t* dst;

    {
        ZONE_SCOPED("INIT")
        src = reinterpret_cast<const std::uint16_t*>(buffer_A[0]);
        dst = reinterpret_cast<std::uint16_t*>(buffer_Res[0]);
        // The index vector is read from L1 once and kept in v12..v15.
        rvv_load_indices(reinterpret_cast<const std::uint8_t*>(buffer_B[0]));
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t row = 0; row < TILE_ROW_COUNT; row++)
            {
                rvv_vrgather_row(src + row * GATHER_ELEM_CNT, dst + row * GATHER_ELEM_CNT);
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
