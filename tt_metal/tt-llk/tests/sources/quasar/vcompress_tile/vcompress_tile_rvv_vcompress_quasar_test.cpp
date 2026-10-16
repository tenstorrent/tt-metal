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
constexpr std::uint32_t TILE_ROW_COUNT    = 32;

static_assert(COMPRESS_ELEM_CNT == 32, "register grouping below assumes 32 elements at VLEN=128");

#ifdef LLK_TRISC_UNPACK

inline void rvv_load_mask(const std::uint8_t* mask)
{
    // The shared mask selects the same number of elements in every row, so vcompress.vm under tail
    // undisturbed rewrites the same leading positions of v16 and the zeros past them survive.
    asm volatile(
        "vsetvli zero, %[n], e16, m4, tu, ma\n"
        "vlm.v v0, (%[mask])\n"
        "vmv.v.i v16, 0\n"
        :
        : [n] "r"(COMPRESS_ELEM_CNT), [mask] "r"(mask)
        : "memory", "v0", "v16", "v17", "v18", "v19");
}

inline void rvv_vcompress_row(const std::uint16_t* src, std::uint16_t* dst)
{
    // vcompress.vm's destination group must not overlap its source or mask
    asm volatile(
        "vle16.v v20, (%[src])\n"
        "vcompress.vm v16, v20, v0\n"
        "vse16.v v16, (%[dst])\n"
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // compress source: row-major 32x32 tile, compressed row by row
    const Operand& buffer_B         = params.buffer_B;   // shared compress mask: bit w (LSB-first) keeps source element w
    const Operand& buffer_Res       = params.buffer_Res; // compress result: row-major 32x32 tile
#endif
    const std::uint16_t* src;
    std::uint16_t* dst;

    {
        ZONE_SCOPED("INIT")
        src = reinterpret_cast<const std::uint16_t*>(buffer_A[0]);
        dst = reinterpret_cast<std::uint16_t*>(buffer_Res[0]);
        rvv_load_mask(reinterpret_cast<const std::uint8_t*>(buffer_B[0]));
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t row = 0; row < TILE_ROW_COUNT; row++)
            {
                rvv_vcompress_row(src + row * COMPRESS_ELEM_CNT, dst + row * COMPRESS_ELEM_CNT);
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
