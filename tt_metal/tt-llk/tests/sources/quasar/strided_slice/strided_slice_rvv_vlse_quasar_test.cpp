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

static_assert(ROW_ELEM_CNT == 32, "m4 grouping below holds 32 e16 lanes at VLEN=128");

#ifdef LLK_TRISC_UNPACK

inline void rvv_set_kept_len(std::uint32_t kept_cnt)
{
    // vl = kept columns per row (<= 32); the setting is reused by every row.
    asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n" : : [n] "r"(kept_cnt) : "memory");
}

inline void rvv_strided_slice_row(const std::uint16_t* src, std::uint16_t* dst, std::uint32_t stride_bytes)
{
    asm volatile(
        "vlse16.v v16, (%[src]), %[stride]\n"
        "vse16.v v16, (%[dst])\n"
        :
        : [src] "r"(src), [dst] "r"(dst), [stride] "r"(stride_bytes)
        : "memory", "v16", "v17", "v18", "v19");
}

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
    const std::uint16_t* src;
    std::uint16_t* dst;
    std::uint32_t stride_bytes;

    {
        ZONE_SCOPED("INIT")
        // Offset to the first kept column once; every row then starts SLICE_START elements in.
        src          = reinterpret_cast<const std::uint16_t*>(buffer_A[0]) + SLICE_START;
        dst          = reinterpret_cast<std::uint16_t*>(buffer_Res[0]);
        stride_bytes = SLICE_STEP * ELEMENT_SIZE;
        rvv_set_kept_len((SLICE_END - SLICE_START + SLICE_STEP - 1) / SLICE_STEP);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t row = 0; row < TILE_ROW_COUNT; row++)
            {
                rvv_strided_slice_row(src + row * ROW_ELEM_CNT, dst + row * ROW_ELEM_CNT, stride_bytes);
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
