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

inline void rvv_vcompress(const std::uint16_t* src, const std::uint8_t* mask, std::uint16_t* dst)
{
    static_assert(COMPRESS_ELEM_CNT == 32, "register grouping below assumes 32 elements at VLEN=128");

    asm volatile(
        "vsetvli zero, %[n], e16, m4, tu, ma\n" // set 16-bit elements and group 4 registers as one operand; tail undisturbed keeps the zero fill
        "vlm.v v0, (%[mask])\n"                 // load the bit-packed mask straight into the mask register (vl / 8 bytes)
        "vle16.v v20, (%[src])\n"               // load data
        "vmv.v.i v16, 0\n"                      // zero the result so positions past the selected count read as 0
        "vcompress.vm v16, v20, v0\n"           // perform rvv-native compress; v16 must not overlap v20 or v0
        "vse16.v v16, (%[dst])\n"               // store back to L1
        :
        : [n] "r"(COMPRESS_ELEM_CNT), [mask] "r"(mask), [src] "r"(src), [dst] "r"(dst)
        : "memory", "v0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;   // compress source: COMPRESS_ELEM_CNT contiguous elements
    const Operand& buffer_B         = params.buffer_B;   // compress mask: bit w (LSB-first) keeps source element w
    const Operand& buffer_Res       = params.buffer_Res; // compress result: COMPRESS_ELEM_CNT contiguous elements
#endif
    const std::uint16_t* src;
    const std::uint8_t* mask;
    std::uint16_t* dst;

    {
        ZONE_SCOPED("INIT")
        src  = reinterpret_cast<const std::uint16_t*>(buffer_A[0]);
        mask = reinterpret_cast<const std::uint8_t*>(buffer_B[0]);
        dst  = reinterpret_cast<std::uint16_t*>(buffer_Res[0]);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            rvv_vcompress(src, mask, dst);
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
