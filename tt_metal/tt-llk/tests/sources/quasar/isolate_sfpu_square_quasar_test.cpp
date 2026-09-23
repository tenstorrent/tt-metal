// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated SFPU square: UNPACK2 (UNP_S) -> SrcS -> SFPU -> PACK1 -> L1.
// All logic runs in LLK_TRISC_ISOLATE_SFPU; UNPACK, MATH, PACK are stubbed.
// SrcS dvalid is controlled by UNPACR2/PACR1 TILE_INC (SetDatValid/ClrDatValid in llk_srcs.h).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

#ifdef LLK_TRISC_UNPACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/ckernel_sfpu_srcs.h"
#include "llk_sfpu_srcs_api.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const volatile FormatConfig& formats = params.formats;
#endif

    llk_sfpu_srcs_unary_init(
        L1_ADDRESS(params.buffer_A[0]),
        static_cast<DataFormat>(formats.unpack_S_src),
        static_cast<DataFormat>(formats.unpack_S_dst),
        L1_ADDRESS(params.buffer_Res[0]),
        static_cast<DataFormat>(formats.pack_S_src),
        static_cast<DataFormat>(formats.pack_S_dst),
        IMPLIED_MATH_FORMAT);

    // Resolve the SrcS register format once, outside the tile/slice loops. The SFPI slice kernel
    // uses the same layout for its input and output; PACK1 reads SrcS directly, so the harness
    // always keeps pack_S_src equal to unpack_S_dst on this path.
    // The pipeline still owns unpack/pack and SrcS completion; square only computes.
    const DataFormat srcs_format = static_cast<DataFormat>(formats.unpack_S_dst);
    LLK_ASSERT(srcs_format == static_cast<DataFormat>(formats.pack_S_src), "SrcS square requires matching unpack destination and pack source formats");
    dispatch_sfpu_srcs_format(
        srcs_format,
        [&](auto layout)
        {
            using Layout = decltype(layout);
            llk_sfpu_srcs_unary(params.TILE_CNT, srcs_format, [](int, int, int) { calculate_square_srcs<Layout::layout>(); });
        });

    wait_sfpu_idle();
    wait_unpack_idle();
    wait_pack_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif
