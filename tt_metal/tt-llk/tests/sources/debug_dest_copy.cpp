// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_debug.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif

#ifdef LLK_TRISC_MATH

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    const DataFormat l1_fmt          = static_cast<DataFormat>(formats.unpack_A_src);
    constexpr std::uint32_t TILE_IDX = 0;

    // L1 -> DEST -> L1 round trip through the RISC-V debug dest.
    dbg_copy_dest_tile<DbgDestTileOp::Write, MathThreadId>(l1_fmt, TILE_IDX, reinterpret_cast<void*>(params.buffer_A[0]));
    dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(l1_fmt, TILE_IDX, reinterpret_cast<void*>(params.buffer_Res[0]));
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
