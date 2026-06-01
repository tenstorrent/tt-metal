// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* UNPACK reads params.buffer_A[0] from L1 and emits its contents as a
   TileSlice via DEVICE_PRINT. MATH and PACK idle. */

#include "dprint_tile.h"

#include <cstdint>

#include "build.h"
#include "ckernel.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    // params.buffer_A[i] is the raw L1 byte address; tile_slice reads bytes
    // directly so we skip the LLK-specific L1_ADDRESS shift used by unpackers.
    DEVICE_PRINT("{}", llk_dprint::tile_slice<64>(params.buffer_A[0], static_cast<DataFormat>(formats.unpack_A_src), SliceRange::hw0_32_8()));
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif
