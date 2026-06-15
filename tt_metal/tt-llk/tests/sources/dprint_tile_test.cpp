// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*  UNPACK emits two tile_slice records in sequence:
    1. tile_slice_from_l1<64> over the 16-cell hw0_32_8 slice, which fits in 64 bytes.
    2. tile_slice_from_l1<64> over the 64-cell hw0_32_4 slice, which gets truncated.
*/

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "dprint.h"
#include "sfpu_stub.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    const DataFormat src_fmt = static_cast<DataFormat>(formats.unpack_A_src);
    DEVICE_PRINT("{}", tile_slice_from_l1<64>(params.buffer_A[0], src_fmt, SliceRange::hw0_32_8()));
    DEVICE_PRINT("{}", tile_slice_from_l1<64>(params.buffer_A[0], src_fmt, SliceRange::hw0_32_4()));
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
