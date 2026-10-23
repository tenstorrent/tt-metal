
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Configuring a format directly (run_idx 0) must leave the same packer state as
// reconfiguring to it from another format (run_idx 1). FormatConfig carries the
// prev formats in the unpack_A slots and the next formats in the pack slots.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t prev_src = (std::uint32_t)params.formats.unpack_A_src;
    const std::uint32_t prev_dst = (std::uint32_t)params.formats.unpack_A_dst;
    const std::uint32_t next_src = (std::uint32_t)params.formats.pack_src;
    const std::uint32_t next_dst = (std::uint32_t)params.formats.pack_dst;

    // Distinct prev/next tile sizes; both paths need to hit NEXT_SIZE.
    constexpr std::uint32_t PREV_SIZE = 16 * 16 * 2;
    constexpr std::uint32_t NEXT_SIZE = 16 * 16 * 4;

    if (params.CONFIGURE_TEST_RUN_IDX == 0)
    {
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(next_src, next_dst, NEXT_SIZE);
    }
    else
    {
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(prev_src, prev_dst, PREV_SIZE);
        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(next_src, next_dst, NEXT_SIZE);
    }

    ckernel::packer::are_packers_configured_correctly(next_src, next_dst);
}

#endif
