
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

#include "llk_math_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t prev_a = (std::uint32_t)params.formats.unpack_A_src;
    const std::uint32_t prev_b = (std::uint32_t)params.formats.unpack_A_dst;
    const std::uint32_t next_a = (std::uint32_t)params.formats.pack_src;
    const std::uint32_t next_b = (std::uint32_t)params.formats.pack_dst;

    if (params.CONFIGURE_TEST_RUN_IDX == 0)
    {
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(
            /* srca_data_format */ next_a,
            /* srcb_data_format */ next_b);
    }
    else
    {
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(
            /* srca_data_format */ prev_a,
            /* srcb_data_format */ prev_b);

        if (prev_a != next_a && prev_b != next_b)
        {
            _llk_math_reconfig_data_format_<is_fp32_dest_acc_en, TO_FROM_INT8>(
                /* srca_data_format */ next_a,
                /* srcb_data_format */ next_b);
        }
        else if (prev_a != next_a)
        {
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, TO_FROM_INT8>(
                /* srca_data_format */ next_a);
        }
        else if (prev_b != next_b)
        {
            _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, TO_FROM_INT8>(
                /* srcb_data_format */ next_b);
        }
    }
}
#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif
