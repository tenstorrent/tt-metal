
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

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    if (params.RECONFIG_RUN_IDX == 0)
    {
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
            /* pack_src_format */ PACK_SRC_FORMAT_NEXT,
            /* pack_dst_format */ PACK_DST_FORMAT_NEXT,
            /* tile_size */ static_cast<std::uint32_t>(params.TILE_SIZE_NEXT),
            /* face_r_dim */ static_cast<std::uint32_t>(params.FACE_R_DIM_NEXT),
#ifdef ARCH_BLACKHOLE
            /* tile_c_dim */ static_cast<std::uint32_t>(params.TILE_C_DIM_NEXT),
#endif
            /* num_faces */ static_cast<std::uint32_t>(params.NUM_FACES_NEXT),
            /* partial_face */ static_cast<bool>(params.PARTIAL_FACE_NEXT),
            /* narrow_tile */ static_cast<bool>(params.NARROW_TILE_NEXT),
            /* relu_config */ 0);
    }
    else
    {
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
            /* pack_src_format */ PACK_SRC_FORMAT,
            /* pack_dst_format */ PACK_DST_FORMAT,
            /* tile_size */ static_cast<std::uint32_t>(params.TILE_SIZE),
            /* face_r_dim */ static_cast<std::uint32_t>(params.FACE_R_DIM),
#ifdef ARCH_BLACKHOLE
            /* tile_c_dim */ static_cast<std::uint32_t>(params.TILE_C_DIM),
#endif
            /* num_faces */ static_cast<std::uint32_t>(params.NUM_FACES),
            /* partial_face */ static_cast<bool>(params.PARTIAL_FACE),
            /* narrow_tile */ static_cast<bool>(params.NARROW_TILE),
            /* relu_config */ 0);

        if (PACK_SRC_FORMAT != PACK_SRC_FORMAT_NEXT || PACK_DST_FORMAT != PACK_DST_FORMAT_NEXT)
        {
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, true>(
                /* pack_src_format */ PACK_SRC_FORMAT_NEXT,
                /* pack_dst_format */ PACK_DST_FORMAT_NEXT,
                /* tile_size */ static_cast<std::uint32_t>(params.TILE_SIZE_NEXT),
                /* face_r_dim */ static_cast<std::uint32_t>(params.FACE_R_DIM_NEXT),
#ifdef ARCH_BLACKHOLE
                /* tile_c_dim */ static_cast<std::uint32_t>(params.TILE_C_DIM_NEXT),
#endif
                /* num_faces */ static_cast<std::uint32_t>(params.NUM_FACES_NEXT),
                /* partial_face */ static_cast<bool>(params.PARTIAL_FACE_NEXT),
                /* narrow_tile */ static_cast<bool>(params.NARROW_TILE_NEXT));
        }
    }
}

#endif
