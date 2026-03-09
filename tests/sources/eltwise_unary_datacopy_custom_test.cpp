// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_A_custom.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    _llk_unpack_hw_configure_<false>(formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(0, 0, FACE_R_DIM, 4, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params->TILE_CNT; ++i)
    {
        _llk_unpack_A_custom_(L1_ADDRESS(params->buffer_A[i]));
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_unary_datacopy_custom.h"
#include "llk_math_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, false>();
    _llk_math_hw_configure_<false>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t tile_num = 0; tile_num < params->TILE_CNT; ++tile_num)
    {
        _llk_math_eltwise_unary_datacopy_custom_(tile_num);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, false>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<false, false, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_init_<false, false, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<DstSync::SyncHalf, false>();
#else
    _llk_pack_hw_configure_<false, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, 4);
    _llk_pack_init_<false, false>(formats.pack_dst, FACE_R_DIM, 4);
    _llk_pack_dest_init_<DstSync::SyncHalf, false, false>();
#endif

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile_num = 0; tile_num < params->TILE_CNT; ++tile_num)
    {
        _llk_pack_<DstSync::SyncHalf, false, false>(tile_num, L1_ADDRESS(params->buffer_Res[tile_num]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
}
#endif
