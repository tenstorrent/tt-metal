// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /*num_faces */, 4 /* num_faces */);
    _llk_unpack_AB_sub_bcast_col_init_custom_<BROADCAST_TYPE>();

    _llk_unpack_AB_sub_bcast_col_custom_<BROADCAST_TYPE>(L1_ADDRESS(params->buffer_A[0]), L1_ADDRESS(params->buffer_B[0]), CT_DIM);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_binary_custom.h"
#include "llk_math_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY>(4, 0);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // call custom LLK
    _llk_math_eltwise_binary_bcast_reuse_custom_(CT_DIM);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#endif

    _llk_pack_init_<false, false>(formats.pack_dst);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, false, false>();
#endif

    // wait for math to finish
    _llk_packer_wait_for_math_done_();

    // pack the result
    for (int i = 0; i < params->TILE_CNT; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, L1_ADDRESS(params->buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
