// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Matmul initialised for an INIT_RT_DIM x INIT_CT_DIM block and called for RT_DIM x CT_DIM blocks without a re-init.
// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        params.num_faces_A,
        params.num_faces_B,
        params.TILE_SIZE_UNPACK_A,
        params.TILE_SIZE_UNPACK_B);
    _llk_unpack_AB_matmul_init_<>(0 /* transpose */, INIT_CT_DIM, INIT_RT_DIM, params.KT_DIM);
    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        // in0 is RT_DIM x KT_DIM tiles, in1 is KT_DIM x INIT_CT_DIM tiles of which the call reads the first CT_DIM columns
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(params.buffer_A[0]),
            L1_ADDRESS(params.buffer_B[0]),
            j,
            j * INIT_CT_DIM,
            params.TILE_SIZE_UNPACK_A,
            params.TILE_SIZE_UNPACK_B,
            false /* unpA_partial_face */,
            false /* unpB_partial_face */,
            params.CT_DIM,
            params.RT_DIM,
            params.KT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

#if defined(ARCH_BLACKHOLE) && defined(MATMUL_ROW_MOP)
#define MATMUL_MATH_TEMPLATE_ARGS MATH_FIDELITY, THROTTLE_LEVEL, true
#else
#define MATMUL_MATH_TEMPLATE_ARGS MATH_FIDELITY, THROTTLE_LEVEL
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_matmul_init_<MATMUL_MATH_TEMPLATE_ARGS>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /* partial_face */, 0 /* transpose */, INIT_CT_DIM, INIT_RT_DIM);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_math_matmul_<MATMUL_MATH_TEMPLATE_ARGS>(0 /* dst_index */, params.CT_DIM, params.RT_DIM);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_matmul_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    // Without PACK_RESULT the pack thread leaves no wait on the math thread behind, so a stopped math thread wedges nothing.
    if constexpr (PACK_RESULT)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t i = 0; i < params.RT_DIM * params.CT_DIM; i++)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
