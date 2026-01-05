
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

// extern const struct RuntimeParams *__runtime_args_start;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en, disable_src_zero_flag>(
        formats.unpack_src,
        formats.unpack_src,
        formats.unpack_dst,
        formats.unpack_dst,
        params->TEST_FACE_R_DIM,
        params->TEST_FACE_R_DIM,
        params->num_faces,
        params->num_faces);
    _llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();
    _llk_unpack_A_init_<BROADCAST_TYPE, ACC_TO_DEST, REUSE_DEST_TYPE, unpack_to_dest>(
        params->UNPACK_TRANSPOSE_FACES,
        params->UNPACK_TRANSPOSE_WITHIN_FACE,
        params->TEST_FACE_R_DIM,
        params->num_faces,
        formats.unpack_src,
        formats.unpack_dst);
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        _llk_unpack_A_<BROADCAST_TYPE, ACC_TO_DEST, REUSE_DEST_TYPE, unpack_to_dest>(L1_ADDRESS(buffer_A[i]), formats.unpack_src, formats.unpack_dst);
    }
    _llk_unpack_A_uninit_<BROADCAST_TYPE>(params->TEST_FACE_R_DIM);
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(const volatile struct RuntimeParams *params)
{
    // Test configuration constants
    constexpr DstSync sync_mode = DstSync::SyncHalf;

    // copy srca to dest
    // Use B2D for all broadcasts except NONE (data in srcB), A2D for NONE (data in srcA)
    constexpr DataCopyType copy_type = (BROADCAST_TYPE == BroadcastType::NONE) ? DataCopyType::A2D : DataCopyType::B2D;
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, false, is_int_fpu_en>(params->num_faces, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en>(params->num_faces, formats.math);
#endif
    _llk_math_pack_sync_init_<sync_mode, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_(formats.math, formats.math);
    _llk_math_wait_for_dest_available_<sync_mode>();
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_datacopy_<copy_type, sync_mode, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(i, formats.math, formats.math);
    }
    _llk_math_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_datacopy_uninit_<copy_type, BROADCAST_TYPE, unpack_to_dest>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    // Test configuration constants
    constexpr DstSync sync_mode = DstSync::SyncHalf;
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
        formats.pack_src, formats.pack_dst, params->TEST_FACE_R_DIM * params->TEST_FACE_C_DIM * 4, params->TEST_FACE_R_DIM, TILE_C_DIM, params->num_faces);
    _llk_pack_init_<false, false>(formats.pack_dst, params->TEST_FACE_R_DIM, TILE_C_DIM, params->num_faces);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
        formats.pack_src, formats.pack_dst, params->TEST_FACE_R_DIM * params->TEST_FACE_C_DIM * 4, params->TEST_FACE_R_DIM, params->num_faces);
    _llk_pack_init_<false, false>(formats.pack_dst, params->TEST_FACE_R_DIM, params->num_faces);
#endif

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<sync_mode, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<sync_mode, false, false>();
#endif

    _llk_packer_wait_for_math_done_();
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        _llk_pack_<sync_mode, is_fp32_dest_acc_en, false>(i, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
}
#endif
