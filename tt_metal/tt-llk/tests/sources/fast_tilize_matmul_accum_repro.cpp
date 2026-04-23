// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multi-iteration fast tilize → matmul with accumulation (conv2d pattern).
// Iter 0: fast_tilize A_row0 → matmul(A0 × B) → pack partial to Res[KT]
// Iter 1+: fast_tilize A_row[i] → reload Res[KT] to DEST → matmul(A[i] × B, accum) → pack to Res[KT]
// Final Res[KT] = sum of all A_row[i] × B.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
std::uint32_t tile_size                = 128;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM    = params.BLOCK_CT_DIM;
    const std::uint32_t NUM_ITERS = params.LOOP_FACTOR;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);

    for (std::uint32_t iter = 0; iter < NUM_ITERS; iter++)
    {
        // === Phase 1: Fast tilize A_row[iter] → Res[0..KT-1] ===
        // Base address is programmed inside _llk_unpack_fast_tilize_block_ via
        // _llk_unpack_configure_single_address_ (respects current cfg context).
        _llk_unpack_fast_tilize_init_(formats.unpack_A_dst, KT_DIM, KT_DIM > 5 ? 4 : KT_DIM == 5 ? 2 : KT_DIM);
        _llk_unpack_fast_tilize_reinit_xdim_(KT_DIM);
        _llk_unpack_fast_tilize_block_(L1_ADDRESS(params.buffer_A[iter * KT_DIM]), 0, formats.unpack_A_src, KT_DIM, 4, 0);

        _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();

        // Wait for tilize pack done
        t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
        t6_semaphore_get<>(semaphore::PACK_DONE);

        // === Phase 2 (iter>0): Reload partial from Res[KT] ===
        if (iter > 0)
        {
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, false>(formats.unpack_A_src, formats.unpack_A_dst, tile_size);
#ifdef ARCH_BLACKHOLE
            _llk_unpack_tilize_uninit_(formats.unpack_A_dst, 4, FACE_R_DIM);
#else
            _llk_unpack_tilize_uninit_(formats.unpack_A_dst, FACE_R_DIM);
#endif
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                0, 0, FACE_R_DIM, 4, formats.unpack_A_src, formats.unpack_A_dst);
            _llk_unpack_A_<>(L1_ADDRESS(params.buffer_Res[KT_DIM]));
        }

        // === Phase 3: Matmul (tilized_A[iter] × B) ===
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, false>(formats.unpack_A_src, formats.unpack_A_dst, tile_size);
        _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, false>(formats.unpack_B_src, formats.unpack_B_dst, tile_size);
        if (iter == 0)
        {
#ifdef ARCH_BLACKHOLE
            _llk_unpack_tilize_uninit_(formats.unpack_A_dst, 4, FACE_R_DIM);
#else
            _llk_unpack_tilize_uninit_(formats.unpack_A_dst, FACE_R_DIM);
#endif
        }
        _llk_unpack_AB_matmul_init_<>(0, 1, 1, KT_DIM, FACE_R_DIM, FACE_R_DIM, 4, 4, false, false);

        for (std::uint32_t k = 0; k < KT_DIM; k++)
        {
            _llk_unpack_AB_matmul_<>(L1_ADDRESS(params.buffer_Res[0]), L1_ADDRESS(params.buffer_B[0]), k, k, tile_size, tile_size, false, false, 1, 1, KT_DIM);
        }

        // Wait for matmul pack done
        t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
        t6_semaphore_get<>(semaphore::PACK_DONE);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM    = params.BLOCK_CT_DIM;
    const std::uint32_t NUM_ITERS = params.LOOP_FACTOR;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (std::uint32_t iter = 0; iter < NUM_ITERS; iter++)
    {
        // === Phase 1: Fast tilize math ===
        _llk_math_fast_tilize_init_<is_fp32_dest_acc_en>(formats.math);

        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_fast_tilize_block_<is_fp32_dest_acc_en>(0, formats.math, 4);
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

        _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(formats.math);

        // Compensate odd section_done count
        update_dest_offset_id();
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());

        // === Phase 2+3: Acquire DEST, optionally reload, then matmul ===
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

        if (iter > 0)
        {
            // Datacopy: reload partial from SrcA to DEST
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(4, formats.math);
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                0, formats.math, formats.math, 4);
            // No section_done — matmul accumulates on same DEST half
        }

        // Matmul: accumulates on top of DEST
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(formats.math);
        _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, false>(formats.math);
        _llk_math_matmul_init_<MathFidelity::HiFi4, 0>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, 1, 1);

        for (std::uint32_t k = 0; k < KT_DIM; k++)
        {
            _llk_math_matmul_<MathFidelity::HiFi4, 0>(0, 1, 1);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_fast_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM    = params.BLOCK_CT_DIM;
    const std::uint32_t NUM_ITERS = params.LOOP_FACTOR;

    TTI_SEMINIT(2, 0, 1 << semaphore::PACK_DONE);

    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
        formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM), FACE_R_DIM, TILE_C_DIM, 4, false);

    for (std::uint32_t iter = 0; iter < NUM_ITERS; iter++)
    {
        // === Phase 1: Fast tilize pack ===
        _llk_pack_fast_tilize_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, formats.pack_dst, KT_DIM, 4);

        _llk_packer_wait_for_math_done_();
        _llk_pack_fast_tilize_block_(0, L1_ADDRESS(params.buffer_Res[0]), KT_DIM, 4);
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

        _llk_pack_fast_tilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(formats.pack_dst, FACE_R_DIM, 4);

        // Compensate odd section_done
        flip_packer_dest_offset_id();
        select_packer_dest_registers<DstSync::SyncHalf>();

        t6_semaphore_post<>(semaphore::PACK_DONE);

        // === Phase 2+3: Pack matmul result ===
        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, tile_size);
#ifdef ARCH_BLACKHOLE
        _llk_pack_init_<false, false, false>(formats.pack_dst);
#endif

        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(params.buffer_Res[KT_DIM]));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        t6_semaphore_post<>(semaphore::PACK_DONE);
    }
}

#endif
