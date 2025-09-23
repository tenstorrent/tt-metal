// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"

void run_kernel()
{
    constexpr uint32_t src_a = 0x1a000;

    {
        ZONE_SCOPED("INIT")
        _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None, false>(
            formats.unpack_src, formats.unpack_dst, FACE_R_DIM, UNPACK_TRANSPOSE_WITHIN_FACE, TILE_NUM_FACES);
        _llk_unpack_A_init_<>(UNPACK_TRANSPOSE_FACES, UNPACK_TRANSPOSE_WITHIN_FACE, FACE_R_DIM, TILE_NUM_FACES, formats.unpack_src, formats.unpack_dst);
        ckernel::tensix_sync(); // -> perf
    }

    {
        ZONE_SCOPED("TILE_LOOP")
        for (uint32_t tile = 0; tile < TILE_CNT; tile++)
        {
            _llk_unpack_A_<>(
                L1_ADDRESS(src_a + (tile % 8) * 0x1000), UNPACK_TRANSPOSE_FACES, formats.unpack_src, formats.unpack_dst); // TODO SS<-LP use PERF_ADDRESS here
        }
        ckernel::tensix_sync(); // -> perf
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<>(formats.math, formats.math);

#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(
            UNPACK_TRANSPOSE_FACES, UNPACK_TRANSPOSE_WITHIN_FACE, TILE_NUM_FACES, formats.math);
#else
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
            UNPACK_TRANSPOSE_FACES, UNPACK_TRANSPOSE_WITHIN_FACE, TILE_NUM_FACES, formats.math);
#endif
        ckernel::tensix_sync(); // -> perf
    }

    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            // _llk_unpack_A sets both A and B valid
            return _perf_math_loop_clear_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
        }

        for (int i = 0; i < TILE_CNT; i++)
        {
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                0, formats.math, formats.math);
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
        ckernel::tensix_sync(); // -> perf
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel()
{
    constexpr uint32_t dst = 0x1E000;
    {
        ZONE_SCOPED("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_<>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        ckernel::tensix_sync(); // -> perf
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }

        for (uint32_t tile = 0; tile < TILE_CNT; tile++)
        {
            _llk_packer_wait_for_math_done_();
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, L1_ADDRESS(dst + (tile % 8) * 0x1000)); // TODO SS<-LP use PERF_ADDRESS here
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
        ckernel::tensix_sync(); // -> perf
    }
}

#endif
