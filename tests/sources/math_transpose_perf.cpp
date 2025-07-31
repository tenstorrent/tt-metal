// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "data_format_inference.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensix_types.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel()
{
    volatile uint32_t* const src_a = reinterpret_cast<volatile uint32_t*>(0x1a000);
    volatile uint32_t* const src_b = reinterpret_cast<volatile uint32_t*>(0x1e000);

    {
        ZONE_SCOPED("INIT")
        _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None, unpack_to_dest>(
            formats.unpack_src, formats.unpack_dst, FACE_R_DIM, false, TILE_NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES, false, FACE_R_DIM, TILE_NUM_FACES, formats.unpack_src, formats.unpack_dst);
        ckernel::tensix_sync(); // -> perf
    }

    {
        ZONE_SCOPED("TILE_LOOP")
        for (uint32_t tile = 0; tile < TILE_CNT; tile++)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(src_a + (tile % 8) * 0x1000), UNPACK_TRANSPOSE_FACES, formats.unpack_src, formats.unpack_dst);
            _llk_unpack_set_srcb_dummy_valid_();
        }
        ckernel::tensix_sync(); // -> perf
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_transpose_dest.h"

void run_kernel()
{
    constexpr bool is32 = is_32bit_format(static_cast<DataFormat>(formats.math));

    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<>(formats.math, formats.math);

        ckernel::tensix_sync(); // -> perf
    }

    {
        ZONE_SCOPED("TILE_LOOP")

        for (int i = 0; i < TILE_CNT; i++)
        {
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

#ifdef ARCH_BLACKHOLE
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(
                UNPACK_TRANSPOSE_FACES, false, TILE_NUM_FACES, formats.math);
#else
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                UNPACK_TRANSPOSE_FACES, false, TILE_NUM_FACES, formats.math);
#endif
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                0, formats.math, formats.math);

            _llk_math_transpose_dest_init_<MATH_TRANSPOSE_FACES, is32>();
            _llk_math_transpose_dest_<MATH_TRANSPOSE_FACES, is32>(0);

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
    volatile uint32_t* const dst = reinterpret_cast<volatile uint32_t*>(0x1E000);
    {
        ZONE_SCOPED("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_<>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        ckernel::tensix_sync(); // -> perf
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (uint32_t tile = 0; tile < TILE_CNT; tile++)
        {
            _llk_packer_wait_for_math_done_();
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, L1_ADDRESS(dst + (tile % 8) * 0x1000));
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
        ckernel::tensix_sync(); // -> perf
    }
}

#endif
