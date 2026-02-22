// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            4 /* num_faces */,
            4 /* num_faces */);
        _llk_unpack_AB_sub_bcast_col_init_custom_<BROADCAST_TYPE>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Per block: signal 1 SrcB + CT_DIM SrcA valids
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(params->LOOP_FACTOR); loop++)
            {
                _perf_unpack_set_valid(ckernel::SrcB);
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _perf_unpack_set_valid(ckernel::SrcA);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(params->LOOP_FACTOR); loop++)
            {
                _llk_unpack_AB_sub_bcast_col_custom_<BROADCAST_TYPE>(PERF_ADDRESS(PERF_INPUT_A, 0), PERF_ADDRESS(PERF_INPUT_B, 0), CT_DIM);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_binary_custom.h"
#include "llk_math_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY>(4, 0);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // Mock: drain the 1 SrcB + CT_DIM SrcA valids per block
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(params->LOOP_FACTOR); loop++)
            {
                _perf_math_clear_valid(ckernel::SrcB);
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _perf_math_clear_valid(ckernel::SrcA);
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // No dest sync in this mode (pack returns immediately).
            // Custom blocked sub+bcast math consumes the valids produced by unpack mock.
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(params->LOOP_FACTOR); loop++)
            {
                _llk_math_eltwise_binary_bcast_reuse_custom_(CT_DIM);
            }
        }
        else // L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(params->LOOP_FACTOR); loop++)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                _llk_math_eltwise_binary_bcast_reuse_custom_(CT_DIM);
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_<false, false>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No wait_for_math_done (math returns/mocks immediately in these modes)
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(params->LOOP_FACTOR); loop++)
            {
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, PERF_ADDRESS(PERF_OUTPUT, i));
                }
            }
        }
        else // L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < static_cast<std::uint32_t>(params->LOOP_FACTOR); loop++)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, PERF_ADDRESS(PERF_OUTPUT, i));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
