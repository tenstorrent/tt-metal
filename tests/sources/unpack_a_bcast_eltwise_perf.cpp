// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

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

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_bcastA_B_hw_config_<false>(formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst);
        _llk_unpack_bcastA_B_init_();
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
            return _perf_unpack_loop_set_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
        }
        else
        {
            for (uint32_t i = 0; i < TILE_CNT / SRCA_REUSE_COUNT; i++)
            {
                _llk_unpack_bcastA_B_(L1_ADDRESS(buffer_A[i]), L1_ADDRESS(buffer_B[i * SRCA_REUSE_COUNT]), SRCA_REUSE_COUNT);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<false, false>(formats.math, formats.math);
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, 0>(SRCA_REUSE_COUNT);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            return _perf_math_loop_clear_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            for (uint32_t i = 0; i < TILE_CNT / SRCA_REUSE_COUNT; i++)
            {
                _llk_math_eltwise_binary_(i * SRCA_REUSE_COUNT /* dst_index */);
            }
        }
        else
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            for (uint32_t i = 0; i < TILE_CNT / SRCA_REUSE_COUNT; i++)
            {
                _llk_math_eltwise_binary_(i * SRCA_REUSE_COUNT /* dst_index */);
            }
        }
        PROFILER_SYNC();
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_<>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        _llk_packer_wait_for_math_done_();
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile, PERF_ADDRESS(PERF_OUTPUT, tile));
            }
        }
        else
        {
            for (uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile, PERF_ADDRESS(PERF_OUTPUT, tile));
            }
        }
        PROFILER_SYNC();
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
