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
    volatile uint32_t* const src_a = reinterpret_cast<volatile uint32_t*>(0x1a000);
    volatile uint32_t* const src_b = reinterpret_cast<volatile uint32_t*>(0x1e000);

    {
        ZONE_SCOPED("INIT")
        _llk_unpack_AB_hw_configure_<is_fp32_dest_acc_en>(UNPACK_A_IN, UNPACK_B_IN, UNPACK_A_OUT, UNPACK_B_OUT, 8, false, 4);
        _llk_unpack_AB_init_<>(FACE_R_DIM, TILE_NUM_FACES, false, false, dest_acc_en_input);
        tensix_sync(); // -> perf
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
            for (uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_unpack_AB_<>(L1_ADDRESS(src_a + (tile % 8) * 0x1000), L1_ADDRESS(src_b + (tile % 8) * 0x1000), false);
            }
        }
        tensix_sync(); // -> perf
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
        _llk_math_hw_configure_<>(MATH_FORMAT, MATH_FORMAT);
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BroadcastType::NONE, MATH_FIDELITY>(TILE_NUM_FACES, false, false);
        tensix_sync(); // -> perf
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            return _perf_math_loop_clear_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY>(
                    TILE_NUM_FACES, 0, false);
            }
        }
        else
        {
            for (uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY>(
                    TILE_NUM_FACES, 0, false);
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        tensix_sync(); // -> perf
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
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(PACK_IN, PACK_OUT, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_<>(PACK_OUT);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        tensix_sync(); // -> perf
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, L1_ADDRESS(dst + (tile % 8) * 0x1000));
            }
        }
        else
        {
            for (uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_packer_wait_for_math_done_();
                _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, L1_ADDRESS(dst + (tile % 8) * 0x1000));
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        tensix_sync(); // -> perf
    }
}

#endif
