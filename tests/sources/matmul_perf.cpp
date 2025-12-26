// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

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

static constexpr uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_src,
            formats.unpack_src,
            formats.unpack_dst,
            formats.unpack_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            params->num_faces_A,
            params->num_faces_B,
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B);
        _llk_unpack_AB_matmul_init_<>(
            params->UNPACK_TRANSPOSE_FACES,
            params->CT_DIM,
            params->RT_DIM,
            params->KT_DIM,
            FACE_R_DIM,
            FACE_R_DIM,
            TILE_NUM_FACES,
            TILE_NUM_FACES,
            false,
            false);
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
            return _perf_unpack_matmul_mock(params->LOOP_FACTOR, params->RT_DIM, params->KT_DIM, params->CT_DIM);
        }
        else
        {
            for (uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                for (uint32_t j = 0; j < params->KT_DIM; j++)
                {
                    _llk_unpack_AB_matmul_<>(
                        L1_ADDRESS(buffer_A[0]),
                        L1_ADDRESS(buffer_B[0]),
                        j,
                        j * params->CT_DIM,
                        TILE_SIZE_UNPACK_A,
                        TILE_SIZE_UNPACK_B,
                        /* partial face */ false,
                        /* partial face */ false,
                        params->CT_DIM,
                        params->RT_DIM,
                        params->KT_DIM);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")
        _llk_math_hw_configure_(formats.math, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
            /* tile A */ TILE_R_DIM,
            /* tile A */ TILE_C_DIM,
            /* tile B */ TILE_R_DIM,
            /* tile B */ TILE_C_DIM,
            /* partial face */ false,
            /* transpose */ false,
            params->CT_DIM,
            params->RT_DIM);

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
            return _perf_math_matmul_mock(params->LOOP_FACTOR, params->RT_DIM, params->KT_DIM, params->CT_DIM);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                for (uint32_t j = 0; j < params->KT_DIM; j++)
                {
                    _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(
                        /* dest_index */ 0, params->CT_DIM, params->RT_DIM);
                }
            }
        }
        else
        {
            for (uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                _llk_math_wait_for_dest_available_<dest_sync>();
                for (uint32_t j = 0; j < params->KT_DIM; j++)
                {
                    _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(
                        /* dest_index */ 0, params->CT_DIM, params->RT_DIM);
                }
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_C_DIM * TILE_R_DIM);
        _llk_pack_init_<
            /* untilize */ false,
            /* zero_output */ false>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                for (uint32_t tile = 0; tile < params->CT_DIM * params->RT_DIM; tile++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
            }
        }
        else
        {
            for (uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();
                for (uint32_t tile = 0; tile < params->CT_DIM * params->RT_DIM; tile++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
