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
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            params->in1_tile_r_dim < FACE_R_DIM ? params->in1_tile_r_dim : FACE_R_DIM,
            params->in0_tile_r_dim < FACE_R_DIM ? params->in0_tile_r_dim : FACE_R_DIM,
            params->num_faces_B, // In1
            params->num_faces_A, // In0
            params->TILE_SIZE_UNPACK_A,
            params->TILE_SIZE_UNPACK_B);
        _llk_unpack_AB_matmul_init_<>(
            params->UNPACK_TRANSPOSE_FACES,
            params->CT_DIM,
            params->RT_DIM,
            params->KT_DIM,
            params->in1_tile_r_dim < FACE_R_DIM ? params->in1_tile_r_dim : FACE_R_DIM,
            params->in0_tile_r_dim < FACE_R_DIM ? params->in0_tile_r_dim : FACE_R_DIM,
            params->num_faces_B,     // In1
            params->num_faces_A,     // In0
            params->PARTIAL_FACE_B,  // In1
            params->PARTIAL_FACE_A); // In0
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
            for (std::uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                for (std::uint32_t j = 0; j < params->KT_DIM; j++)
                {
                    _llk_unpack_AB_matmul_<>(
                        L1_ADDRESS(params->buffer_A[0]),
                        L1_ADDRESS(params->buffer_B[0]),
                        j,
                        j * params->CT_DIM,
                        params->TILE_SIZE_UNPACK_A,
                        params->TILE_SIZE_UNPACK_B,
                        params->PARTIAL_FACE_B, // In1
                        params->PARTIAL_FACE_A, // In0
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
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    {
        ZONE_SCOPED("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        // Use tile dimensions from runtime params for tiny tiles support
        _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
            params->in0_tile_r_dim,
            params->in0_tile_c_dim,
            params->in1_tile_r_dim,
            params->in1_tile_c_dim,
            params->PARTIAL_FACE_MATH,
            params->UNPACK_TRANSPOSE_FACES,
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
            for (std::uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                for (std::uint32_t j = 0; j < params->KT_DIM; j++)
                {
                    _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(params->DST_INDEX, params->CT_DIM, params->RT_DIM);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                _llk_math_wait_for_dest_available_<dest_sync>();
                for (std::uint32_t j = 0; j < params->KT_DIM; j++)
                {
                    _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(params->DST_INDEX, params->CT_DIM, params->RT_DIM);
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
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    {
        ZONE_SCOPED("INIT")
#ifdef ARCH_BLACKHOLE
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
            formats.pack_src,
            formats.pack_dst,
            params->TILE_SIZE_PACK,
            params->in0_tile_r_dim < FACE_R_DIM ? params->in0_tile_r_dim : FACE_R_DIM,
            TILE_C_DIM,
            params->num_faces,
            params->PARTIAL_FACE_PACK);
        _llk_pack_init_<false, false, false>(
            formats.pack_dst,
            params->in0_tile_r_dim < FACE_R_DIM ? params->in0_tile_r_dim : FACE_R_DIM,
            TILE_C_DIM,
            params->num_faces,
            false /* partial_face parameter is unused on BH */);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
            formats.pack_src,
            formats.pack_dst,
            params->TILE_SIZE_PACK,
            params->in0_tile_r_dim < FACE_R_DIM ? params->in0_tile_r_dim : FACE_R_DIM,
            params->num_faces,
            params->PARTIAL_FACE_PACK);
        _llk_pack_init_<false, false>(
            formats.pack_dst, params->in0_tile_r_dim < FACE_R_DIM ? params->in0_tile_r_dim : FACE_R_DIM, params->num_faces, params->PARTIAL_FACE_PACK);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en, false>();
#endif
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
            for (std::uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < params->CT_DIM * params->RT_DIM; tile++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en>(params->DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t tile = 0; tile < params->CT_DIM * params->RT_DIM; tile++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en>(params->DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
