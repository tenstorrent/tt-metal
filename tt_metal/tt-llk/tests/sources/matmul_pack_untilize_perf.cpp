// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole pack untilize init forms: per block a one-tile matmul into DEST, packed with the pack
// untilize; PACK_UNTILIZE_INIT 0 inits once, 1 re-runs pack_untilize_dest_init per block, 2 custom_pack_untilize_dest_init.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
const std::uint32_t ct_dim             = 1;
const bool UNTILIZE                    = true;
std::uint32_t face_size                = 128;
std::uint32_t tile_size                = 16 * 16 * 4;
const ckernel::DstSync sync            = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_blocks  = static_cast<std::uint32_t>(params.NUM_BLOCKS);
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        _llk_unpack_AB_matmul_init_<>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock(LOOP_FACTOR * num_blocks, 1, 1, 1);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    _llk_unpack_AB_matmul_<>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]), 0, 0, face_size, face_size);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_blocks  = static_cast<std::uint32_t>(params.NUM_BLOCKS);
    {
        START_PERF_MEASURE("INIT")
        _llk_math_matmul_init_<MATH_FIDELITY>();
        _llk_math_pack_sync_init_<sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reconfig_remap_wrapper_(true);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_matmul_mock(LOOP_FACTOR * num_blocks, 1, 1, 1);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PACK_UNTILIZE_INIT == 1)
                    {
                        _llk_math_reconfig_remap_wrapper_(true);
                    }
                    if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                    {
                        _llk_math_wait_for_dest_available_<sync>();
                    }
                    _llk_math_matmul_<MATH_FIDELITY>(0);
                    if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                    {
                        _llk_math_dest_section_done_<sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_matmul_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"

inline void untilize_init(const FormatConfig& formats)
{
    _llk_pack_untilize_init_wrapper_<ct_dim>(formats.pack_src, formats.pack_dst, FACE_R_DIM, 4);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_blocks  = static_cast<std::uint32_t>(params.NUM_BLOCKS);
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>(formats.pack_src, formats.pack_dst, tile_size);
        _llk_pack_dest_init_wrapper_<sync, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();
        untilize_init(formats);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PACK_UNTILIZE_INIT == 1)
                    {
                        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, tile_size, TILE_C_DIM, 4, false);
                        untilize_init(formats);
                        _llk_init_packer_dest_offset_registers_<sync>();
                    }
                    else if constexpr (PACK_UNTILIZE_INIT == 2)
                    {
                        _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
                            formats.pack_src, formats.pack_dst, tile_size, FACE_R_DIM, TILE_C_DIM, 4, false, 0);
                        untilize_init(formats);
                        _llk_init_packer_dest_offset_registers_<sync>();
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    _llk_pack_untilize_wrapper_<ct_dim>(L1_ADDRESS(params.buffer_Res[block]), formats.pack_dst, FACE_R_DIM, 4, 0);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_pack_dest_section_done_<sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_pack_untilize_uninit_wrapper_(formats.pack_src, FACE_R_DIM);
}

#endif
