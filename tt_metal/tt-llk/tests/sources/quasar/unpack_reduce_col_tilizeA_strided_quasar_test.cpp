// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"
#include "tensor_shape.h"

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_reduce_col_tilizeA_strided.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT     = params.TILE_CNT;
    const std::uint32_t num_faces    = params.num_faces;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t FULL_CT_DIM  = params.FULL_CT_DIM;
    const Operand& buffer_A          = params.buffer_A;
    const Operand& buffer_B          = params.buffer_B;
#endif
    const auto tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
        static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));
    {
        START_PERF_MEASURE("INIT")
        // SrcA is tilized out of a row-major tensor, so its descriptor must address L1 in units of a single
        // 16-datum row (y_dim = z_dim = 1) for every tile shape.
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0, ckernel::trisc::L1AccessMode::Strided>(
            tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_a = ckernel::trisc::BFD_ID_UNP0;

        // SrcB holds the reduce scalar and is unpacked face by face, so it uses the tile-shaped descriptor.
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP1>(tensor_shape, L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
        constexpr std::uint8_t bfd_b = ckernel::trisc::BFD_ID_UNP1;

        _llk_unpack_reduce_col_tilizeA_strided_init_<POOL_TYPE>(bfd_a, bfd_b, FULL_CT_DIM, tensor_shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        std::uint32_t y_stride_external = FULL_CT_DIM * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    // The strided reduce MOP emits one SrcB scale face and
                    // all SrcA data faces for each tile.
                    perf_unpack_set_srcb_once_then_srca_per_face(1 /*srcb_once_count*/, num_faces);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
                {
                    for (std::uint32_t block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
                    {
                        const std::uint32_t l1_unpack_tilize_idx = block_rt * y_stride_external + block_ct;
                        _llk_unpack_reduce_col_tilizeA_strided_(tensor_shape, l1_unpack_tilize_idx, 0 /*start_l1_tile_idx_1*/);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
    // No uninit: the production op has none on Quasar.
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "params.h"
#include "tensor_shape.h"

using namespace ckernel;

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
#endif
    const auto tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
    {
        _llk_math_pack_sync_init_<dest_sync>();
    }
    DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Math isolate keeps the FPU→PACK dest-dvalid producer setup.
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(
            static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math), tensor_shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    perf_math_clear_srca_per_face_then_srcb_once(num_faces, 1 /*srcb_once_count*/);
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_reduce_<POOL_TYPE, REDUCE_DIM>(i, tensor_shape);
                }
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                // Stands in for the idle pack's release, so MATH_PACK never fills up.
                ckernel::trisc::t6_semaphore_get(ckernel::trisc::semaphore::MATH_PACK);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_math_wait_for_dest_available_();
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_reduce_<POOL_TYPE, REDUCE_DIM>(i, tensor_shape);
                }
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
}

#endif

#ifdef LLK_TRISC_PACK

#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    const auto tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
    {
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
    }
    constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, tensor_shape, TILE_CNT);
    {
        START_PERF_MEASURE("INIT")
        // PACK_ISOLATE packs without a math↔pack handshake.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }

        _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, tensor_shape);
                // Stands in for the idle math's post, so the section done's SEMGET never underflows.
                ckernel::trisc::t6_semaphore_post(ckernel::trisc::semaphore::MATH_PACK);
                _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();
                _llk_pack_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, tensor_shape);
                _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
            }
        }
        _llk_pack_reduce_mask_clear_();
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
    // The pack reduce mask clear stays at the end of TILE_LOOP above.
}

#endif
