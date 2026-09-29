// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"
#include "tensor_shape.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_reduce_col_tilizeA_strided.h"
#include "params.h"

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
    const auto tensor_shape = tensor_shape_from_params(params);

    {
        ZONE_SCOPED("INIT")
        // SrcA is tilized out of a row-major tensor, so its descriptor must address L1 in units of a single
        // 16-datum row (y_dim = z_dim = 1) for every tile shape.
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0, ckernel::trisc::L1AccessMode::Strided>(
            tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);

        // SrcB holds the reduce scalar and is unpacked face by face, so it uses the tile-shaped descriptor.
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(tensor_shape, L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);

        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));

        _llk_unpack_reduce_col_tilizeA_strided_init_<POOL_TYPE>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
            FULL_CT_DIM,
            tensor_shape);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
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
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "params.h"
#include "tensor_shape.h"

using namespace ckernel;

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
    const auto tensor_shape = tensor_shape_from_params(params);

    {
        ZONE_SCOPED("INIT")
        // Only end-to-end and math-isolate runs use the FPU→PACK dest-dvalid
        // handshake.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        DataFormat math_format = static_cast<DataFormat>(formats.math);
        if constexpr (is_fp32_dest_acc_en)
        {
            if (static_cast<DataFormat>(formats.pack_src) == DataFormat::Int32)
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
            }
            else
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
            }
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
        }

        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, MATH_FIDELITY>(tensor_shape);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
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
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_reduce_<POOL_TYPE, REDUCE_DIM>(i, tensor_shape);
                }
                _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

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
    const auto tensor_shape = tensor_shape_from_params(params);

    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE and L1_CONGESTION pack without a math↔pack handshake.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());

        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape, TILE_CNT);
        _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: WH/BH isolate packs without math handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, tensor_shape);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, tensor_shape);
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        _llk_pack_reduce_mask_clear_();
        PROFILER_SYNC();
    }
}

#endif
