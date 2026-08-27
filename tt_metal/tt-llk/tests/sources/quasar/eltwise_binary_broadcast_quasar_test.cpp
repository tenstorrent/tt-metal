// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_binary_broadcast_operands.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;
#endif

    {
        ZONE_SCOPED("INIT")
        set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));
        _llk_unpack_binary_broadcast_operands_init_<BROADCAST_TYPE>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
            TILE_CNT);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            const std::uint32_t srcb_dvalids_per_tile = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1u : num_faces;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    // Real broadcast unpack emits SrcA once, then SrcB once
                    // per broadcast face. Combining these handshakes blocks
                    // the second SrcB face while SrcA remains valid.
                    _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(1 /*iterations*/);
                    _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(srcb_dvalids_per_tile);
                }
            }
        }
        else // UNPACK_ISOLATE, L1_CONGESTION, L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_unpack_binary_broadcast_operands_(0 /*start_l1_tile_idx_0*/, 0 /*start_l1_tile_idx_1*/);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary_broadcast.h"
#include "params.h"

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
    {
        ZONE_SCOPED("INIT")
        // End-to-end and math-isolate runs require FPU destination ownership.
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

        _llk_math_eltwise_binary_broadcast_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY>(DEFAULT_TENSOR_SHAPE);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            const std::uint32_t srcb_only_clears = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 0u : num_faces - 1u;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    // ROW/COL math clears SrcB after each non-final face,
                    // then clears SrcA and the final SrcB together.
                    _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(srcb_only_clears);
                    _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(1 /*iterations*/);
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_eltwise_binary_broadcast_(i);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_eltwise_binary_broadcast_(i);
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

    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE / L1_CONGESTION: no math↔pack dest-dvalid handshake.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, TILE_CNT);
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
                _llk_pack_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }

        PROFILER_SYNC();
    }
}

#endif
