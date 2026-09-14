// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;
#endif

    {
        ZONE_SCOPED("INIT")
        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        unsigned l1_addr_16B;
        if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
        {
            l1_addr_16B = L1_ADDRESS(buffer_B[0]);
        }
        else
        {
            l1_addr_16B = L1_ADDRESS(buffer_A[0]);
        }

        // Record the descriptor under the engine UNPACKER_ENGINE_SEL actually drives (UNP_B -> UNPACR1/Unp1).
        constexpr auto unp_res = (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B) ? ckernel::trisc::BfdResource::Unp1 : ckernel::trisc::BfdResource::Unp0;
        ckernel::trisc::bfd_alloc_and_program<unp_res>(tensor_shape_A, l1_addr_16B, formats.unpack_A_src);

        if constexpr (is_fp32_dest_acc_en)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
                static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
        }
        else
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        }

        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            ckernel::trisc::bfd_current<unp_res>(), tensor_shape_A, TILE_CNT);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            const std::uint32_t src_handshake_iters = LOOP_FACTOR * TILE_CNT;
            if constexpr (is_fp32_dest_acc_en)
            {
                // 32-bit datacopy uses ELWADD and consumes both the selected
                // source and the dummy source produced by unpack.
                _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(src_handshake_iters);
            }
            else if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
            {
                _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(src_handshake_iters);
            }
            else
            {
                _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(src_handshake_iters);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, tensor_shape_A);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t DST_INDEX       = params.DST_INDEX;
#endif
    {
        ZONE_SCOPED("INIT")
        // End-to-end and math-isolate runs require FPU destination ownership.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        DataFormat src_format = static_cast<DataFormat>(formats.math);
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            const std::uint32_t src_handshake_iters = LOOP_FACTOR * TILE_CNT;
            if constexpr (is_fp32_dest_acc_en)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(src_handshake_iters);
            }
            else if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(src_handshake_iters);
            }
            else
            {
                _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(src_handshake_iters);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_eltwise_unary_datacopy_(DST_INDEX + i);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_eltwise_unary_datacopy_(DST_INDEX + i);
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
    const std::uint32_t DST_INDEX   = params.DST_INDEX;
    const Operand& buffer_Res       = params.buffer_Res;
#endif

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

        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(tensor_shape_A, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape_A, TILE_CNT);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: WH/BH isolate packs without math handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(DST_INDEX, 0 /*start_l1_tile_idx*/, tensor_shape_A);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(DST_INDEX, 0 /*start_l1_tile_idx*/, tensor_shape_A);
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
#endif
