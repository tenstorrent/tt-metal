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

#include "llk_math_common.h"
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
#endif
    const std::uint32_t SELECTED_UNPACKER    = unpack_to_dest ? p_unpacr::UNP_DEST : p_unpacr::UNP_A;
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
            }
            else
            {
                // CFG persists across run types, so isolates must not inherit
                // the L1_TO_L1 unpack-to-dest handshake.
                std::uint32_t volatile* cfg                      = (std::uint32_t volatile*)TENSIX_CFG_BASE;
                cfg[UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
            }

            DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
            if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Float32)
            {
                _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>();
            }
            else if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
            {
                _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>();
            }
            else
            {
                _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>();
            }
        }

        const auto tensor_shape_A = tensor_shape_from_params(params);

        tdma_descriptor_t td_val =
            ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, buf_desc_id, formats.unpack_A_dst);

        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
        }
        else
        {
            _llk_unpack_configure_unary_<SELECTED_UNPACKER>(td_val);
        }
        _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, tensor_shape_A, num_tiles_per_unpack);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const auto tensor_shape_A = tensor_shape_from_params(params);

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                if constexpr (is_fp32_dest_acc_en)
                {
                    _perf_unpack_loop_set_valid<true, true>(LOOP_FACTOR * TILE_CNT);
                }
                else
                {
                    _perf_unpack_loop_set_valid<true, false>(LOOP_FACTOR * TILE_CNT);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_unpack_unary_operand_<SELECTED_UNPACKER>(0, tensor_shape_A);
                if constexpr (unpack_to_dest && PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

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
#endif
    if constexpr (!unpack_to_dest)
    {
        {
            ZONE_SCOPED("INIT")
            // Only L1_TO_L1 and MATH_ISOLATE use the FPU→PACK dest-dvalid
            // handshake; the remaining isolate modes have no FPU consumer.
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }

            DataFormat math_format     = static_cast<DataFormat>(formats.math);
            DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
            if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Float32)
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
            }
            else if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
            }
            else
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
            }

            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
                num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
            PROFILER_SYNC();
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
            {
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
            {
                if constexpr (is_fp32_dest_acc_en)
                {
                    _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * TILE_CNT);
                }
                else
                {
                    _perf_math_loop_clear_valid<true, false>(LOOP_FACTOR * TILE_CNT);
                }
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(i);
                    }
                }
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(i);
                    }
                    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                }
            }
            PROFILER_SYNC();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

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
    const int RELU_CONFIG           = params.RELU_CONFIG;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        // Match WH/BH PACK_ISOLATE: no math↔pack handshake; pack from whatever is in dest.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            std::uint32_t volatile* cfg                 = (std::uint32_t volatile*)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
        }

        const auto tensor_shape_A = tensor_shape_from_params(params);

        tdma_descriptor_t tdma_desc =
            ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(buffer_Res[0]), formats.pack_dst, buf_desc_id, formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        const ckernel::ReluConfig relu_config = ckernel::ReluConfig::from_packed(RELU_CONFIG);
        _llk_pack_init_<is_fp32_dest_acc_en>(buf_desc_id, tensor_shape_A, num_tiles_per_pack, relu_config);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const auto tensor_shape_A = tensor_shape_from_params(params);

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: WH/BH isolate packs without math handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(0, 0, tensor_shape_A);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(0, 0, tensor_shape_A);
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
#endif
