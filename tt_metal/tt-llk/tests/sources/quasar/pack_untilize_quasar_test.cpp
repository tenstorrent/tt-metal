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
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_A             = params.buffer_A;
#endif
    const std::uint32_t SELECTED_UNPACKER = unpack_to_dest ? p_unpacr::UNP_DEST : p_unpacr::UNP_A;
    tdma_descriptor_t td_val;
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            // Only the end-to-end path uses the unpack→pack dest-dvalid
            // handshake. Isolates deliberately have no consumer.
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
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
        else
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        buffer_descriptor_u bd_val = {0};

        bd_val.f.l1_addr_16B = buffer_A[0] / 16;
        bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
        bd_val.f.x_dim       = TEST_FACE_C_DIM;
        bd_val.f.y_dim       = TEST_FACE_R_DIM;
        bd_val.f.z_dim       = num_faces;

        td_val.buf_desc        = bd_val;
        td_val.buf_desc_id     = buf_desc_id;
        td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        if constexpr (unpack_to_dest)
        {
            _llk_unpack_configure_unary_<SELECTED_UNPACKER>(td_val);
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, BLOCK_CT_DIM);
        }
        else
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
            }
            else
            {
                _llk_unpack_configure_unary_<SELECTED_UNPACKER>(td_val);
            }
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_unpack);
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                // Math consumes every tile in every block row.
                if constexpr (is_fp32_dest_acc_en)
                {
                    // 32-bit datacopy uses ELWADD and consumes both SrcA
                    // and the dummy SrcB produced by unpack.
                    _perf_unpack_loop_set_valid<true, true>(LOOP_FACTOR * TILE_CNT);
                }
                else
                {
                    _perf_unpack_loop_set_valid<true, false>(LOOP_FACTOR * TILE_CNT);
                }
            }
        }
        else if constexpr (unpack_to_dest)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
                {
                    _llk_unpack_unary_operand_<SELECTED_UNPACKER>(block_rt * BLOCK_CT_DIM, ckernel::DEFAULT_TENSOR_SHAPE);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_unpack_unary_operand_<SELECTED_UNPACKER>(0, ckernel::DEFAULT_TENSOR_SHAPE);
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
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
#endif
    if constexpr (!unpack_to_dest)
    {
        {
            ZONE_SCOPED("INIT")
            // PACK_ISOLATE measures pack alone (WH/BH style): skip FPU→PACK dest-dvalid.
            if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE && PERF_RUN_TYPE != PerfRunType::L1_CONGESTION)
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
                    _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * BLOCK_RT_DIM * BLOCK_CT_DIM);
                }
                else
                {
                    _perf_math_loop_clear_valid<true, false>(LOOP_FACTOR * BLOCK_RT_DIM * BLOCK_CT_DIM);
                }
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
                    {
                        for (std::uint32_t block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
                        {
                            _llk_math_eltwise_unary_datacopy_(block_ct);
                        }
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
                            _llk_math_eltwise_unary_datacopy_(block_ct);
                        }
                        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                    }
                }
            }
            PROFILER_SYNC();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_Res           = params.buffer_Res;
#endif
    {
        ZONE_SCOPED("INIT")
        // Match WH/BH PACK_ISOLATE: no math↔pack handshake; pack from whatever is in dest.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            auto cfg                                    = (std::uint32_t volatile*)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else
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

        tdma_descriptor_t tdma_desc;
        std::uint32_t const buf_desc_id = 31;

        buffer_descriptor_u bd_val = {0};

        bd_val.f.l1_addr_16B = buffer_Res[0] / 16;
        bd_val.f.format      = static_cast<std::uint8_t>(formats.pack_dst);
        bd_val.f.x_dim       = TEST_FACE_C_DIM;
        bd_val.f.y_dim       = TEST_FACE_R_DIM;
        bd_val.f.z_dim       = num_faces;

        tdma_desc.buf_desc        = bd_val;
        tdma_desc.buf_desc_id     = buf_desc_id;
        tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);

        constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        _llk_pack_untilize_init_<FULL_CT_DIM, BLOCK_CT_DIM>(buf_desc_id, tensor_shape);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;
        const std::uint32_t y_stride_external       = FULL_CT_DIM * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: WH/BH isolate packs without math handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
                {
                    _llk_pack_untilize_(0, y * y_stride_external);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
                {
                    _llk_pack_untilize_(0, y * y_stride_external);
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
