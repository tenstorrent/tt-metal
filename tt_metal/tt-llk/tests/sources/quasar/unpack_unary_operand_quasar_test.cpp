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
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const int num_faces_r_dim_A         = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A         = params.num_faces_c_dim_A;
    const Operand& buffer_A             = params.buffer_A;
    const Operand& buffer_B             = params.buffer_B;
#endif
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        if (unpack_to_dest)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
        }
        else
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        const auto tensor_shape_A   = tensor_shape_from_face_dims(TEST_FACE_R_DIM, TEST_FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);
        constexpr bool TRANSPOSE_EN = UNPACK_TRANSPOSE_FACES && UNPACK_TRANSPOSE_WITHIN_FACE;

        unsigned l1_addr_16B;
        if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_A || UNPACKER_ENGINE_SEL == p_unpacr::UNP_DEST)
        {
            l1_addr_16B = L1_ADDRESS(buffer_A[0]);
        }
        else if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
        {
            l1_addr_16B = L1_ADDRESS(buffer_B[0]);
        }

        tdma_descriptor_t td_val = ckernel::trisc::construct_tdma_desc(tensor_shape_A, l1_addr_16B, formats.unpack_A_src, buf_desc_id, formats.unpack_A_dst);

        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        if (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
        }
        else
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
        }

        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, TRANSPOSE_EN, is_fp32_dest_acc_en>(buf_desc_id, tensor_shape_A, num_tiles_per_unpack);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const auto tensor_shape_A = tensor_shape_from_face_dims(TEST_FACE_R_DIM, TEST_FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
            {
                _perf_unpack_loop_set_valid<true, false>(LOOP_FACTOR);
            }
            else
            {
                _perf_unpack_loop_set_valid<false, true>(LOOP_FACTOR);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0, tensor_shape_A);
                if (unpack_to_dest)
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

            DataFormat src_format = static_cast<DataFormat>(formats.math);
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

            _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(
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
                if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
                {
                    _perf_math_loop_clear_valid<true, false>(LOOP_FACTOR * TILE_CNT);
                }
                else
                {
                    _perf_math_loop_clear_valid<false, true>(LOOP_FACTOR * TILE_CNT);
                }
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(num_faces * TEST_FACE_R_DIM /*num_rows_per_tile*/, i);
                    }
                }
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(num_faces * TEST_FACE_R_DIM /*num_rows_per_tile*/, i);
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
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const int num_faces_r_dim_A         = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A         = params.num_faces_c_dim_A;
    const Operand& buffer_Res           = params.buffer_Res;
#endif
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        // Match WH/BH PACK_ISOLATE: no math↔pack handshake; pack from whatever is in dest.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            auto cfg = (std::uint32_t volatile *)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else
        {
            if (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
        }

        const auto tensor_shape_A = tensor_shape_from_face_dims(TEST_FACE_R_DIM, TEST_FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);

        tdma_descriptor_t tdma_desc =
            ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(buffer_Res[0]), formats.pack_dst, buf_desc_id, formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        _llk_pack_init_(buf_desc_id, tensor_shape_A, num_tiles_per_pack);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        const auto tensor_shape_A = tensor_shape_from_face_dims(TEST_FACE_R_DIM, TEST_FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);

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
