// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const Operand& buffer_A             = params.buffer_A;
    const Operand& buffer_B             = params.buffer_B;
#endif
    tdma_descriptor_t td_val;
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
        }
        else
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        buffer_descriptor_u bd_val = {0};

        unsigned l1_addr_16B;
        if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
        {
            l1_addr_16B = L1_ADDRESS(buffer_B[0]);
        }
        else
        {
            l1_addr_16B = L1_ADDRESS(buffer_A[0]);
        }

        bd_val.f.l1_addr_16B = l1_addr_16B;
        bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
        bd_val.f.x_dim       = TEST_FACE_C_DIM;
        bd_val.f.y_dim       = TEST_FACE_R_DIM;
        bd_val.f.z_dim       = num_faces;

        td_val.buf_desc        = bd_val;
        td_val.buf_desc_id     = buf_desc_id;
        td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
        }
        else
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
        }

        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_unpack);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Real unpack produces all SrcA tiles first, then the dummy SrcB
            // tiles consumed by transpose. Preserve that ordering: the two
            // MOPs cannot be mocked as one simultaneous SrcAB handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if constexpr (!unpack_to_dest)
                {
                    _perf_unpack_loop_set_valid<true, false>(TILE_CNT);
                }
                _perf_unpack_loop_set_valid<false, true>(TILE_CNT);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0, ckernel::DEFAULT_TENSOR_SHAPE);

                if constexpr (unpack_to_dest)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                    }
                }

                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_set_srcB_dummy_valid_();
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
#include "llk_math_transpose_dest.h"
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
    const int DST_INDEX                 = params.DST_INDEX;
#endif
    const DataFormat math_format     = static_cast<DataFormat>(formats.math);
    const DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (!unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE && unpack_to_dest)
        {
            // L1_TO_L1 leaves FPU configured as the middle client in the
            // UNPACK→FPU→PACK chain. Math isolate has no unpack destination
            // pulse, so make FPU the producer and restore immediate ownership
            // of the destination register.
            set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

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
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // Real unpack emits SrcA and dummy SrcB as two ordered batches.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if constexpr (!unpack_to_dest)
                {
                    _perf_math_loop_clear_valid<true, false>(TILE_CNT);
                }
                _perf_math_loop_clear_valid<false, true>(TILE_CNT);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if constexpr (!unpack_to_dest)
                {
                    // Datacopy and transpose both use bank0's instruction
                    // buffer, so each operation must program its MOP before
                    // execution. Initializing both in INIT overwrites datacopy.
                    _configure_default_alu_data_format_state_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(math_format, math_format);
                    _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(
                        num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(DST_INDEX + i);
                    }
                }
                _configure_mov_ops_explicit_alu_data_format_state_<is_fp32_dest_acc_en>(math_format, math_format);
                _llk_math_transpose_dest_init_<MATH_TRANSPOSE_FACES, is_fp32_dest_acc_en>();
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_transpose_dest_(DST_INDEX + i);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if constexpr (!unpack_to_dest)
                {
                    _configure_default_alu_data_format_state_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(math_format, math_format);
                    _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(
                        num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(DST_INDEX + i);
                    }
                }
                _configure_mov_ops_explicit_alu_data_format_state_<is_fp32_dest_acc_en>(math_format, math_format);
                _llk_math_transpose_dest_init_<MATH_TRANSPOSE_FACES, is_fp32_dest_acc_en>();
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_math_transpose_dest_(DST_INDEX + i);
                }
                _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
            }
        }
        PROFILER_SYNC();
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
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const int DST_INDEX                 = params.DST_INDEX;
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
            auto cfg                                    = (std::uint32_t volatile*)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
            }
        }

        buffer_descriptor_u bd_val = {0};
        tdma_descriptor_t tdma_desc;

        bd_val.f.l1_addr_16B = L1_ADDRESS(buffer_Res[0]);
        bd_val.f.format      = static_cast<std::uint8_t>(formats.pack_dst);
        bd_val.f.x_dim       = TEST_FACE_C_DIM;
        bd_val.f.y_dim       = TEST_FACE_R_DIM;
        bd_val.f.z_dim       = num_faces;

        tdma_desc.buf_desc        = bd_val;
        tdma_desc.buf_desc_id     = buf_desc_id;
        tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        _llk_pack_init_<is_fp32_dest_acc_en>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_pack);
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
                _llk_pack_(DST_INDEX, 0, ckernel::DEFAULT_TENSOR_SHAPE);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(DST_INDEX, 0, ckernel::DEFAULT_TENSOR_SHAPE);
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
#endif
