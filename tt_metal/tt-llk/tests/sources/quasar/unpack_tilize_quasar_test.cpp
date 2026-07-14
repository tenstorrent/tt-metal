// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
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
    const Operand& buffer_A             = params.buffer_A;
    const Operand& buffer_B             = params.buffer_B;
#endif
    tdma_descriptor_t td_val;
    const std::uint32_t buf_desc_id = 0;

    {
        ZONE_SCOPED("INIT")
        constexpr auto dest_producer = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_producer, dest_dvalid_client::PACK});

        if constexpr (unpack_to_dest && is_fp32_dest_acc_en)
        {
            const bool int32_dest = static_cast<DataFormat>(formats.unpack_A_src) == DataFormat::Int32;
            if (int32_dest)
            {
                _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false, true>();
            }
            else
            {
                _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, true, false>();
            }
        }
        else if constexpr (unpack_to_dest)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false, false>();
        }

        buffer_descriptor_u bd_val = {0};

        unsigned l1_addr_16B;
        if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_A || UNPACKER_ENGINE_SEL == p_unpacr::UNP_DEST)
        {
            l1_addr_16B = buffer_A[0] / 16;
        }
        else if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
        {
            l1_addr_16B = buffer_B[0] / 16;
        }

        bd_val.f.l1_addr_16B = l1_addr_16B;
        bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
        bd_val.f.x_dim       = TEST_FACE_C_DIM;
        bd_val.f.y_dim       = TEST_FACE_R_DIM;
        bd_val.f.z_dim       = num_faces;

        td_val.buf_desc        = bd_val;
        td_val.buf_desc_id     = buf_desc_id;
        td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);

        constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;

        _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
        }
        else
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
        }

        if constexpr (unpack_to_dest)
        {
            _llk_unpack_tilize_block_init_<FULL_CT_DIM, BLOCK_CT_DIM>(buf_desc_id, tensor_shape);
        }
        else
        {
            _llk_unpack_tilize_init_<UNPACKER_ENGINE_SEL, is_fp32_dest_acc_en>(buf_desc_id, FULL_CT_DIM, BLOCK_CT_DIM, tensor_shape);
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        constexpr ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;
        std::uint32_t y_stride_external             = FULL_CT_DIM * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;

        // Quasar fused tilize emits 1 SrcA dvalid per `_llk_unpack_tilize_` call (BLOCK_RT_DIM
        // calls per outer loop). With is_fp32_dest_acc_en it also pulses SrcB via UNPACR_NOP
        // because FP32 datacopy uses ELWADD on SrcA+SrcB.
        const std::uint32_t total_tilize_dvalids = LOOP_FACTOR * BLOCK_RT_DIM;

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                _perf_unpack_loop_set_valid<true, true>(total_tilize_dvalids);
            }
            else if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
            {
                _perf_unpack_loop_set_valid<true, false>(total_tilize_dvalids);
            }
            else
            {
                _perf_unpack_loop_set_valid<false, true>(total_tilize_dvalids);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if constexpr (unpack_to_dest)
                {
                    for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
                    {
                        _llk_unpack_tilize_block_(y * y_stride_external, y * BLOCK_CT_DIM);
                    }
                    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                }
                else
                {
                    for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
                    {
                        _llk_unpack_tilize_<UNPACKER_ENGINE_SEL>(y * y_stride_external);
                    }
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
            // PACK_ISOLATE / L1_CONGESTION: skip FPU→PACK dest-dvalid (math is src-clear only
            // in congestion; pulsing never happens — matches WH/BH unpack_tilize_perf).
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
                // Match tilize producer: SrcA only, or SrcA+SrcB when FP32 dest uses ELWADD.
                const std::uint32_t total_tilize_dvalids = LOOP_FACTOR * TILE_CNT;
                if constexpr (is_fp32_dest_acc_en)
                {
                    _perf_math_loop_clear_valid<true, true>(total_tilize_dvalids);
                }
                else if constexpr (DATA_COPY_TYPE == DataCopyType::A2D)
                {
                    _perf_math_loop_clear_valid<true, false>(total_tilize_dvalids);
                }
                else
                {
                    _perf_math_loop_clear_valid<false, true>(total_tilize_dvalids);
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
    const std::uint32_t TEST_FACE_C_DIM = params.TEST_FACE_C_DIM;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_Res           = params.buffer_Res;
#endif
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE / L1_CONGESTION: no math↔pack dest-dvalid handshake (WH/BH tilize perf).
        // Math only clears src dvalids in L1_CONGESTION, so waiting on section_done times out
        // (~28k/iter) and leaves dest-dvalid CFG wedged for later L1_TO_L1 on the same simulator.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            auto cfg = (std::uint32_t volatile *)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else
        {
            constexpr auto dest_producer = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
            set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_producer, dest_dvalid_client::PACK});
        }

        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B       = buffer_Res[0] / 16;
        bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
        bd_val.f.x_dim             = TEST_FACE_C_DIM;
        bd_val.f.y_dim             = TEST_FACE_R_DIM;
        bd_val.f.z_dim             = num_faces;

        tdma_descriptor_t tdma_desc;
        tdma_desc.buf_desc        = bd_val;
        tdma_desc.buf_desc_id     = buf_desc_id;
        tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_pack);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: WH/BH L1_CONGESTION packs without math handshake
            // (math is only a src-clear mock). Per-iter section_done was causing ~28k stalls and
            // poisoning subsequent L1_TO_L1 on the persistent Quasar simulator.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(0, 0, ckernel::DEFAULT_TENSOR_SHAPE);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_(0, 0, ckernel::DEFAULT_TENSOR_SHAPE);
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}
#endif
