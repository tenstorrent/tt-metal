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
#include "tensor_shape.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_reduce.h"
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
    tdma_descriptor_t td_val_A;
    tdma_descriptor_t td_val_B;
    const std::uint32_t buf_desc_id_a = 0;
    const std::uint32_t buf_desc_id_b = 1;
    const auto tensor_shape_A         = tensor_shape_from_params(params);

    {
        ZONE_SCOPED("INIT")
        td_val_A = ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, buf_desc_id_a, formats.unpack_A_dst);
        td_val_B = ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(buffer_B[0]), formats.unpack_B_src, buf_desc_id_b, formats.unpack_B_dst);

        _configure_buf_desc_table_(td_val_A.buf_desc_id, td_val_A.buf_desc);
        _configure_buf_desc_table_(td_val_B.buf_desc_id, td_val_B.buf_desc);
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
        _llk_unpack_reduce_init_<POOL_TYPE, REDUCE_DIM>(buf_desc_id_a, buf_desc_id_b, tensor_shape_A, 1 /*num_tiles_per_unpack*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    // Reduce unpack emits one persistent SrcB scale face,
                    // followed by all SrcA faces for each tile.
                    _perf_unpack_loop_set_valid<false, true>(1);
                    _perf_unpack_loop_set_valid<true, false>(num_faces);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_reduce_(i, 0, tensor_shape_A);
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
    DataFormat src_format         = static_cast<DataFormat>(formats.math);
    DataFormat pack_src_format    = static_cast<DataFormat>(formats.pack_src);
    const bool use_int32_dest_alu = is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32;
    const bool is_int_fpu_en      = use_int32_dest_alu && (REDUCE_DIM == ReduceDim::REDUCE_ROW || REDUCE_DIM == ReduceDim::REDUCE_SCALAR);
    const auto tensor_shape_A     = tensor_shape_from_params(params);

    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE measures pack alone (WH/BH style): skip FPU→PACK dest-dvalid.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        if (use_int32_dest_alu)
        {
            _llk_math_srcAB_hw_configure_<false, false /* fp32 dest */, true /* int32 dest */>(src_format, src_format);
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /* int32 dest */>(src_format, src_format);
        }

        if (is_int_fpu_en)
        {
            // Int Scalar SUM is unsupported, see SFPU reduce.
            if constexpr (!(REDUCE_DIM == ReduceDim::REDUCE_SCALAR && POOL_TYPE == PoolType::SUM))
            {
                _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, MATH_FIDELITY, true /* is_int_fpu_en */>(tensor_shape_A);
            }
        }
        else
        {
            _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, MATH_FIDELITY, false /* is_int_fpu_en */>(tensor_shape_A);
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
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    _perf_math_loop_clear_valid<true, false>(num_faces);
                    _perf_math_loop_clear_valid<false, true>(1);
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if (is_int_fpu_en)
            {
                if constexpr (!(REDUCE_DIM == ReduceDim::REDUCE_SCALAR && POOL_TYPE == PoolType::SUM))
                {
                    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                    {
                        for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                        {
                            _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, true /* is_int_fpu_en */>(i, tensor_shape_A);
                        }
                    }
                }
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, false /* is_int_fpu_en */>(i, tensor_shape_A);
                    }
                }
            }
        }
        else
        {
            if (is_int_fpu_en)
            {
                if constexpr (!(REDUCE_DIM == ReduceDim::REDUCE_SCALAR && POOL_TYPE == PoolType::SUM))
                {
                    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                    {
                        for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                        {
                            _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, true /* is_int_fpu_en */>(i, tensor_shape_A);
                        }
                        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                    }
                }
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, false /* is_int_fpu_en */>(i, tensor_shape_A);
                    }
                    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                }
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    const std::uint32_t buf_desc_id = 8;
    const auto tensor_shape_A       = tensor_shape_from_params(params);

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
            set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        tdma_descriptor_t tdma_desc =
            ckernel::trisc::construct_tdma_desc(tensor_shape_A, L1_ADDRESS(buffer_Res[0]), formats.pack_dst, buf_desc_id, formats.pack_src);

        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);
        _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
        _llk_pack_init_(buf_desc_id, tensor_shape_A, 1 /*num_tiles_per_pack*/);
        _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape_A);
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
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_pack_(i, i, tensor_shape_A);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_pack_(i, i, tensor_shape_A);
                }
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        _llk_pack_reduce_mask_clear_();
        PROFILER_SYNC();
    }
}
#endif
