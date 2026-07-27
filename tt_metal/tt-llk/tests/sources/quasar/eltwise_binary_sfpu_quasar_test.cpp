// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Buffer layout (params.TILE_CNT input tiles in buffer_A):
//   buffer_A[i] -> Dest[i]; SFPU writes the result to Dest[DST_TILE_IDX];
//   PACK reads Dest[DST_TILE_IDX] (one tile).
// For max/min the convention is in0=Dest[0], in1=Dest[1], out=Dest[2].
//
// Two execution paths, selected by `unpack_to_dest`:
//   * unpack_to_dest=true  — UNPACK writes Dest directly (32-bit Dest formats,
//                            and all integer/float mul/div variants).
//   * unpack_to_dest=false — UNPACK -> SrcA -> FPU datacopy -> Dest (required for
//                            non-32-bit / MX block formats used by max/min).

#include <cstdint>

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
    const std::uint32_t buf_desc_id = 0;
    const std::uint32_t num_tiles   = params.TILE_CNT; // input tiles staged in buffer_A
    const std::uint32_t loop_factor = params.LOOP_FACTOR;

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
            }
        }

        if constexpr (unpack_to_dest)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
        }

        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
        bd_val.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
        bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
        bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
        bd_val.f.z_dim             = params.num_faces;

        tdma_descriptor_t td_val;
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

        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_loop_set_valid<true, false>(loop_factor * num_tiles);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0, ckernel::DEFAULT_TENSOR_SHAPE);
                if constexpr (unpack_to_dest && PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"
#include "sfpu_operations_quasar.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Binary SFPU stages both operands from buffer_A into Dest. TILE_CNT is the
    // number of staged tiles and is fixed by the test layout, not free-running:
    // 2 when operands occupy Dest tiles 0/1 (result in-place or a separately
    // written out tile, e.g. max/min's tile 2), or 4 for the result-aliasing
    // variant that places operands at Dest tiles 2/3. Guard against any other count.
    LLK_ASSERT(params.TILE_CNT == 2 || params.TILE_CNT == 4, "Binary SFPU expects TILE_CNT of 2 or 4");
    const std::uint32_t loop_factor = params.LOOP_FACTOR;
    const std::uint32_t num_tiles   = params.TILE_CNT;

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
                set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
            }
        }
        const DataFormat math_format = static_cast<DataFormat>(formats.math);
        if constexpr (test_utils::quasar_binary_op_is_max_min(SFPU_BINARY_OP))
        {
            const DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
            if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Float32)
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, true, false>(math_format, math_format);
            }
            else if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false, true>(math_format, math_format);
            }
            else
            {
                _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false, false>(math_format, math_format);
            }
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false>(math_format, math_format);
        }

        if constexpr (!unpack_to_dest)
        {
            const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
            _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);
        }
        _llk_math_eltwise_sfpu_init_();
        test_utils::init_binary_sfpu_operation_quasar<SFPU_BINARY_OP>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_math_loop_clear_valid<true, false>(loop_factor * num_tiles);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            const DataFormat math_format = static_cast<DataFormat>(formats.math);
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                if constexpr (!unpack_to_dest)
                {
                    for (std::uint32_t i = 0; i < num_tiles; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(params.DST_INDEX + i);
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                    }
                }
                test_utils::call_binary_sfpu_operation_quasar<SFPU_BINARY_OP, dest_sync, is_fp32_dest_acc_en>(
                    params.SRC0_TILE_IDX, params.SRC1_TILE_IDX, params.DST_TILE_IDX, math_format);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
                }
            }
            wait_sfpu_idle();
            wait_fpu_idle();
            wait_mop_idle();
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = 1; // only the SFPU result tile is packed
    const std::uint32_t loop_factor        = params.LOOP_FACTOR;

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            auto cfg                                    = (volatile std::uint32_t*)TENSIX_CFG_BASE;
            cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
            }
            else
            {
                set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
            }
        }

        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_Res[0]);
        bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
        bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
        bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
        bd_val.f.z_dim             = params.num_faces;

        tdma_descriptor_t tdma_desc;
        tdma_desc.buf_desc        = bd_val;
        tdma_desc.buf_desc_id     = buf_desc_id;
        tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
        _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(tdma_desc, ckernel::ReluConfig::none());
        _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles_per_pack);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                _llk_pack_(params.DST_TILE_IDX, 0, ckernel::DEFAULT_TENSOR_SHAPE);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif // LLK_TRISC_PACK
