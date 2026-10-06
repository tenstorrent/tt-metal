// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// max_pool_with_indices: column-wise arg-max of a values tile, carrying an indices tile in lockstep.
//
// Buffer layout (params.TILE_CNT input tiles in buffer_A):
//   buffer_A[i] -> Dest[i] through Unpack-to-Dest. SRC0_TILE_IDX is the values tile, SRC1_TILE_IDX the
//   indices tile; with MAX_POOL_ACCUMULATE the running max lives in Dest[SRC0 + 1] / Dest[SRC1 + 1].
//   PACK reads Dest[DST_TILE_IDX] (one tile), so the test picks which of those tiles to check.
// The kernel is called the way the compute API calls it: once per tile pair, under VectorMode::None.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "cfg_defines.h"
#include "llk_bfd_alloc.h"
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
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
#endif

    {
        ZONE_SCOPED("INIT")
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, TILE_CNT);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_unpack_dest_dvalid_section_done_<dest_sync>();
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_max_pool_indices.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "max_pool_with_indices test only implements L1_TO_L1");
static_assert(unpack_to_dest, "max_pool_with_indices test stages its operands through Unpack-to-Dest");

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT       = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t SRC0_TILE_IDX  = params.SRC0_TILE_IDX;
    const std::uint32_t SRC1_TILE_IDX  = params.SRC1_TILE_IDX;
    const std::uint32_t MAX_POOL_CHUNK = params.MAX_POOL_CHUNK;
#endif
    LLK_ASSERT(SRC0_TILE_IDX < TILE_CNT && SRC1_TILE_IDX < TILE_CNT, "max_pool operands must be staged tiles");
    LLK_ASSERT(!MAX_POOL_ACCUMULATE || (SRC0_TILE_IDX + 1 < TILE_CNT && SRC1_TILE_IDX + 1 < TILE_CNT), "accumulate needs the running-max tiles staged");
    const DataFormat math_format = static_cast<DataFormat>(formats.math);

    {
        ZONE_SCOPED("INIT")
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(math_format, math_format);
        _llk_math_eltwise_sfpu_init_();
        // After _llk_math_eltwise_sfpu_init_(), which resets the SFPU Control Register init sets.
        init_max_pool_with_indices<APPROX_MODE, MAX_POOL_LAYOUT>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            SFPU_BINARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_max_pool_with_indices,
                (APPROX_MODE, is_fp32_dest_acc_en, MAX_POOL_NUM_ROWS, SFPU_ITERATIONS, MAX_POOL_LAYOUT, MAX_POOL_ACCUMULATE),
                SRC0_TILE_IDX,
                SRC1_TILE_IDX,
                0 /* unused out tile */,
                VectorMode::None,
                MAX_POOL_CHUNK);
            _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
        }
        // Drain SFPU and MOP queues before PACK takes over.
        wait_sfpu_idle();
        wait_mop_idle();
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
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
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const std::uint32_t DST_TILE_IDX = params.DST_TILE_IDX;
    const Operand& buffer_Res        = params.buffer_Res;
#endif

    {
        ZONE_SCOPED("INIT")
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, 1 /*one checked tile*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            _llk_pack_(DST_TILE_IDX, 0 /*tile index*/, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
