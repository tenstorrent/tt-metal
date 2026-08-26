// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

using namespace ckernel;
#include "params.h" // POOL_TYPE, REDUCE_DIM, BLOCK_CT_DIM, BLOCK_RT_DIM, IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en

// SFPU reduce on Quasar: collapse a Dest block along one axis with SUM, AVG or MAX.
//
// Three programs in one file, one per thread:
//   T0 unpack: load every tile of buffer_A from L1 into Dest. One _llk_unpack_unary_operand_
//              call covers the whole TILE_CNT bank - no FPU datacopy is involved.
//   T1 math:   run calculate_reduce over Dest. REDUCE_COL is called once per tile, since a
//              column sits inside one tile. REDUCE_ROW is called once for the whole block,
//              since a row spans every tile of its tile row.
//   T2 pack:   write the reduced tiles back out to buffer_Res in L1.
//
// The reduce only writes the axis it collapses onto - row 0 for REDUCE_COL, column 0 for
// REDUCE_ROW - and leaves the rest of each tile holding leftovers. The Python side compares
// just that axis.

#ifdef LLK_TRISC_UNPACK

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

    // Dest handoff chain: T0 produces, T1 (SFPU) and T2 (PACK) consume. No FPU stage - the
    // reduce is SFPU-only.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    // Int32 needs Dest in int32 mode; floats follow is_fp32_dest_acc_en.
    constexpr bool is_int_reduce = static_cast<DataFormat>(MATH_FORMAT) == DataFormat::Int32;
    if constexpr (is_int_reduce)
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>();
    }
    else
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>();
    }

    // Where to read from: buffer_A in L1, holding unpack_A_src, with the harness's face geometry.
    // unpack_A_dst below is what it converts to on the way into Dest.
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);

    // Configure the unpacker, set it up for TILE_CNT tiles, then pull the whole bank into Dest.
    _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, params.TILE_CNT);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);

    // Hand Dest to the SFPU.
    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_sfpu/ckernel_sfpu_reduce.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// The reduce bakes its SFPLOAD/SFPSTORE mode into the instruction words, so the format has to be
// known at compile time. TestConfig's compile_time_formats is what makes MATH_FORMAT a constant
// this kernel can pass straight to calculate_reduce as a template argument.
constexpr DataFormat REDUCE_MATH_FORMAT = static_cast<DataFormat>(MATH_FORMAT);

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Math is the SFPU link in the Dest handoff chain.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    const DataFormat math_format = static_cast<DataFormat>(formats.math);
    constexpr bool is_int_reduce = (REDUCE_MATH_FORMAT == DataFormat::Int32);

    if constexpr (is_int_reduce)
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
    }
    else
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(math_format, math_format);
    }

    _llk_math_eltwise_sfpu_init_();
    ckernel::sfpu::init_reduce<POOL_TYPE, REDUCE_MATH_FORMAT, is_fp32_dest_acc_en>();

    if constexpr (REDUCE_DIM == ckernel::ReduceDim::REDUCE_COL)
    {
        // A column lives inside one tile, so each tile reduces onto its own row 0.
        for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
        {
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_reduce,
                (POOL_TYPE, REDUCE_DIM, REDUCE_MATH_FORMAT, is_fp32_dest_acc_en),
                params.DST_INDEX + tile,
                VectorMode::RC_custom,
                1 /*block_ct_dim: unused by the column reduce*/,
                1 /*block_rt_dim: unused by the column reduce*/);
        }
    }
    else
    {
        // A row spans the whole tile row, so one call handles the entire block, walking Dest
        // itself from the tile-0 base.
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_reduce,
            (POOL_TYPE, REDUCE_DIM, REDUCE_MATH_FORMAT, is_fp32_dest_acc_en),
            params.DST_INDEX,
            VectorMode::RC_custom,
            BLOCK_CT_DIM,
            BLOCK_RT_DIM);
    }

    // Hand Dest to PACK.
    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    // Let every queue drain before this thread returns.
    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

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

    // PACK is the last link in the Dest handoff chain.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    // Where to write to: buffer_Res in L1, holding pack_dst, with the harness's face geometry.
    // pack_src below is the Dest-side format the packer reads.
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);

    // Configure pack engine 0, set it up for TILE_CNT tiles, write the bank out, release Dest.
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, params.TILE_CNT);
    _llk_pack_(params.DST_INDEX, 0 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
