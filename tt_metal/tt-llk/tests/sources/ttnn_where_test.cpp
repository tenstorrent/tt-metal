// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    using DataFormatUT = std::underlying_type_t<DataFormat>;
    auto to_ufmt       = [](DataFormat fmt) constexpr { return static_cast<DataFormatUT>(fmt); };

    std::uint8_t UNPACK_FMT;
    if (UNPACK_A_IN == to_ufmt(DataFormat::Float32))
    {
        UNPACK_FMT = to_ufmt(DataFormat::Float32);
    }
    else if (UNPACK_A_IN == to_ufmt(DataFormat::Bfp8_b))
    {
        UNPACK_FMT = to_ufmt(DataFormat::Bfp8_b);
    }
    else if (UNPACK_A_IN == to_ufmt(DataFormat::Int32))
    {
        UNPACK_FMT = to_ufmt(DataFormat::Int32);
    }
    else
    {
        UNPACK_FMT = to_ufmt(DataFormat::UInt16);
    }

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[0]), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_B[0]), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_C[0]), UNPACK_FMT, UNPACK_FMT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "ckernel_sfpu_where.h"
#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

// llk_math_eltwise_ternary_sfpu_macros.h instantiates asserts with these; must match params.h / JIT.
static constexpr ckernel::DstSync DST_SYNC_MODE = ckernel::DstSync::SyncHalf;
static constexpr bool DST_ACCUM_MODE            = is_fp32_dest_acc_en;

#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"

// using namespace sfpu;

void run_kernel(RUNTIME_PARAMETERS)
{
    using DataFormatUT = std::underlying_type_t<DataFormat>;
    auto to_ufmt       = [](DataFormat fmt) constexpr { return static_cast<DataFormatUT>(fmt); };

    std::uint8_t MATH_FMT;
    if (UNPACK_A_IN == to_ufmt(DataFormat::Float32))
    {
        MATH_FMT = to_ufmt(DataFormat::Float32);
    }
    else if (UNPACK_A_IN == to_ufmt(DataFormat::Bfp8_b))
    {
        MATH_FMT = to_ufmt(DataFormat::Bfp8_b);
    }
    else if (UNPACK_A_IN == to_ufmt(DataFormat::Int32))
    {
        MATH_FMT = to_ufmt(DataFormat::Int32);
    }
    else
    {
        MATH_FMT = to_ufmt(DataFormat::UInt16);
    }

    // copy srca to dest
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        4 /* num_faces */, MATH_FMT);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, MATH_FMT, MATH_FMT); // buffer condition
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        1, MATH_FMT, MATH_FMT); // buffer true
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        2, MATH_FMT, MATH_FMT); // buffer false

    // calculation of sfpu operation on dest
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
    ckernel::sfpu::_init_where_<false>();

    // One SFPU replay advances one row of 32 lanes; 8 rows per face (matches llk_math_eltwise_ternary_sfpu_where).
    constexpr int k_where_iterations = 8;
    SFPU_TERNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_where_,
        (false /*APPROXIMATE*/, static_cast<DataFormat>(UNPACK_A_IN), k_where_iterations),
        0 /*DST_IN0*/,
        1 /*DST_IN1*/,
        2 /*DST_IN2*/,
        0 /*DST_OUT*/,
        VectorMode::RC);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    using DataFormatUT = std::underlying_type_t<DataFormat>;
    auto to_ufmt       = [](DataFormat fmt) constexpr { return static_cast<DataFormatUT>(fmt); };

    std::uint8_t PACK_FMT;
    if (UNPACK_A_IN == to_ufmt(DataFormat::Float32))
    {
        PACK_FMT = to_ufmt(DataFormat::Float32);
    }
    else if (UNPACK_A_IN == to_ufmt(DataFormat::Bfp8_b))
    {
        PACK_FMT = to_ufmt(DataFormat::Bfp8_b);
    }
    else if (UNPACK_A_IN == to_ufmt(DataFormat::Int32))
    {
        PACK_FMT = to_ufmt(DataFormat::Int32);
    }
    else
    {
        PACK_FMT = to_ufmt(DataFormat::UInt16);
    }

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_FMT, PACK_FMT, 16 * 16 * 4 /* tile_size */);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_FMT);

    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
