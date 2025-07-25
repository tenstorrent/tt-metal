// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

constexpr bool disable_src_zero_flag = true;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel()
{
    using DataFormatUT = std::underlying_type_t<DataFormat>;
    auto to_ufmt       = [](DataFormat fmt) constexpr { return static_cast<DataFormatUT>(fmt); };

    uint8_t UNPACK_FMT;
    if (UNPACK_A_IN == to_ufmt(DataFormat::Float32))
    {
        UNPACK_FMT = to_ufmt(DataFormat::Float32);
    }
    else if (UNPACK_A_IN == to_ufmt(DataFormat::Bfp8_b))
    {
        UNPACK_FMT = to_ufmt(DataFormat::Bfp8_b);
    }
    else
    {
        UNPACK_FMT = to_ufmt(DataFormat::UInt16);
    }

    volatile uint32_t* const buffer_condition = reinterpret_cast<volatile uint32_t*>(0x1a000);
    volatile uint32_t* const buffer_true      = reinterpret_cast<volatile uint32_t*>(0x1b000);
    volatile uint32_t* const buffer_false     = reinterpret_cast<volatile uint32_t*>(0x1c000);

    _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None, disable_src_zero_flag>(UNPACK_FMT, UNPACK_FMT, FACE_R_DIM, 0, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, FACE_R_DIM, 4, UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(buffer_condition), 0, UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(buffer_true), 0, UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(buffer_false), 0, UNPACK_FMT, UNPACK_FMT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "ckernel_sfpu_where.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

using namespace ckernel;

// using namespace sfpu;

void run_kernel()
{
    using DataFormatUT = std::underlying_type_t<DataFormat>;
    auto to_ufmt       = [](DataFormat fmt) constexpr { return static_cast<DataFormatUT>(fmt); };

    uint8_t MATH_FMT;
    if (UNPACK_A_IN == to_ufmt(DataFormat::Float32))
    {
        MATH_FMT = to_ufmt(DataFormat::Float32);
    }
    else if (UNPACK_A_IN == to_ufmt(DataFormat::Bfp8_b))
    {
        MATH_FMT = to_ufmt(DataFormat::Bfp8_b);
    }
    else
    {
        MATH_FMT = to_ufmt(DataFormat::UInt16);
    }

    // copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(0, 0, 4, MATH_FMT);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(0, 0, 4, MATH_FMT);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(MATH_FMT, MATH_FMT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, MATH_FMT, MATH_FMT); // buffer condition
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        1, MATH_FMT, MATH_FMT); // buffer true
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        2, MATH_FMT, MATH_FMT); // buffer false

    // calculation of sfpu operation on dest
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
    _llk_math_eltwise_ternary_sfpu_start_<DstSync::SyncHalf>(0);

    constexpr int iterations = 32;

    ckernel::sfpu::_calculate_where_<false, static_cast<DataFormat>(UNPACK_A_IN), iterations>();

    _llk_math_eltwise_ternary_sfpu_done_();

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
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
    else
    {
        PACK_FMT = to_ufmt(DataFormat::UInt16);
    }

    volatile uint32_t* const buffer_Dest = reinterpret_cast<volatile uint32_t*>(0x1d000);

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(PACK_FMT, PACK_FMT, 16 * 16 * 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(PACK_FMT, PACK_FMT, 16 * 16 * 4);
#endif

    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(PACK_FMT);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, false>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, false, is_fp32_dest_acc_en>(0, L1_ADDRESS(buffer_Dest));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
