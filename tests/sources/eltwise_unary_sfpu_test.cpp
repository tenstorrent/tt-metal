// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel()
{
    volatile uint32_t* const buffer_A = reinterpret_cast<volatile uint32_t*>(0x1a000);

    _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_A_OUT, FACE_R_DIM, 0, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, FACE_R_DIM, 4, UNPACK_A_IN, UNPACK_A_OUT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(buffer_A), 0, UNPACK_A_IN, UNPACK_A_OUT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

namespace
{
void call_sfpu_operation(SfpuType operation)
{
    switch (operation)
    {
        case SfpuType::abs:
            ckernel::sfpu::_calculate_abs_<APPROX_MODE, 32>(32);
            break;
        case SfpuType::cosine:
            ckernel::sfpu::_calculate_cosine_<APPROX_MODE, 32>(32);
            break;
        case SfpuType::log:
            ckernel::sfpu::_init_log_<APPROX_MODE>();
            ckernel::sfpu::_calculate_log_<APPROX_MODE, false, 32>(32, 0);
            break;
        case SfpuType::reciprocal:
            ckernel::sfpu::_init_reciprocal_<APPROX_MODE>();
            ckernel::sfpu::_calculate_reciprocal_<APPROX_MODE, 32, is_fp32_dest_acc_en>(32);
            break;
        case SfpuType::sine:
            ckernel::sfpu::_calculate_sine_<APPROX_MODE, 32>(32);
            break;
        case SfpuType::sqrt:
            ckernel::sfpu::_init_sqrt_<APPROX_MODE>();
            ckernel::sfpu::_calculate_sqrt_<APPROX_MODE, 0, 32>(32);
            break;
        case SfpuType::square:
            ckernel::sfpu::_calculate_square_<APPROX_MODE, 32>(32);
            break;
        case SfpuType::celu:
            ckernel::sfpu::_calculate_activation_<APPROX_MODE, ActivationType::Celu, 10>(10, 1 / 10);
            break;
        default:
            return;
    }
}
} // namespace

void run_kernel()
{
// copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(0, 0, 4, MATH_FORMAT);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(0, 0, 4, MATH_FORMAT);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(MATH_FORMAT, MATH_FORMAT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, MATH_FORMAT, MATH_FORMAT);

    // calculation of sfpu operation on dest
    _llk_math_eltwise_unary_sfpu_init_<SFPU_OPERATION>();
    _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(0);
    // calling sfpu function from ckernel
    // this part is where parametrization of operation takes part
    call_sfpu_operation(SFPU_OPERATION);

    _llk_math_eltwise_unary_sfpu_done_();
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    // If data foramt is Bfp8 it is calculated correctly in Dest but packer cannot pack just that one face
    // TODO: make it so It can
    // So for now It is packed as Float16_b

#ifdef PACK_DST_BFP8_B
    constexpr auto PACK_OUT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b);
#endif

    volatile uint32_t* const buffer_Dest = reinterpret_cast<volatile uint32_t*>(0x1c000);

    std::fill(buffer_Dest, buffer_Dest + 16 * 16 * 4, 0xdeadbeef);

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(PACK_IN, PACK_OUT, 16 * 16 * 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(PACK_IN, PACK_OUT, 16 * 16 * 4);
#endif

    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(PACK_OUT);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, false, DstTileFaceLayout::RowMajor, false>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(buffer_Dest));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
