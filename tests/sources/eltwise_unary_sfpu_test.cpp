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
    _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(formats.unpack_src, formats.unpack_dst, FACE_R_DIM, 0, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, 4, formats.unpack_src, formats.unpack_dst);

    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(buffer_A[i]), 0, formats.unpack_src, formats.unpack_dst);
    }
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

const int iterations = 32;

namespace
{
void call_sfpu_operation(SfpuType operation, uint32_t math_format)
{
    switch (operation)
    {
        case SfpuType::abs:
            ckernel::sfpu::_calculate_abs_<APPROX_MODE, iterations>(iterations);
            break;
        case SfpuType::atanh:
            ckernel::sfpu::_init_atanh_<APPROX_MODE>();
            ckernel::sfpu::_calculate_atanh_<APPROX_MODE, is_fp32_dest_acc_en, iterations>();
            break;
        case SfpuType::asinh:
            ckernel::sfpu::_init_inverse_hyperbolic_<APPROX_MODE>();
            ckernel::sfpu::_calculate_asinh_<APPROX_MODE, iterations>();
            break;
        case SfpuType::acosh:
            ckernel::sfpu::_init_inverse_hyperbolic_<APPROX_MODE>();
            ckernel::sfpu::_calculate_acosh_<APPROX_MODE, iterations>();
            break;
        case SfpuType::cosine:
            ckernel::sfpu::_calculate_cosine_<APPROX_MODE, iterations>(iterations);
            break;
        case SfpuType::log:
            ckernel::sfpu::_init_log_<APPROX_MODE>();
            ckernel::sfpu::_calculate_log_<APPROX_MODE, false, iterations>(iterations, 0);
            break;
        case SfpuType::reciprocal:
            ckernel::sfpu::_init_reciprocal_<APPROX_MODE>();
            ckernel::sfpu::_calculate_reciprocal_<APPROX_MODE, iterations, is_fp32_dest_acc_en>(iterations);
            break;
        case SfpuType::sine:
            ckernel::sfpu::_calculate_sine_<APPROX_MODE, iterations>(iterations);
            break;
        case SfpuType::sqrt:
            ckernel::sfpu::_init_sqrt_<APPROX_MODE>();
            ckernel::sfpu::_calculate_sqrt_<APPROX_MODE, iterations, 2>(iterations);
            break;
        case SfpuType::square:
            ckernel::sfpu::_calculate_square_<APPROX_MODE, iterations>(iterations);
            break;
        case SfpuType::celu:
            ckernel::sfpu::_calculate_activation_<APPROX_MODE, ActivationType::Celu, iterations>(10, 1 / 10);
            break;
        case SfpuType::silu:
            ckernel::sfpu::_calculate_silu_<APPROX_MODE, iterations>();
            break;
        case SfpuType::gelu:
            ckernel::sfpu::_init_gelu_<APPROX_MODE>();
            ckernel::sfpu::_calculate_gelu_<APPROX_MODE, iterations>();
            break;
        case SfpuType::neg:
            if (math_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32))
            {
                ckernel::sfpu::_calculate_negative_int_<APPROX_MODE, iterations>();
            }
            else
            {
                ckernel::sfpu::_calculate_negative_<APPROX_MODE, iterations>();
            }
            break;
        case SfpuType::fill:
            if (math_format == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Int32))
            {
                ckernel::sfpu::_calculate_fill_int_<APPROX_MODE, iterations>(5);
            }
            else
            {
                ckernel::sfpu::_calculate_fill_<APPROX_MODE, iterations>(5.0f);
            }
            break;
        case SfpuType::elu:
            ckernel::sfpu::_init_elu_<APPROX_MODE>();
            ckernel::sfpu::_calculate_elu_<APPROX_MODE, iterations>(1);
            break;
        case SfpuType::exponential:
            ckernel::sfpu::_init_exponential_<APPROX_MODE, false /*fast_mode*/, 0x3F800000 /* exp_base_scale_factor */>();
            ckernel::sfpu::_calculate_exponential_<APPROX_MODE, false /* scale_en */, iterations, false /* fast_approx */, false /* skip_positive_check */>(
                p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
            break;
        case SfpuType::exp2:
            ckernel::sfpu::_init_exp2_<APPROX_MODE>();
            ckernel::sfpu::_calculate_exp2_<APPROX_MODE, iterations>();
            break;
        case SfpuType::hardsigmoid:
            ckernel::sfpu::_init_hardsigmoid_<APPROX_MODE>();
            ckernel::sfpu::_calculate_activation_<APPROX_MODE, ckernel::ActivationType::Hardsigmoid, iterations>();
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
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(0, 0, 4, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(0, 0, 4, formats.math);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(formats.math, formats.math);

    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats.math, formats.math);

        // calculation of sfpu operation on dest
        _llk_math_eltwise_unary_sfpu_init_<SFPU_UNARY_OPERATION>();
        _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(i);
        // calling sfpu function from ckernel
        // this part is where parametrization of operation takes part
        call_sfpu_operation(SFPU_UNARY_OPERATION, formats.math);

        _llk_math_eltwise_unary_sfpu_done_();
    }

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#endif

    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(formats.pack_dst);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, false, DstTileFaceLayout::RowMajor, false>();
#endif

    _llk_packer_wait_for_math_done_();
    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
