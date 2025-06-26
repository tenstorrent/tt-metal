// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;
uint32_t tile_size                = 128;
const int iterations              = 32; // Dependant on size of input tensor (1024 currently). Could be made dynamic once tensor size becomes variable.

volatile uint32_t* const buffer_A_tilized = reinterpret_cast<volatile uint32_t*>(0x17000);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel()
{
    std::uint32_t ct_dim = 1;
    std::uint32_t rt_dim = 1;
    std::uint32_t kt_dim = 1;

    _llk_unpack_AB_matmul_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_B_IN, UNPACK_A_OUT, UNPACK_B_OUT);
    _llk_unpack_AB_matmul_init_<>();
    _llk_unpack_AB_matmul_<>(L1_ADDRESS(buffer_A[0]), L1_ADDRESS(buffer_B[0]), 0, 0, tile_size, tile_size);

    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // Start of second unpack kernel to perform unpack matmul on now tilized input data
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, false>(UNPACK_A_IN, UNPACK_A_OUT, tile_size);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, FACE_R_DIM, 4, UNPACK_A_IN, UNPACK_A_OUT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(buffer_A_tilized), 0, UNPACK_A_IN, UNPACK_A_OUT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_matmul.h"
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
            ckernel::sfpu::_calculate_abs_<APPROX_MODE, iterations>(iterations);
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
            ckernel::sfpu::_calculate_sqrt_<APPROX_MODE, 0, iterations>(iterations);
            break;
        case SfpuType::square:
            ckernel::sfpu::_calculate_square_<APPROX_MODE, iterations>(iterations);
            break;
        default:
            return;
    }
}
} // namespace

void run_kernel()
{
    _llk_math_matmul_init_<MATH_FIDELITY, DstTileFaceLayout::RowMajor>();
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(MATH_FORMAT, MATH_FORMAT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_matmul_<MATH_FIDELITY, DstTileFaceLayout::RowMajor>(0);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Start of second math kernel to perform matmul on now tilized input data
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(MATH_FORMAT);
    // copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(0, 0, 4, MATH_FORMAT);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(0, 0, 4, MATH_FORMAT);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false, false>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, false>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(buffer_A_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    t6_semaphore_post<>(semaphore::PACK_DONE);

    // Start of second pack kernel to perform final pack after executing matmul on tilized data
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(PACK_IN, PACK_OUT, tile_size);
#ifdef PACK_DST_BFP8_B
    constexpr auto PACK_OUT = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Float16_b);
#endif

    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(PACK_OUT);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, false, DstTileFaceLayout::RowMajor, false>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
