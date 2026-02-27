// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "data_format_inference.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
std::uint32_t tile_size                = 128;
const int iterations                   = 32; // Dependent on size of input tensor (1024 currently). Could be made dynamic once tensor size becomes variable.

constexpr std::uint32_t buffer_A_tilized = 0x66000; // L1 address of buffer, placed so both coverage and non-coverage elfs don't run it over

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig* formats_array = params->formats;
#endif
    int run = 0; // first L1-to-L1 run, we access the first set of formats_array in our array
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_B_src,
        formats_array[run].unpack_A_dst,
        formats_array[run].unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        4 /* num_faces */,
        4 /* num_faces */);
    _llk_unpack_AB_matmul_init_<>();
    _llk_unpack_AB_matmul_<>(L1_ADDRESS(params->buffer_A[0]), L1_ADDRESS(params->buffer_B[0]), 0, 0, params->TILE_SIZE_UNPACK_A, params->TILE_SIZE_UNPACK_B);

    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // Start of second unpack kernel to perform unpack matmul on now tilized input data
    run = 1; // second L1-to-L1 run, we access the second set of formats_array in our array
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, false>(formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst, TILE_SIZE_PACK);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, 4, formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(buffer_A_tilized), formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_matmul.h"
#include "params.h"
#include "sfpu_operations.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig* formats_array = params->formats;
#endif
    int run = 0; // first L1-to-L1 run, we access the first set of formats_array in our array
    _llk_math_matmul_init_<MATH_FIDELITY>();
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats_array[run].math, formats_array[run].math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_matmul_<MATH_FIDELITY>(0);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Start of second math kernel to perform matmul on now tilized input data
    run = 1; // second L1-to-L1 run, we access the second set of formats_array in our array
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(formats_array[run].math);
    // copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(4, formats_array[run].math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(4, formats_array[run].math);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats_array[run].math, formats_array[run].math);

    // calculation of sfpu operation on dest
    _llk_math_eltwise_unary_sfpu_init_<SFPU_UNARY_OPERATION>();
    _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(0);
    // calling sfpu function from ckernel
    // this part is where parametrization of operation takes part
    test_utils::call_sfpu_operation<APPROX_MODE, is_fp32_dest_acc_en, 32>(SFPU_UNARY_OPERATION);

    _llk_math_eltwise_unary_sfpu_done_();
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig* formats_array = params->formats;
#endif
    int run = 0; // first L1-to-L1 run, we access the first set of formats_array in our array
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats_array[run].pack_src, formats_array[run].pack_dst, 16 * 16 * 4);
    _llk_pack_init_<false, false, false>(formats_array[run].pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats_array[run].pack_src, formats_array[run].pack_dst, 16 * 16 * 4);
    _llk_pack_init_<false, false>(formats_array[run].pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(buffer_A_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    t6_semaphore_post<>(semaphore::PACK_DONE);

    // Start of second pack kernel to perform final pack after executing matmul on tilized data
    run = 1; // second L1-to-L1 run, we access the second set of formats_array in our array
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(formats_array[run].pack_src, formats_array[run].pack_dst, params->TILE_SIZE_PACK);

    _llk_pack_init_<false, false>(formats_array[run].pack_dst);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, false, false>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(params->buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
