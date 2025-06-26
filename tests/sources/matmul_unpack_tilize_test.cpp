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

// Remove later
volatile uint32_t* const buffer_A_tilized = reinterpret_cast<volatile uint32_t*>(0x1e000);
volatile uint32_t* const buffer_B_tilized = reinterpret_cast<volatile uint32_t*>(0x1f000);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "params.h"

void run_kernel()
{
    _llk_unpack_tilize_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_A_OUT, FACE_R_DIM, 0, 4);

    _llk_unpack_tilize_init_(UNPACK_A_IN, UNPACK_A_OUT, 1, FACE_R_DIM, false);
    _llk_unpack_tilize_(L1_ADDRESS(buffer_A[0]), 0, UNPACK_A_IN, 1, FACE_R_DIM, 4, false);

    _llk_unpack_tilize_init_(UNPACK_B_IN, UNPACK_B_OUT, 1, FACE_R_DIM, false);
    _llk_unpack_tilize_(L1_ADDRESS(buffer_B[0]), 0, UNPACK_B_IN, 1, FACE_R_DIM, 4, false);

    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(
        semaphore::PACK_DONE); // Unpacker waits on signal when packer will increment semaphore to 1 (waits while semaphore == 0), utilizing SEMWAIT.
    t6_semaphore_get<>(semaphore::PACK_DONE); // It will acquire the semaphore t6_semaphore_get (decrementing the semaphore back to 0) signalling it has begun

    // Start of second unpack kernel to perform unpack matmul on now tilized input data
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, false>(
        UNPACK_A_IN, UNPACK_A_OUT, tile_size); // have to reconfigure unpack kernel data formats if they change in this run
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, false>(UNPACK_B_IN, UNPACK_B_OUT, tile_size);
    _llk_unpack_tilize_uninit_(UNPACK_A_OUT);
    _llk_unpack_AB_matmul_init_<>();
    _llk_unpack_AB_matmul_<>(L1_ADDRESS(buffer_A_tilized), L1_ADDRESS(buffer_B_tilized), 0, 0, tile_size, tile_size);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_matmul.h"
#include "params.h"

using namespace ckernel;

void run_kernel()
{
    const bool is_int_fpu_en                = false;
    const std::uint32_t operand_A_dst_index = 1;
    const std::uint32_t operand_B_dst_index = 2;
    const std::uint32_t res_dst_index       = 0;
    const bool TILIZE                       = true;

// copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, TILIZE, is_int_fpu_en>(0, 0, 4, MATH_FORMAT);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(0, 0, 4, MATH_FORMAT);
#endif

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(MATH_FORMAT, MATH_FORMAT);

    // copy tilized inputs to dest indexes 0 and 1
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
        operand_A_dst_index, MATH_FORMAT, MATH_FORMAT);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
        operand_B_dst_index, MATH_FORMAT, MATH_FORMAT);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Start of second math kernel to perform matmul on now tilized input data
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(MATH_FORMAT); // have to reconfigure math kernel data formats if they change in this run
    _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, false>(MATH_FORMAT);
    _llk_math_matmul_init_<MATH_FIDELITY, DstTileFaceLayout::RowMajor>();
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_matmul_<MATH_FIDELITY, DstTileFaceLayout::RowMajor>(0);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    const std::uint32_t ct_dim              = 1;
    const std::uint32_t operand_A_dst_index = 1;
    const std::uint32_t operand_B_dst_index = 2;
    const std::uint32_t res_dst_index       = 0;
    const bool UNTILIZE                     = false;
    const bool TILIZE                       = true;

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, TILIZE>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<UNTILIZE, false, DstTileFaceLayout::RowMajor, false, TILIZE>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<UNTILIZE, false, DstTileFaceLayout::RowMajor, false>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, UNTILIZE>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>(operand_A_dst_index, L1_ADDRESS(buffer_A_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>(operand_B_dst_index, L1_ADDRESS(buffer_B_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>(); // Packer will execute _llk_pack_dest_section_done_ function which ensures the write
                                                                            // to L1 is fully is complete.
    t6_semaphore_post<>(semaphore::PACK_DONE); // The packer signals to the unpacker that it has finished writing to L1 by posting (incrementing) the semaphore.
                                               // Now unpacker's wait condition is satisfied, allowing it to begin processing data from L1.

    // Start of second pack kernel to perform final pack after executing matmul on tilized data
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        PACK_IN,
        PACK_OUT,
        tile_size); // need to reconfigure data formats for next pack, also calls set_packer_strides to readjust strides after pack tilizing

#ifdef ARCH_BLACKHOLE
    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false, false>(PACK_OUT);
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(res_dst_index, L1_ADDRESS(buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
