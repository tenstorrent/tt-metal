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

// TODO: CLEANUP

volatile uint32_t* const buffer_A_tilized = reinterpret_cast<volatile uint32_t*>(0x16000);
volatile uint32_t* const buffer_B_tilized = reinterpret_cast<volatile uint32_t*>(0x17000);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
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

    /*
    In this test we fuse two LLK pipeline runs, one is to unpack untilized buffers/operands from L1 (39-45) and pack them in tilized format(130-145).
    The next run unpacks these two tilized operands, performs a math compute and pack them out in utilized format.
    Since we have set all three TRISCs to run at the same time, fusing these two runs will cause a race condition where unpacker will immediately read from
    L1 before the packer has completed writing to L1. To prevent the unpacker from prematurely reading from L1 before packer has completed write
    the unpacker needs to wait for packer to finish writing to L1 before it starts reading from L1 for the second iteration of LLK pipeline.

    Synchronization is accomplished between the packer and unpacker operations using the PACK_DONE semaphore.
    The packer first writes data to L1 and signals the unpacker by incrementing the semaphore (PACK_DONE = 1).
    The unpacker waits for the semaphore to be set to 1 before reading the data from L1.
    This ensures that the unpacker does not read premature or incorrect data, preventing data race conditions.
    Once the unpacker starts reading, it decrements the semaphore (PACK_DONE = 0) signalling it has started processing data.
    */

    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(
        semaphore::PACK_DONE); // Unpacker waits on signal when packer will increment semaphore to 1 (waits while semaphore == 0), utilizing SEMWAIT.
    t6_semaphore_get<>(semaphore::PACK_DONE); // It will acquire the semaphore t6_semaphore_get (decrementing the semaphore back to 0) signalling it has begun
                                              // processing data from L1
    _llk_unpack_AB_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_B_IN, UNPACK_A_OUT, UNPACK_B_OUT, FACE_R_DIM, 0, 4);
    _llk_unpack_AB_init_<>();
    _llk_unpack_AB_<>(L1_ADDRESS(buffer_A_tilized), L1_ADDRESS(buffer_B_tilized));
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
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

    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BroadcastType::NONE, MATH_FIDELITY>(4, 0, 0);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        4, res_dst_index, false);
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

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, !TILIZE>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<UNTILIZE, false, DstTileFaceLayout::RowMajor, false, !TILIZE>(PACK_OUT);
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>(res_dst_index, L1_ADDRESS(buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
