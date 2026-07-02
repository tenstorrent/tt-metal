// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

// Translation of these lines:
// const FormatConfig(&formats_array)[2] = params.formats;
// to English:
// Constant reference to an array of 2 FormatConfig objects

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif

    const std::uint32_t block_ct_dim = _llk_unpack_tilize_block_ct_dim_wrapper_(BLOCK_CT_DIM);

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

    _llk_unpack_tilize_init_wrapper_(formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst, 1 /* ct_dim */, FACE_R_DIM, false /* narrow_tile */);
    _llk_unpack_tilize_wrapper_(
        L1_ADDRESS(params.buffer_A[0]),
        0 /* tile_index */,
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_A_dst,
        block_ct_dim,
        FACE_R_DIM,
        4 /* num_faces */,
        false /* narrow_tile */);

    _llk_unpack_tilize_init_wrapper_(formats_array[run].unpack_B_src, formats_array[run].unpack_B_dst, 1 /* ct_dim */, FACE_R_DIM, false /* narrow_tile */);
    _llk_unpack_tilize_wrapper_(
        L1_ADDRESS(params.buffer_B[0]),
        0 /* tile_index */,
        formats_array[run].unpack_B_src,
        formats_array[run].unpack_B_dst,
        block_ct_dim,
        FACE_R_DIM,
        4 /* num_faces */,
        false /* narrow_tile */);

    /*
    In this test we fuse two LLK pipeline runs, one is to unpack untilized buffers/operands from L1 (39-45) and pack them in tilized format(130-145).
    The next run unpacks these two tilized operands, performs a math compute and pack them out in untilized format.
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

    run = 1; // second L1-to-L1 run, we access the second set of formats_array in our array
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_B_src,
        formats_array[run].unpack_A_dst,
        formats_array[run].unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        4 /* num_faces */,
        4 /* num_faces */);
    _llk_unpack_AB_init_<>(DEFAULT_TENSOR_SHAPE);
    _llk_unpack_AB_<>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]));
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    const bool is_int_fpu_en                = false;
    const std::uint32_t operand_A_dst_index = 1;
    const std::uint32_t operand_B_dst_index = 2;
    const std::uint32_t res_dst_index       = 0;
    int run                                 = 0; // first L1-to-L1 run, we access the first set of formats_array in our array

    // copy srca to dest
    const bool TILIZE = true;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, TILIZE>>(4 /* num_faces */, formats_array[run].math);

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats_array[run].math, formats_array[run].math);

    // copy tilized inputs to dest indexes 0 and 1
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
        operand_A_dst_index, formats_array[run].math, formats_array[run].math);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
        operand_B_dst_index, formats_array[run].math, formats_array[run].math);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    run = 1; // second L1-to-L1 run, we access the second set of formats_array in our array
    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BroadcastType::NONE>(DEFAULT_TENSOR_SHAPE, 0 /* acc_to_dest */);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncHalf, is_fp32_dest_acc_en>(
        DEFAULT_TENSOR_SHAPE, res_dst_index, false /* clear_fp32_dst_acc */);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    const std::uint32_t operand_A_dst_index = 1;
    const std::uint32_t operand_B_dst_index = 2;
    const std::uint32_t res_dst_index       = 0;
    static constexpr bool UNTILIZE          = false;
    int run                                 = 0;

    static constexpr bool TILIZE = true;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
        formats_array[run].pack_src, formats_array[run].pack_dst, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(formats_array[run].pack_dst);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(operand_A_dst_index, L1_ADDRESS(params.buffer_A[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(operand_B_dst_index, L1_ADDRESS(params.buffer_B[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>(); // Packer will execute _llk_pack_dest_section_done_ function which ensures the write
                                                                            // to L1 is fully is complete.
    t6_semaphore_post<>(semaphore::PACK_DONE); // The packer signals to the unpacker that it has finished writing to L1 by posting (incrementing) the semaphore.
                                               // Now unpacker's wait condition is satisfied, allowing it to begin processing data from L1.
    run = 1;                                   // second L1-to-L1 run, we access the second set of formats_array in our array
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats_array[run].pack_src, formats_array[run].pack_dst, 16 * 16 * 4);
    // The hw-configure above re-established the packer strides, so skip strides here. The X (datum)
    // counter is not touched by configure_pack; this init programs it (init owns SETADCXX).
    _llk_pack_init_<ckernel::PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats_array[run].pack_src, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(res_dst_index, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
