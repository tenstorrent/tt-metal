// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
std::uint32_t tile_size                = 128;

// Remove later
constexpr std::uint32_t buffer_A_tilized         = 0x30000;
constexpr std::uint32_t buffer_B_tilized         = 0x50000;
constexpr std::uint32_t intermediate_tile_stride = 0x1000;

// Translation of these lines:
// const FormatConfig(&formats_array)[2] = params.formats;
// to English:
// Constant reference to an array of 2 FormatConfig objects

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_AB_matmul.h"
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
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_unpack_tilize_wrapper_(
            L1_ADDRESS(params.buffer_A[0]), 0, formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst, block_ct_dim, FACE_R_DIM, 4, false);
    }

    _llk_unpack_tilize_init_wrapper_(formats_array[run].unpack_B_src, formats_array[run].unpack_B_dst, 1 /* ct_dim */, FACE_R_DIM, false /* narrow_tile */);
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_unpack_tilize_wrapper_(
            L1_ADDRESS(params.buffer_B[0]), 0, formats_array[run].unpack_B_src, formats_array[run].unpack_B_dst, block_ct_dim, FACE_R_DIM, 4, false);
    }

    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(
        semaphore::PACK_DONE); // Unpacker waits on signal when packer will increment semaphore to 1 (waits while semaphore == 0), utilizing SEMWAIT.
    t6_semaphore_get<>(semaphore::PACK_DONE); // It will acquire the semaphore t6_semaphore_get (decrementing the semaphore back to 0) signalling it has begun

    // Start of second unpack kernel to perform unpack matmul on now tilized input data
    run = 1; // second L1-to-L1 run, we access the second set of formats_array in our array
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_A_dst,
        tile_size); // have to reconfigure unpack kernel data formats_array if they change in this run
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
        formats_array[run].unpack_B_src, formats_array[run].unpack_B_dst, tile_size);
    _llk_unpack_tilize_uninit_wrapper_(formats_array[run].unpack_A_dst, 4 /* num_faces */);
    _llk_unpack_AB_matmul_init_<>();
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        const std::uint32_t offset = block * intermediate_tile_stride;
        _llk_unpack_AB_matmul_<>(L1_ADDRESS(buffer_A_tilized + offset), L1_ADDRESS(buffer_B_tilized + offset), 0, 0, tile_size, tile_size);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_matmul.h"
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

    // copy srca to dest
    int run = 0; // first L1-to-L1 run, we access the first set of formats_array in our array

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
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
            operand_A_dst_index, formats_array[run].math, formats_array[run].math);
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
            operand_B_dst_index, formats_array[run].math, formats_array[run].math);
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }

    // Start of second math kernel to perform matmul on now tilized input data
    run = 1; // second L1-to-L1 run, we access the second set of formats_array in our array
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(
        formats_array[run].math); // have to reconfigure math kernel data formats_array if they change in this run
    _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, false>(formats_array[run].math);
    _llk_math_matmul_init_<MATH_FIDELITY>();
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_matmul_<MATH_FIDELITY>(0);
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
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

    int run                      = 0; // first L1-to-L1 run, we access the first set of formats_array in our array
    static constexpr bool TILIZE = true;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
        formats_array[run].pack_src, formats_array[run].pack_dst, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(formats_array[run].pack_dst);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(
            operand_A_dst_index, L1_ADDRESS(buffer_A_tilized + block * intermediate_tile_stride));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(
            operand_B_dst_index, L1_ADDRESS(buffer_B_tilized + block * intermediate_tile_stride));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
    t6_semaphore_post<>(semaphore::PACK_DONE); // The packer signals to the unpacker that it has finished writing to L1 by posting (incrementing) the semaphore.
                                               // Now unpacker's wait condition is satisfied, allowing it to begin processing data from L1.

    // Start of second pack kernel to perform final pack after executing matmul on tilized data
    run = 1; // second L1-to-L1 run, we access the second set of formats_array in our array
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        formats_array[run].pack_src,
        formats_array[run].pack_dst,
        tile_size); // need to reconfigure data formats_array for next pack, also calls set_packer_strides to readjust strides after pack tilizing

#ifdef ARCH_BLACKHOLE
    // Strides + X counter were re-established by the reconfig above, so skip strides here.
    _llk_pack_init_<ckernel::PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats_array[run].pack_src, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
#endif

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(res_dst_index, L1_ADDRESS(params.buffer_Res[block]));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
