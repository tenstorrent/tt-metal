// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole regression: dense untilize must restore linear Dest addressing.
// Run 1 writes under inherited state, then forces linear readback.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t dense_untilize_scratch = 0xA0000;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);

    // Run 0: feed dense pack-untilize.
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);

    // Do not start run 1 until the packer has drained run 0 and its output can
    // no longer overlap the remap teardown.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // Run 1: a normal datacopy using the same operand configuration.
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces);

    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(tensor_shape, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // Run 0: dense pack-untilize with remapped Dest reads.
    _llk_math_reconfig_remap_(true);
    _llk_math_wait_for_dest_available_<dest_sync>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats.math, formats.math, tensor_shape);
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();

    // This is the restore under test. It waits for run 0's pack before changing
    // the shared remap bits.
    _llk_math_reconfig_remap_(false);

    // Run 1 writes with the state left by the teardown under test.
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(tensor_shape, formats.math);
    _llk_math_wait_for_dest_available_<dest_sync>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats.math, formats.math, tensor_shape);
    // Force only the readback linear so a leaked remap cannot round-trip.
    _llk_math_reconfig_remap_(false);
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces = params.num_faces;

    // Run 0: dense pack-untilize into scratch.
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, 0 /* tile_size */);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_pack_untilize_init_<BLOCK_CT_DIM * 2, FULL_CT_DIM * 2, false, TILE_C_DIM, true>(formats.pack_src, formats.pack_dst, FACE_R_DIM, num_faces / 2);
    _llk_packer_wait_for_math_done_();
    _llk_pack_untilize_<BLOCK_CT_DIM * 2, FULL_CT_DIM * 2, false, 0, true>(L1_ADDRESS(dense_untilize_scratch), num_faces / 2, 0 /* tile_dst_rt_offset */);
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    _llk_pack_untilize_uninit_(formats.pack_src);

    // Release unpack only after run 0 has drained and its packer state is torn down.
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // Run 1: restore a normal packer and write the checked output.
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
        formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces, FACE_R_DIM, TILE_C_DIM, num_faces);
    _llk_pack_init_<ckernel::PackMode::Default, false /* zero_output */>(
        formats.pack_src, FACE_R_DIM, TILE_C_DIM, num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
    // Preserve the Dest half selected by run 0; reinit would desynchronize pack from math.
    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
