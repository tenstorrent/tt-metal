// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Single-chip repro for tt-metal#47016 (and, by the same root cause, #47049).
//
// Pipeline (one Tensix core, two L1->L1 runs that share the unpacker config
// context):
//   run 0: tilize operand A and operand B into L1
//          -> ends by calling _llk_unpack_tilize_uninit_(dst, UNINIT_NUM_FACES)
//   run 1: matmul of the now-tilized operands, read back as a BFP-compressed
//          (Bfp8_b) operand.
//
// The tile-descriptor z-dim that the tilize-uninit leaves behind sizes the
// per-tile BFP exponent / RowStart array for the following matmul
// (NumBlobs = BlobsPerXYPlane * ZDim * WDim). For a full tile the matmul needs
// z-dim = 4. #45179 changed the uninit to restore the operand's num_faces
// instead of the hardcoded 4, so for a num_faces != 4 operand the exponent
// array is mis-sized and the BFP matmul decodes garbage exponents
// (the +-FLT_MAX saturation seen in the VLLM nightly).
//
// Deciding variable, matching the A/B table in the issue:
//   UNINIT_NUM_FACES == 4  -> PASS
//   UNINIT_NUM_FACES != 4  -> FAIL (corrupt / saturated matmul output)
//
// This is the same defect that produces Qwen3-32B Top-1 0% on Galaxy (#47049):
// tt_transformers lm_head matmuls bfloat8_b weights right after an in-kernel
// tilize of the activations.

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

constexpr std::uint32_t buffer_A_tilized = 0xA0000;
constexpr std::uint32_t buffer_B_tilized = 0xA1000;

// UNINIT_NUM_FACES is injected as a template constant by test_repro_47016.py
// (helpers.test_variant_parameters.UNINIT_NUM_FACES). It is the z-dim value the
// tilize-uninit restores: 4 = correct full-tile face count, anything else
// reproduces the #45179 regression.

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

    int run = 0;
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
        4,
        false);

    _llk_unpack_tilize_init_wrapper_(formats_array[run].unpack_B_src, formats_array[run].unpack_B_dst, 1 /* ct_dim */, FACE_R_DIM, false /* narrow_tile */);
    _llk_unpack_tilize_wrapper_(
        L1_ADDRESS(params.buffer_B[0]),
        0 /* tile_index */,
        formats_array[run].unpack_B_src,
        formats_array[run].unpack_B_dst,
        block_ct_dim,
        FACE_R_DIM,
        4,
        false);

    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // Start of second unpack kernel to perform unpack matmul on now tilized input data.
    run = 1;
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
        formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst, tile_size);
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
        formats_array[run].unpack_B_src, formats_array[run].unpack_B_dst, tile_size);

    // *** The one defect under test ***
    // Pre-#45179 this was hardcoded to 4 (always correct for the descriptor
    // z-dim a BFP matmul consumer expects). #45179 made it the operand's
    // num_faces. UNINIT_NUM_FACES lets the test flip the deciding variable.
    _llk_unpack_tilize_uninit_wrapper_(formats_array[run].unpack_A_dst, UNINIT_NUM_FACES);

    _llk_unpack_AB_matmul_init_<>();
    _llk_unpack_AB_matmul_<>(L1_ADDRESS(buffer_A_tilized), L1_ADDRESS(buffer_B_tilized), 0, 0, tile_size, tile_size);
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

    int run = 0;

    const bool TILIZE = true;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, TILIZE>>(4 /* num_faces */, formats_array[run].math);

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats_array[run].math, formats_array[run].math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
        operand_A_dst_index, formats_array[run].math, formats_array[run].math);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
        operand_B_dst_index, formats_array[run].math, formats_array[run].math);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    run = 1;
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(formats_array[run].math);
    _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, false>(formats_array[run].math);
    _llk_math_matmul_init_<MATH_FIDELITY>();
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_matmul_<MATH_FIDELITY>(0);
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

    int run                      = 0;
    static constexpr bool TILIZE = true;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
        formats_array[run].pack_src, formats_array[run].pack_dst, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(formats_array[run].pack_dst);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(operand_A_dst_index, L1_ADDRESS(buffer_A_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(operand_B_dst_index, L1_ADDRESS(buffer_B_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    t6_semaphore_post<>(semaphore::PACK_DONE);

    run = 1;
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(formats_array[run].pack_src, formats_array[run].pack_dst, tile_size);

#ifdef ARCH_BLACKHOLE
    _llk_pack_init_<ckernel::PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats_array[run].pack_src, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(res_dst_index, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
