// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Bit-exactness probe for the value-preserving SFPU threshold filter (Blackhole).
//
// One tile of fused FP32 sort keys [bf16 value (high 16) | u16 (index+1) (low 16)]
// is unpacked straight into Dst, the SFPU filter zeroes every key that is not
// strictly greater than the threshold (sign-magnitude total order), and the
// packer emits the result with zero-compression on so the survivors come out
// compacted. The host decodes the compressed stream and compares all 1024 words
// against a torch golden.
//
// This is the NEGATIVE-THRESHOLD fallback for the packer-resident path: the
// packer's MIN_THRESHOLD_RELU cannot express signbit(Threshold)
// (Packers/ReLU.md:41), the SFPU's SFPGT can.
//
// Compile-time knobs (emitted into build.h by the python driver):
//   FILTER_EN               - run the SFPU filter on the math thread
//   THR_BITS                - raw 32-bit threshold the SFPGT compares against
//   COMPRESS_EN             - clear THCON_SEC0_REG1_Disable_zero_compress
//   ROW_START_SECTION_SIZE  - THCON_SEC0_REG1_Row_start_section_size, 16 B units
//   DIAG_TILE_INDEX         - buffer_Res tile index used for the metadata dump
//   DOWNSAMPLE_MASK         - ALWAYS written: set_packer_config never touches
//                             THCON_SEC0_REG1 word 3, so a mask left behind by an
//                             earlier kernel survives an ELF reload and silently
//                             decimates this pack

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"
#include "topk_negfilter_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        params.num_faces, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    if constexpr (FILTER_EN)
    {
        // Establishes the SFPU config register and clears LaneConfig, which is
        // the precondition for the VD >= 12 backdoor template writes inside
        // configure().
        _llk_math_eltwise_unary_sfpu_init_once_();
        topk_negfilter::configure(THR_BITS);
        topk_negfilter::program_replay();
    }

    _llk_math_wait_for_dest_available_<dest_sync>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats.math, formats.math, params.num_faces);

    if constexpr (FILTER_EN)
    {
        _llk_math_eltwise_sfpu_start_(0);
        topk_negfilter::run_tile();
        _llk_math_eltwise_sfpu_done_();
    }

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

using namespace ckernel;

constexpr std::uint32_t TDMA_PACKED_SIZE_T2 = RISCV_TDMA_REG_PACKED_SIZE + 0x080; // PackerTileSize(0, 2)

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
        formats.pack_src,
        formats.pack_dst,
        16 * 16 * 4 /* tile_size */,
        FACE_R_DIM,
        TILE_C_DIM,
        params.num_faces,
        false /* partial_face */,
        false /* narrow_tile */,
        params.RELU_CONFIG /* relu_config */);
    _llk_pack_init_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
        if constexpr (ROW_START_SECTION_SIZE != 0)
        {
            cfg_reg_rmw_tensix<THCON_SEC0_REG1_Row_start_section_size_RMW>(ROW_START_SECTION_SIZE);
        }
        if constexpr (COMPRESS_EN)
        {
            cfg_reg_rmw_tensix<THCON_SEC0_REG1_Disable_zero_compress_RMW>(0);
        }
        // Written unconditionally, including the disabled (0) case -- see the
        // header comment on DOWNSAMPLE_MASK.
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Downsample_mask_RMW>(DOWNSAMPLE_MASK);
    }

    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();

    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
    tensix_sync();

    volatile std::uint32_t* diag = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[DIAG_TILE_INDEX]);

    diag[0]  = 0xC0DEBA5E; // ran-to-here sentinel
    diag[3]  = reg_read(TDMA_PACKED_SIZE_T2);
    diag[4]  = COMPRESS_EN ? 1u : 0u;
    diag[5]  = FILTER_EN ? 1u : 0u;
    diag[6]  = THR_BITS;
    diag[15] = params.num_faces;
    diag[12] = 0xC0DEE0D1;
}
#endif
