// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Switch the pack output between the Dst format and a block-float format with a format reconfig, on a
// short-face tile (8x32, 4x32): the matmul's switch from its L1-accumulated partials to a block-float output.
//
// Three tiles of one Dst block are packed: tile 0 in the Dst format (pack_src) to buffer_C[0], tile 1 to the
// test's output format (pack_dst) to buffer_Res[0] after a format reconfig, tile 2 in the Dst format to
// buffer_C[1] after a reconfig back. The pack init ran for the Dst format; a short-face block-float output of
// more than one face packs through one interface where every other output packs one face per instruction, so
// the reconfig must rebuild the pack MOP both ways, or the block-float tile carries one exponent chunk per face
// (and the pack writes past its page: a guard page after buffer_Res[0] is checked against a sentinel).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Written over every guard page by the pack thread before the packer is configured; repeated in the Python test.
constexpr std::uint32_t GUARD_SENTINEL = 0xC3C3C3C3;

// Dst slots: tile 0 and tile 2 go to buffer_C (Dst format), tile 1 to buffer_Res (output format).
constexpr std::uint32_t NUM_TILES          = 3;
constexpr std::uint32_t OUTPUT_FORMAT_TILE = 1;

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
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.TEST_FACE_R_DIM,
        params.TEST_FACE_R_DIM,
        params.num_faces,
        params.num_faces);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(params.TEST_FACE_R_DIM, params.num_faces),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < NUM_TILES; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        params.num_faces, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t tile = 0; tile < NUM_TILES; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false /* unpack_to_dest */>(
            tile, formats.math, formats.math, params.num_faces);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t face_r_dim = params.TEST_FACE_R_DIM;
    const std::uint32_t tile_c_dim = params.in0_tile_c_dim;
    const std::uint32_t num_faces  = params.num_faces;
    const bool partial_face        = face_r_dim < FACE_R_DIM;

    // One output-format tile in buffer_Res; the guard pages follow it at the result operand's page stride.
    const std::uint32_t page_bytes = params.buffer_Res[1] - params.buffer_Res[0];
    for (std::uint32_t guard = 0; guard < params.NUM_GUARD_TILES; ++guard)
    {
        volatile std::uint32_t* page = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[1 + guard]);
        for (std::uint32_t word = 0; word < page_bytes / sizeof(std::uint32_t); ++word)
        {
            page[word] = GUARD_SENTINEL;
        }
    }

    // Configure and init for the Dst format (the matmul's partials), as the kernel under test does once.
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_src, page_bytes, face_r_dim, tile_c_dim, num_faces, partial_face);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_src, face_r_dim, tile_c_dim, num_faces, partial_face);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();

    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_C[0]));

    // The output format: a format reconfig, no re-init (the pack MOP is the init's unless the reconfig rebuilds it).
    _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(
        formats.pack_src, formats.pack_dst, page_bytes, face_r_dim, tile_c_dim, num_faces, partial_face);
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(OUTPUT_FORMAT_TILE, L1_ADDRESS(params.buffer_Res[0]));

    // Back to the Dst format (the matmul's next output block starts with its partials again).
    _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(
        formats.pack_src, formats.pack_src, page_bytes, face_r_dim, tile_c_dim, num_faces, partial_face);
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(2, L1_ADDRESS(params.buffer_C[1]));

    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_PACK
