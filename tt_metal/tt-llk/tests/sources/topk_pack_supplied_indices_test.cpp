// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Bit-exact supplied-index packing, including both 512-element score halves.
// Raw UInt32 inputs isolate packing from score conversion and sorting.

#include <cstdint>
#include <initializer_list>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t ELEMENTS_PER_TILE = ckernel::TILE_R_DIM * ckernel::TILE_C_DIM;
static constexpr std::uint32_t INPUT_TILES       = (PACK_K + ELEMENTS_PER_TILE - 1) / ELEMENTS_PER_TILE;
static constexpr std::uint32_t SCORES            = 0;
static constexpr std::uint32_t IDS               = 2;
static constexpr std::uint32_t OUT               = PACK_ALIAS == 0 ? 4 : (PACK_ALIAS == 1 ? SCORES : IDS);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (const auto buffer : {params.buffer_A, params.buffer_B})
    {
        for (std::uint32_t t = 0; t < INPUT_TILES; ++t)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(buffer[t]), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/experimental/ckernel_sfpu_topk_pack_indices.h"

using namespace ckernel;

static_assert(is_fp32_dest_acc_en, "supplied-index packing addresses 32-bit DEST words");

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        tensor_shape, formats.math);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_pack_supplied_indices_init_();

    _llk_math_wait_for_dest_available_<dest_sync>();

    for (const auto base_tile : {SCORES, IDS})
    {
        for (std::uint32_t t = 0; t < INPUT_TILES; ++t)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                base_tile + t /* dst_index */, formats.math, formats.math, tensor_shape);
        }
    }
    _llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::_topk_xl_pack_supplied_indices_<PACK_K>, SCORES, IDS, OUT, VectorMode::RC_custom, PACK_HALF);

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Preserve the complete packed score/ID word.
    constexpr std::uint32_t pack_format = ckernel::to_underlying(DataFormat::UInt32);
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(pack_format, pack_format, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(pack_format);
    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t t = 0; t < INPUT_TILES; ++t)
    {
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(OUT + t /* tile_index */, L1_ADDRESS(params.buffer_Res[t]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
