// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The rmsnorm bcast-scalar dest-reuse op in the call order of its callers (DeepSeek sampling, blaze softmax_top_p,
// softmax_lanes): a LoFi ELWSUB and a multiply at MATH_FIDELITY with its own init, one tile each, in both orders.
// Section 0 runs the subtract before the multiply, section 1 after it, with no other unpack init in between, so a
// multiply init that left the unpacker state behind breaks the subtract. Each section seeds DEST[0] and DEST[1] with
// the scalar tile; the subtract overwrites DEST[1] with A - s, the multiply adds A * s onto DEST[0].

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_rmsnorm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    const auto seed = [&]
    {
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0, 0, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        for (std::uint32_t tile = 0; tile < 2; ++tile)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
        }
    };
    const auto op = [&](const bool whole_tile)
    {
        _llk_unpack_A_rmsnorm_init_<1, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0, 0, FACE_R_DIM, 4, 0, 0, whole_tile);
        _llk_unpack_A_<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
            L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    };

    seed();
    op(false);
    op(RMSNORM_WHOLE_TILE);
    seed();
    op(RMSNORM_WHOLE_TILE);
    op(false);
}

#endif

#ifdef LLK_TRISC_MATH

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    const auto seed = [&]
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            TILE_NUM_FACES, formats.math);
        for (std::uint32_t tile = 0; tile < 2; ++tile)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, formats.math, formats.math);
        }
    };
    const auto sub = []
    {
        _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<EltwiseBinaryType::ELWSUB, 1, MathFidelity::LoFi>(4, 0);
        _llk_math_rmsnorm_bcast_scalar_dest_reuse_<EltwiseBinaryType::ELWSUB, 1, DST_SYNC, is_fp32_dest_acc_en, MathFidelity::LoFi, false>(1, 1);
    };
    const auto mul = []
    {
        _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<EltwiseBinaryType::ELWMUL, 1, MATH_FIDELITY>(4, 0, RMSNORM_WHOLE_TILE);
        _llk_math_rmsnorm_bcast_scalar_dest_reuse_<EltwiseBinaryType::ELWMUL, 1, DST_SYNC, is_fp32_dest_acc_en, MATH_FIDELITY, false>(0, 0);
    };

    seed();
    sub();
    mul();
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    seed();
    mul();
    sub();
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();
    for (std::uint32_t section = 0; section < 2; ++section)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < 2; ++tile)
        {
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[2 * section + tile]));
        }
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
