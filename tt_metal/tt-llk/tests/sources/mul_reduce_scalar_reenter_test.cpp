// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Is the mul_reduce_scalar family re-enterable? (experimental, Blackhole only)
//
// Why this exists. mul_reduce_scalar_chunked_tile
// (api/compute/experimental/mul_reduce_scalar.h) has no test and no in-tree caller. Two
// attempts at a full chunked driver were reverted: every bf16 variant came back 5-30x
// golden, and the fault was localised to a SINGLE batch's reduce rather than the
// cross-batch accumulation -- packing DEST[0] for num_tiles=3/dst_capacity=2 read 37120
// where one tile's sum is ~512, about 72x too large. The accumulator fill and a missing
// UNPACK/MATH barrier were both tried on silicon and both changed the output byte for byte
// not at all, so neither is the cause.
//
// That left one structural difference between the chunked form and the working
// non-chunked driver (mul_reduce_scalar_test.cpp, green across 54 variants): the chunked
// form invokes the per-batch inits -- math _llk_math_mul_reduce_scalar_init_, unpack
// _llk_unpack_AB_init_ + switch_to_reduce -- once per batch instead of once per kernel.
// Hence the hypothesis this file tests in isolation: the reduce family is not
// re-enterable, i.e. a second init accumulates addrmod/counter state instead of
// re-establishing it.
//
// This is deliberately NOT the chunked driver. It runs the known-good non-chunked
// sequence REDUCE_PASSES times over the SAME input, re-doing exactly what the chunked
// loop re-does per batch, and packs each pass's scalar to its own result tile. Two passes
// over identical input must produce identical scalars. If they do not, the family is not
// re-enterable and that is the chunked bug; if they do, the hypothesis is dead and the
// fault is in the chunking arithmetic or the accumulator instead -- either outcome is
// worth more than another attempt at the full sweep.
//
// REDUCE_PASSES=1 is the control: it reduces to the non-chunked sequence, so it must match
// the same golden test_mul_reduce_scalar.py asserts. That guards against reading a bug in
// this driver as a finding about the LLKs.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// Everything collapses into DEST[0].
static constexpr std::uint32_t DST_INDEX = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    // compute_kernel_hw_startup -- once per kernel, as in the non-chunked driver.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        tensor_shape.face_r_dim,
        tensor_shape.face_r_dim,
        tensor_shape.total_num_faces(),
        tensor_shape.total_num_faces());

    for (std::uint32_t pass = 0; pass < REDUCE_PASSES; ++pass)
    {
        // Per-batch in the chunked form: re-arm the multiply phase.
        _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::None);

        // Same tiles every pass, so every pass must reduce to the same scalar.
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
        }

        _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_fill.h"

// Matches the Compute API default.
static constexpr float REDUCE_SCALER = 1.0f;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t tile_cnt            = params.TILE_CNT;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    // compute_kernel_hw_startup -- once per kernel.
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_init_once_();

    // One pass of the non-chunked sequence, ending with the scalar in DEST[0]. Everything
    // in here is what the chunked loop re-issues per batch.
    auto reduce_once = [&]()
    {
        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
            tensor_shape, 0 /* acc_to_dest */);

        for (std::uint32_t i = 0; i < tile_cnt; ++i)
        {
            LLK_ASSERT(
                (i < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Multiply tile index exceeds maximum destination tiles");
            _llk_math_eltwise_binary_<
                EltwiseBinaryType::ELWMUL,
                BroadcastType::NONE,
                DST_SYNC,
                is_fp32_dest_acc_en,
                MATH_FIDELITY,
                EltwiseBinaryReuseDestType::NONE>(tensor_shape, i, true /* clear_fp32_dst_acc */);
        }

        // THE CALL UNDER TEST on pass >= 1: the chunked form issues this once per batch.
        _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();

        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DST_INDEX);
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, REDUCE_SCALER);
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(DST_INDEX);
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, 0.0f /* clear DEST[0] */);

        _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
        for (std::uint32_t i = 1; i < tile_cnt; ++i)
        {
            _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i);
            _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
        }

        _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();
        _llk_math_mul_reduce_scalar_clear_dvalid_();
    };

    if constexpr (SINGLE_DEST_SECTION)
    {
        // What mul_reduce_scalar_chunked_tile actually does: the caller acquires DST once
        // and every batch re-enters inside that one section, with no pack handshake in
        // between. Only the final scalar is packed, which is all this mode needs -- the
        // question is whether the LAST reduce is right, not whether earlier ones were.
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        for (std::uint32_t pass = 0; pass < REDUCE_PASSES; ++pass)
        {
            reduce_once();
        }
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
    else
    {
        // A full section boundary between passes, so each scalar can be packed on its own.
        // Re-establishes more state than the chunked form does -- see the module comment.
        for (std::uint32_t pass = 0; pass < REDUCE_PASSES; ++pass)
        {
            _llk_math_wait_for_dest_available_<DST_SYNC>();
            reduce_once();
            _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face);

    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);

    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();

    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    if constexpr (SINGLE_DEST_SECTION)
    {
        // One section, so one handshake and one scalar: the last pass's.
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
    else
    {
        // One scalar per pass, each to its own result tile, so the Python side can compare
        // pass 1 against pass 0 rather than only against the golden.
        for (std::uint32_t pass = 0; pass < REDUCE_PASSES; ++pass)
        {
            _llk_packer_wait_for_math_done_();
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[pass]));
            _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
        }
    }

    _llk_pack_reduce_mask_clear_();
}

#endif
