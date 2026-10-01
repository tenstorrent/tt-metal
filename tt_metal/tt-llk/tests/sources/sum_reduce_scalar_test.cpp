// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sum + reduce-to-scalar LLK test (experimental, Blackhole only).
//
// Mirrors the ttnn compute-kernel flow of ckernel::sum_reduce_scalar_tile
// (api/compute/experimental/sum_reduce_scalar.h). That op is the sum-only
// counterpart of mul_reduce_scalar_tile: instead of an ELWMUL multiply phase
// with an all-ones second operand, it copies each input tile into DEST via
// datacopy (A2D), then runs the *identical* mul_reduce_scalar reduce tail.
//
//   1. Copy phase:   DEST[i] = A[i]  (datacopy A2D, no second operand).
//   2. Switch UNPACK -> reduce phase (DEST reused as SrcA/SrcB via MOVD2A/MOVD2B).
//   3. Fill SrcB with the scaler, clear DEST[0].
//   4. Column-reduce every tile with GAPOOL, accumulating into DEST[0].
//   5. Collapse DEST[0] to a single scalar (transpose + GAPOOL).
// The packer applies the REDUCE_SCALAR mask so only element [0] is emitted.
//
// GOLDEN (see sum_reduce_scalar.h):
//   result = sum(all elements of all tiles) * scaler^2   -> DEST[0]
// The scaler is loaded into SrcB once and applied by BOTH GAPOOL passes (the
// per-tile column accumulate and the final scalar collapse), so it lands on
// the result twice. This test uses scaler == 1.0 (the Compute API default), so
// scaler^2 == 1 and the golden is simply sum(A) over all tiles/elements.
// Only element [0] of the output is defined; every other lane is unspecified.

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

// The reduction collapses everything into DEST[0], so every DEST access in this
// kernel targets index 0. Name it once instead of repeating the bare literal.
static constexpr std::uint32_t DST_INDEX = 0;

// The tile geometry is parametrized by the Python test: the full 32x32 tile
// (2x2 faces, num_faces=4) plus the 16x32 (1x2, num_faces=2) and 16x16 (1x1,
// num_faces=1) "tiny tiles". Only num_faces_{r,c}_dim vary; face_r_dim /
// face_c_dim stay at the full 16 (FACE_R_DIM / FACE_C_DIM). Each thread rebuilds
// the shape from the runtime params, matching the reduce_test.cpp idiom.

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "llk_unpack_A.h"
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

    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    // compute_kernel_hw_startup
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        tensor_shape.face_r_dim,
        tensor_shape.face_r_dim,
        num_faces,
        num_faces);

    // sum_reduce_scalar_init -> copy_tile_to_dst_init_short: unpack A only,
    // no broadcast/transpose (datacopy A2D source path).
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);

    // Copy phase: stream A[i] into SrcA for each tile (MATH moves it into DEST[i]).
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }

    // Switch to the reduce phase: reset counters and re-arm SrcA/SrcB DVALID so
    // MATH can reuse DEST as source operands. Shared with mul_reduce_scalar.
    _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "params.h"

// NOTE: we deliberately do NOT include "sfpu/ckernel_sfpu_fill.h" here. In this
// environment the installed SFPI predates the sfpi::DataLayout enum, so the
// header's _calculate_fill_int_ template body fails to parse under
// -Wtemplate-body -Werror (sfpi::DataLayout::I32 / ::U16 are unknown), which
// breaks the whole translation unit even though this kernel only needs the
// float fill. We redeclare just the float _calculate_fill_ (identical to the
// header's lines 16-27) in the same namespace; it depends only on sfpi::vFloat
// and sfpi::dst_reg and never touches DataLayout.
namespace ckernel::sfpu
{
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_fill_(const float value)
{
    sfpi::vFloat fill_val = value;
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = fill_val;
        sfpi::dst_reg++;
    }
}
} // namespace ckernel::sfpu

// Scaler multiplier applied to the reduction (matches the Compute API default).
// scaler == 1.0 => scaler^2 == 1.0, so the golden is a plain sum(A).
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

    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    // compute_kernel_hw_startup
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // compute_kernel_hw_startup programs the SFPU config register once per
    // kernel; this standalone harness bypasses it, so run the idempotent
    // once-init before the reduce-phase _calculate_fill_ SFPU stores.
    _llk_math_eltwise_unary_sfpu_init_once_();

    // sum_reduce_scalar_init -> copy_tile_to_dst_init_short: datacopy A2D.
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(num_faces, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Step 1 - copy phase: DEST[i] = A[i] via datacopy A2D.
    for (std::uint32_t i = 0; i < tile_cnt; ++i)
    {
        LLK_ASSERT((i < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Copy tile index exceeds maximum destination tiles");
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats.math, formats.math, num_faces);
    }

    // Step 3 - initialize the reduce phase (addr mods + counter reset).
    // Shared reduce tail with mul_reduce_scalar.
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();

    // Step 4 - stage tile 0 into SrcA, fill SrcB with the scaler, clear DEST[0].
    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DST_INDEX);
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, REDUCE_SCALER);
    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(DST_INDEX);
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, 0.0f /* clear DEST[0] */);

    // Step 6 - column-reduce every tile, accumulating into DEST[0].
    // (narrow_tile / num_faces are derived internally from the TensorShape.)
    _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
    for (std::uint32_t i = 1; i < tile_cnt; ++i)
    {
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i);
        _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
    }

    // Step 7 - collapse DEST[0] to a single scalar.
    _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();

    // Step 8 - clear DVALID flags.
    _llk_math_mul_reduce_scalar_clear_dvalid_();

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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

    // Blackhole-only test: call the pack LLKs directly (the _wrapper_ helpers exist
    // only to paper over the WH/BH signature split for dual-arch tests).
    // compute_kernel_hw_startup
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face);

    // No-src init: packer strides are owned by the hw-configure above, so skip re-programming them here.
    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);

    // sum_reduce_scalar_tile: mask so only the reduced scalar [0] is packed.
    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();

    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // Single output tile: the scalar lives in DEST[0].
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    // mul_reduce_scalar_uninit
    _llk_pack_reduce_mask_clear_();
}

#endif
