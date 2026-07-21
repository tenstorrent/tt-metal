// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused multiply + reduce-to-scalar LLK test (experimental, Blackhole only).
//
// Mirrors the ttnn compute-kernel flow of ckernel::mul_reduce_scalar_tile:
//   1. Multiply phase: C[i] = A[i] * B[i] element-wise (ELWMUL) into DEST tiles.
//   2. Switch UNPACK -> reduce phase (DEST reused as SrcA/SrcB via MOVD2A/MOVD2B).
//   3. Column-reduce every tile with GAPOOL, accumulating into DEST[0].
//   4. Collapse DEST[0] to a single scalar (transpose + GAPOOL).
// The packer applies the REDUCE_SCALAR mask so only element [0] is emitted.
//
// This expands the Compute API (api/compute/experimental/mul_reduce_scalar.h)
// into its underlying _llk_* calls so the kernel runs inside the tt-llk harness.

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
// The reduce phase reuses DEST as source, so the whole tile (4 faces) is live.
static constexpr std::uint32_t NUM_FACES = 4;
// Scaler multiplier applied to the reduction (matches the Compute API default).
static constexpr float REDUCE_SCALER = 1.0f;

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
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2, 2};

    // compute_kernel_hw_startup
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, NUM_FACES, NUM_FACES);

    // mul_reduce_scalar_init: unpack A and B, no broadcast/transpose.
    _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::None);

    // Multiply phase: stream A[i] and B[i] into SrcA/SrcB for each tile.
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
    }

    // Switch to the reduce phase: reset counters and re-arm SrcA/SrcB DVALID so
    // MATH can reuse DEST as source operands.
    _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_fill.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t tile_cnt            = params.TILE_CNT;
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2, 2};

    // compute_kernel_hw_startup
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // compute_kernel_hw_startup programs the SFPU config register once per
    // kernel; this standalone harness bypasses it, so run the idempotent
    // once-init before the reduce-phase _calculate_fill_ SFPU stores.
    _llk_math_eltwise_unary_sfpu_init_once_();

    // mul_reduce_scalar_init: element-wise multiply, no accumulate-to-dest.
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        tensor_shape, 0 /* acc_to_dest */);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Step 1 - multiply phase: C[i] = A[i] * B[i] into DEST[i].
    for (std::uint32_t i = 0; i < tile_cnt; ++i)
    {
        LLK_ASSERT((i < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Multiply tile index exceeds maximum destination tiles");
        _llk_math_eltwise_binary_<
            EltwiseBinaryType::ELWMUL,
            BroadcastType::NONE,
            DST_SYNC,
            is_fp32_dest_acc_en,
            MATH_FIDELITY,
            EltwiseBinaryReuseDestType::NONE>(tensor_shape, i, true /* clear_fp32_dst_acc */);
    }

    // Step 3 - initialize the reduce phase (addr mods + counter reset).
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();

    // Step 4 - stage tile 0 into SrcA, fill SrcB with the scaler, clear DEST[0].
    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0);
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, 0 /* dst_index */, VectorMode::RC_custom, REDUCE_SCALER);
    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0);
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, 0 /* dst_index */, VectorMode::RC_custom, 0.0f /* clear DEST[0] */);

    // Step 6 - column-reduce every tile, accumulating into DEST[0].
    // (narrow_tile / num_faces are derived internally from the TensorShape.)
    _llk_math_mul_reduce_column_<MATH_FIDELITY>(0, tensor_shape);
    for (std::uint32_t i = 1; i < tile_cnt; ++i)
    {
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i);
        _llk_math_mul_reduce_column_<MATH_FIDELITY>(0, tensor_shape);
    }

    // Step 7 - collapse DEST[0] to a single scalar.
    _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();

    // Step 8 - clear DVALID flags.
    _llk_math_mul_reduce_scalar_clear_dvalid_();

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2, 2};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile        = tensor_shape.num_faces_c_dim == 1;

    // compute_kernel_hw_startup
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    // mul_reduce_scalar_tile step 5: mask so only the reduced scalar [0] is packed.
    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();

    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    // Single output tile: the scalar lives in DEST[0].
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    // mul_reduce_scalar_uninit
    _llk_pack_reduce_mask_clear_();
}

#endif
