// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Chunked fused multiply + reduce-to-scalar LLK test (experimental, Blackhole only).
//
// This is the REVERTED "chunked" driver for mul_reduce_scalar (promotion strategy
// §3, open-question #1). The non-chunked mul_reduce_scalar_tile caps num_tiles at
// the DEST half-sync capacity (8 bf16 / 4 fp32) because every multiply product must
// be resident in DEST before the reduce phase consumes it. The chunked driver lifts
// that cap by processing the tile stream in fixed-size chunks: for each chunk it runs
// the full multiply -> switch-to-reduce -> column-reduce -> collapse-to-scalar
// pipeline, then accumulates the chunk's scalar into a running total kept in DEST[0]
// between reduces.
//
// GOLDEN DERIVATION (from api/compute/experimental/mul_reduce_scalar.h and
// llk_lib/experimental/llk_math_mul_reduce_scalar.h):
//   Per-tile the op computes C[i] = A[i] * B[i] (ELWMUL), column-reduces every tile
//   (GAPOOL) accumulating into DEST[0], then collapses DEST[0] to a single scalar
//   (transpose + GAPOOL). Chunking only changes the *order* of accumulation: the math
//   is still an exact sum, so
//       result = sum over all tiles i, all elements e of ( A[i][e] * B[i][e] ).
//   With B held at 1.0 (as in the on-silicon gtest / fpu_reduce_scalar.yaml) this is
//   sum(A). Only DEST element [0] is defined (REDUCE_SCALAR pack mask); every other
//   lane is unspecified and MUST NOT be validated.
//
// KNOWN FAILURE (why this test is xfail): on silicon the chunked result comes out
// ~5-30x too high. The suspected cause is the between-chunk DEST[0] restore: the
// running scalar in DEST[0] is clobbered / double-counted when the next chunk's
// multiply phase and clear/fill sequence re-touch DEST[0]. The test is written to
// COMPILE cleanly for Blackhole; it is expected to FAIL numerically at runtime.
//
// This expands the Compute API (api/compute/experimental/mul_reduce_scalar.h) into
// its underlying _llk_* calls, wrapped in a per-chunk loop, so the kernel runs inside
// the tt-llk harness.

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
// kernel targets index 0.
static constexpr std::uint32_t DST_INDEX = 0;

// Number of tiles processed per chunk. CHUNK_SIZE tiles must fit the DEST half-sync
// slot budget (<= 8 bf16 / <= 4 fp32). Supplied as a compile-time runtime param.
// The tile stream length TILE_CNT need not be a multiple of CHUNK_SIZE; the last
// chunk handles the remainder.

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

    // compute_kernel_hw_startup
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        tensor_shape.face_r_dim,
        tensor_shape.face_r_dim,
        tensor_shape.total_num_faces(),
        tensor_shape.total_num_faces());

    // mul_reduce_scalar_init: unpack A and B, no broadcast/transpose.
    _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::None);

    const std::uint32_t tile_cnt   = params.TILE_CNT;
    const std::uint32_t chunk_size = CHUNK_SIZE;

    // Chunked stream: for each chunk, unpack its tiles into SrcA/SrcB, then switch to
    // the reduce phase so MATH can reuse DEST as source operands for the chunk reduce.
    for (std::uint32_t base = 0; base < tile_cnt; base += chunk_size)
    {
        const std::uint32_t this_chunk = (tile_cnt - base < chunk_size) ? (tile_cnt - base) : chunk_size;

        // Multiply phase: stream A[base+j] and B[base+j] into SrcA/SrcB.
        for (std::uint32_t j = 0; j < this_chunk; ++j)
        {
            const std::uint32_t i = base + j;
            _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
        }

        // Switch to the reduce phase: reset counters and re-arm SrcA/SrcB DVALID so
        // MATH can reuse DEST as source operands for this chunk's reduction.
        _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_fill.h"

// Scaler multiplier applied to the reduction (matches the Compute API default).
static constexpr float REDUCE_SCALER = 1.0f;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t tile_cnt            = params.TILE_CNT;
    const std::uint32_t chunk_size          = CHUNK_SIZE;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    // compute_kernel_hw_startup
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // compute_kernel_hw_startup programs the SFPU config register once per kernel;
    // this standalone harness bypasses it, so run the idempotent once-init before the
    // reduce-phase _calculate_fill_ SFPU stores.
    _llk_math_eltwise_unary_sfpu_init_once_();

    // mul_reduce_scalar_init: element-wise multiply, no accumulate-to-dest.
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        tensor_shape, 0 /* acc_to_dest */);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // The running scalar total accumulates across chunks in DEST[0]. On the FIRST
    // chunk it is initialized to 0; on later chunks the between-reduces restore is
    // where the reverted driver's ~5-30x error is believed to originate.
    bool first_chunk = true;

    for (std::uint32_t base = 0; base < tile_cnt; base += chunk_size)
    {
        const std::uint32_t this_chunk = (tile_cnt - base < chunk_size) ? (tile_cnt - base) : chunk_size;

        // Step 1 - multiply phase: C[j] = A[base+j] * B[base+j] into DEST[j].
        for (std::uint32_t j = 0; j < this_chunk; ++j)
        {
            LLK_ASSERT(
                (j < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Chunk tile index exceeds maximum destination tiles");
            _llk_math_eltwise_binary_<
                EltwiseBinaryType::ELWMUL,
                BroadcastType::NONE,
                DST_SYNC,
                is_fp32_dest_acc_en,
                MATH_FIDELITY,
                EltwiseBinaryReuseDestType::NONE>(tensor_shape, j, true /* clear_fp32_dst_acc */);
        }

        // Step 3 - initialize the reduce phase (addr mods + counter reset).
        _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();

        // Step 4 - stage chunk tile 0 into SrcA, fill SrcB with the scaler, then set
        // up DEST[0]. On the first chunk DEST[0] is cleared to 0. On later chunks the
        // running total already lives in DEST[0] and must be preserved so this chunk
        // accumulates onto it -- this between-reduces restore is the reverted step.
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DST_INDEX);
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, REDUCE_SCALER);
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(DST_INDEX);
        if (first_chunk)
        {
            _llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, 0.0f /* clear DEST[0] */);
        }

        // Step 6 - column-reduce every tile in the chunk, accumulating into DEST[0].
        _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
        for (std::uint32_t j = 1; j < this_chunk; ++j)
        {
            _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(j);
            _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
        }

        // Step 7 - collapse DEST[0] to a single running scalar.
        _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();

        first_chunk = false;
    }

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

    // No-src init: packer strides are owned by the hw-configure above.
    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);

    // mul_reduce_scalar_tile step 5: mask so only the reduced scalar [0] is packed.
    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();

    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // Single output tile: the accumulated scalar lives in DEST[0].
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    // mul_reduce_scalar_uninit
    _llk_pack_reduce_mask_clear_();
}

#endif
