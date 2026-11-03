// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Chunked fused multiply + reduce-to-scalar LLK test (experimental, Blackhole only).
//
// The non-chunked mul_reduce_scalar_tile caps num_tiles at the DEST half-sync
// capacity. This driver processes the stream in chunks of CAPACITY-1 products,
// reduces each chunk to a scalar, and adds that scalar into a separate accumulator
// tile (the last DEST slot).
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
// The math chunk loop follows mul_reduce_scalar_chunked_tile in
// api/compute/experimental/rmsnorm.h: one DEST slot is the cross-chunk
// accumulator, each chunk re-inits the multiply, clears only the product slot
// it is about to reuse, reduces that chunk to a scalar, and adds the scalar
// into the accumulator. Scaler is 1, so the packed accumulator element [0] is
// sum(A * B).

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// Chunk scalar is reduced into DEST[0]. The last slot of the capacity is the
// cross-chunk accumulator (mul_reduce_scalar_chunked_tile). The harness CHUNK_SIZE
// is that capacity; one slot is reserved, so a chunk multiplies CAPACITY-1 tiles.
static constexpr std::uint32_t DST_INDEX    = 0;
static constexpr std::uint32_t DST_CAPACITY = 4;
static constexpr std::uint32_t BATCH_SIZE   = DST_CAPACITY - 1;
static constexpr std::uint32_t ACCUMULATOR  = BATCH_SIZE;

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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    const std::uint32_t tile_cnt   = params.TILE_CNT;
    const std::uint32_t chunk_size = params.CHUNK_SIZE;

    {
        START_PERF_MEASURE("INIT")
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
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                LLK_ASSERT(chunk_size == DST_CAPACITY, "CHUNK_SIZE is the DEST capacity, one slot of which is the accumulator");
                for (std::uint32_t base = 0; base < tile_cnt; base += BATCH_SIZE)
                {
                    const std::uint32_t count = (tile_cnt - base < BATCH_SIZE) ? (tile_cnt - base) : BATCH_SIZE;
                    _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(count * tensor_shape.total_num_faces());
                    _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(1); // switch_to_reduce dummy SrcA+SrcB
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                // One product slot is reserved as the accumulator, so a chunk unpacks
                // BATCH_SIZE tiles. Re-init after the previous chunk's reduce switch.
                LLK_ASSERT(chunk_size == DST_CAPACITY, "CHUNK_SIZE is the DEST capacity, one slot of which is the accumulator");
                for (std::uint32_t base = 0; base < tile_cnt; base += BATCH_SIZE)
                {
                    if (base != 0)
                    {
                        _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::None);
                    }
                    const std::uint32_t count = (tile_cnt - base < BATCH_SIZE) ? (tile_cnt - base) : BATCH_SIZE;
                    for (std::uint32_t j = 0; j < count; ++j)
                    {
                        const std::uint32_t i = base + j;
                        _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
                    }
                    _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"
#include "llk_sfpu/ckernel_sfpu_fill.h"
#include "params.h"

// Scaler multiplier applied to the reduction (matches the Compute API default).
static constexpr float REDUCE_SCALER = 1.0f;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const std::uint32_t tile_cnt            = params.TILE_CNT;
    const std::uint32_t chunk_size          = params.CHUNK_SIZE;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    {
        START_PERF_MEASURE("INIT")
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
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            LLK_ASSERT(chunk_size == DST_CAPACITY, "CHUNK_SIZE is the DEST capacity, one slot of which is the accumulator");
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t base = 0; base < tile_cnt; base += BATCH_SIZE)
                {
                    const std::uint32_t count = (tile_cnt - base < BATCH_SIZE) ? (tile_cnt - base) : BATCH_SIZE;
                    _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(count * tensor_shape.total_num_faces());
                    _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(1);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
                }

                // Caller init: zero the accumulator and arm the dest-to-dest add.
                // ADDR_MOD_7 survives the reduce init, which only programs 0-2.
                LLK_ASSERT(chunk_size == DST_CAPACITY, "CHUNK_SIZE is the DEST capacity, one slot of which is the accumulator");
                _llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>();
                _llk_math_eltwise_unary_sfpu_params_(
                    ckernel::sfpu::_calculate_fill_<false /* APPROX */, 8 /* ITERATIONS */>, ACCUMULATOR, VectorMode::RC, 0.0f);

                for (std::uint32_t base = 0; base < tile_cnt; base += BATCH_SIZE)
                {
                    const std::uint32_t count = (tile_cnt - base < BATCH_SIZE) ? (tile_cnt - base) : BATCH_SIZE;
                    if (base != 0)
                    {
                        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
                            tensor_shape, 0 /* acc_to_dest */);
                    }

                    for (std::uint32_t j = 0; j < count; ++j)
                    {
                        _llk_math_rmsnorm_clear_product_tile_<DST_CAPACITY, is_fp32_dest_acc_en>(j);
                        _llk_math_eltwise_binary_<
                            EltwiseBinaryType::ELWMUL,
                            BroadcastType::NONE,
                            DST_SYNC,
                            is_fp32_dest_acc_en,
                            MATH_FIDELITY,
                            EltwiseBinaryReuseDestType::NONE>(tensor_shape, j, true /* clear_fp32_dst_acc */);
                    }

                    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();
                    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DST_INDEX);
                    _llk_math_eltwise_unary_sfpu_params_(
                        ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, REDUCE_SCALER);
                    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(DST_INDEX);
                    _llk_math_eltwise_unary_sfpu_params_(
                        ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, 0.0f);
                    _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
                    for (std::uint32_t j = 1; j < count; ++j)
                    {
                        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(j);
                        _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
                    }
                    _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();
                    _llk_math_mul_reduce_scalar_clear_dvalid_();
                    _llk_math_eltwise_binary_sfpu_params_(
                        ckernel::sfpu::calculate_sfpu_binary<false /* APPROX */, BinaryOp::ADD, 8 /* ITERATIONS */, is_fp32_dest_acc_en>,
                        ACCUMULATOR,
                        DST_INDEX,
                        ACCUMULATOR,
                        VectorMode::RC);
                }

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;

    {
        START_PERF_MEASURE("INIT")
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
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(ACCUMULATOR, L1_ADDRESS(params.buffer_Res[0]));
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    // mul_reduce_scalar_uninit
    _llk_pack_reduce_mask_clear_();
}

#endif
