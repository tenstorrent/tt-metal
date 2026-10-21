// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Short-face source clearing (Blackhole only).
//
// A tile whose faces are shorter than 16 rows unpacks only its own rows of each SrcA/SrcB face, but a
// datacopy moves all 16 rows of a face and ELWMUL reads at least 8. The unpackers must zero the rest, or
// the reduce that follows sums whatever the previous op left in the source banks.
//
// The kernel runs five sum-to-scalar ops back to back, each packing its scalar into its own result tile:
//   op 0: full 32x32 tiles of poison (finite, NaN or Inf) that fill every row of both banks of SrcA and SrcB
//   op 1: short tiles (SHORT_FACE_R_DIM rows per face, 1x2 faces)
//   op 2: the same short tiles again, after op 1's reduce tail left its own values in the banks
//   op 3: full-face 16x32 tiles (control)
//   op 4: the short tiles again, after op 3 filled every row with finite data
// Each op reads TILE_CNT tiles from its own slots of buffer_A (and buffer_B). MUL_PATH selects
// mul_reduce_scalar (unpack AB + ELWMUL) over sum_reduce_scalar (unpack A + datacopy).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC        = DstSync::SyncHalf;
static constexpr std::uint32_t DST_INDEX = 0;
static constexpr std::uint32_t NUM_OPS   = 5;
static constexpr float REDUCE_SCALER     = 1.0f;

namespace
{
/** @brief Tile shape that op `op` reads: op 0 full 32x32, op 3 full-face 16x32, the others short 1x2 faces. */
inline ckernel::TensorShape op_shape(const std::uint32_t op, const std::uint32_t short_face_r_dim)
{
    if (op == 0)
    {
        return {FACE_R_DIM, FACE_C_DIM, 2, 2};
    }
    const std::uint32_t face_r_dim = op == 3 ? FACE_R_DIM : short_face_r_dim;
    return {static_cast<std::uint8_t>(face_r_dim), FACE_C_DIM, 1, 2};
}
} // namespace

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);

    for (std::uint32_t op = 0; op < NUM_OPS; ++op)
    {
        const ckernel::TensorShape shape = op_shape(op, params.SHORT_FACE_R_DIM);
        _llk_unpack_reconfig_tile_shape_srca_(0, shape.face_r_dim, shape.total_num_faces());
        _llk_unpack_reconfig_tile_shape_srcb_(0, shape.face_r_dim, shape.total_num_faces());

        const std::uint32_t first = op * params.TILE_CNT;
        if constexpr (MUL_PATH)
        {
            _llk_unpack_AB_init_<BroadcastType::NONE>(shape, ckernel::Transpose::None);
            for (std::uint32_t i = first; i < first + params.TILE_CNT; ++i)
            {
                _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
            }
        }
        else
        {
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, shape, formats.unpack_A_src, formats.unpack_A_dst);
            for (std::uint32_t i = first; i < first + params.TILE_CNT; ++i)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                    L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_fill.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_init_once_();

    for (std::uint32_t op = 0; op < NUM_OPS; ++op)
    {
        const ckernel::TensorShape shape = op_shape(op, params.SHORT_FACE_R_DIM);
        const std::uint32_t num_faces    = shape.total_num_faces();

        _llk_math_wait_for_dest_available_<DST_SYNC>();

        // Product or copy i goes to DEST[i].
        if constexpr (MUL_PATH)
        {
            _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
                shape, 0 /* acc_to_dest */);
            for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
            {
                _llk_math_eltwise_binary_<
                    EltwiseBinaryType::ELWMUL,
                    BroadcastType::NONE,
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    MATH_FIDELITY,
                    EltwiseBinaryReuseDestType::NONE>(shape, i, true /* clear_fp32_dst_acc */);
            }
        }
        else
        {
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(num_faces, formats.math);
            for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
            {
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                    i, formats.math, formats.math, num_faces);
            }
        }

        // The mul_reduce_scalar tail: column-reduce every tile into DEST[0], then collapse it to [0].
        _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DST_INDEX);
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, REDUCE_SCALER);
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(DST_INDEX);
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, 0.0f /* clear DEST[0] */);
        _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, shape);
        for (std::uint32_t i = 1; i < params.TILE_CNT; ++i)
        {
            _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i);
            _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, shape);
        }
        _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();
        _llk_math_mul_reduce_scalar_clear_dvalid_();

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Every op packs one 32x32 tile whose element [0] holds its scalar.
    constexpr ckernel::TensorShape shape = {FACE_R_DIM, FACE_C_DIM, 2, 2};
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, shape.total_tensor_size(), shape.face_r_dim, shape.total_col_dim(), shape.total_num_faces(), false);
    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats.pack_src, shape.face_r_dim, shape.total_col_dim(), shape.total_num_faces(), 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    for (std::uint32_t op = 0; op < NUM_OPS; ++op)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[op]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }

    _llk_pack_reduce_mask_clear_();
}

#endif
