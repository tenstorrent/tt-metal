// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// mul_reduce_scalar_chunked_tile (api/compute/experimental/rmsnorm.h) expanded into its _llk_* calls (Blackhole only).
// CHUNK_SIZE is the API's dst_capacity: slot CHUNK_SIZE - 1 accumulates, the others stage a batch of products.

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

    // mul_reduce_scalar_init
    _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::None);

    const std::uint32_t tile_cnt   = params.TILE_CNT;
    const std::uint32_t batch_size = params.CHUNK_SIZE - 1;

    for (std::uint32_t base = 0; base < tile_cnt; base += batch_size)
    {
        const std::uint32_t count = (tile_cnt - base < batch_size) ? (tile_cnt - base) : batch_size;
        for (std::uint32_t j = 0; j < count; ++j)
        {
            _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[base + j]), L1_ADDRESS(params.buffer_B[base + j]));
        }
        _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_fill.h"

// Scaler multiplier applied to the reduction (matches the Compute API default).
static constexpr float REDUCE_SCALER = 1.0f;
// Row of the accumulator that collects the column sums; the scalar reduce writes its zeroed row 0.
static constexpr std::uint32_t SUM_ROW = 4;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t tile_cnt            = params.TILE_CNT;
    const std::uint32_t batch_size          = params.CHUNK_SIZE - 1;
    const std::uint32_t accumulator         = batch_size;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};
    // The clear's capacity argument only bounds its slot assert; CHUNK_SIZE is a runtime parameter here.
    constexpr std::uint32_t max_dst_tiles = get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();

    // compute_kernel_hw_startup
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_init_once_();

    // mul_reduce_scalar_init, fill_tile_init, add_binary_tile_init
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        tensor_shape, 0 /* acc_to_dest */);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::fill>();
    SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (false, BinaryOp::ADD));

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Poison every slot, as a fresh acquire need not be zero: the product clears and the accumulator fill must cover it.
    for (std::uint32_t slot = 0; slot < params.CHUNK_SIZE; ++slot)
    {
        _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_fill_<false /* APPROX */, 8 /* ITERATIONS */>, slot, VectorMode::RC, 7.0f);
    }
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_fill_<false /* APPROX */, 4 /* ITERATIONS */>, accumulator, VectorMode::RC_custom, 0.0f);

    for (std::uint32_t base = 0; base < tile_cnt; base += batch_size)
    {
        const std::uint32_t count = (tile_cnt - base < batch_size) ? (tile_cnt - base) : batch_size;
        if (base > 0)
        {
            eltwise_binary_configure_addrmod<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY>();
        }
        for (std::uint32_t j = 0; j < count; ++j)
        {
            _llk_math_rmsnorm_clear_product_tile_<max_dst_tiles, is_fp32_dest_acc_en>(j);
            _llk_math_eltwise_binary_<
                EltwiseBinaryType::ELWMUL,
                BroadcastType::NONE,
                DST_SYNC,
                is_fp32_dest_acc_en,
                MATH_FIDELITY,
                EltwiseBinaryReuseDestType::NONE>(tensor_shape, j, true /* clear_fp32_dst_acc */);
        }

        _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0);
        _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, 0, VectorMode::RC_custom, REDUCE_SCALER);
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0);

        _llk_math_mul_reduce_column_<MATH_FIDELITY, SUM_ROW>(accumulator, tensor_shape);
        for (std::uint32_t j = 1; j < count; ++j)
        {
            _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA, true>(j);
            _llk_math_mul_reduce_column_<MATH_FIDELITY, SUM_ROW>(accumulator, tensor_shape);
        }
        if (base + count >= tile_cnt)
        {
            _llk_math_mul_reduce_scalar_<MATH_FIDELITY, SUM_ROW>();
        }
        _llk_math_mul_reduce_scalar_clear_dvalid_();
    }

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

    const std::uint32_t tile_size   = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces   = tensor_shape.total_num_faces();
    const bool partial_face         = tensor_shape.face_r_dim < FACE_R_DIM;
    const std::uint32_t accumulator = params.CHUNK_SIZE - 1;

    // compute_kernel_hw_startup
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face);

    // No-src init: packer strides are owned by the hw-configure above.
    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);

    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();

    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(accumulator, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    // mul_reduce_scalar_uninit
    _llk_pack_reduce_mask_clear_();
}

#endif
