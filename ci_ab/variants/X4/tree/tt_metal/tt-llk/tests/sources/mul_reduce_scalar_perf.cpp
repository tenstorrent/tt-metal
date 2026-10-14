// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the mul_reduce_scalar row; ROW_TILES > 0 compiles the row length in, as the API runs a constant count.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

using namespace ckernel;

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC        = DstSync::SyncHalf;
// fl3 scratch: ROW_TILES = 100 * twin + row length
constexpr std::uint32_t RT   = ROW_TILES % 100;
constexpr std::uint32_t TWIN = ROW_TILES / 100;
static constexpr std::uint32_t DST_INDEX = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const auto& TILE_CNT            = params.TILE_CNT;
    const auto& num_faces_r_dim_A   = params.num_faces_r_dim_A;
    const auto& num_faces_c_dim_A   = params.num_faces_c_dim_A;
    const auto& buffer_A            = params.buffer_A;
    const auto& buffer_B            = params.buffer_B;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            tensor_shape.face_r_dim,
            tensor_shape.face_r_dim,
            num_faces,
            num_faces);
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
                for (std::uint32_t i = 0; i < TILE_CNT * num_faces; ++i)
                {
                    _perf_unpack_set_valid(ckernel::SrcA);
                    _perf_unpack_set_valid(ckernel::SrcB);
                }
                _perf_unpack_set_valid(ckernel::SrcA);
                _perf_unpack_set_valid(ckernel::SrcB);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(buffer_A[i]), L1_ADDRESS(buffer_B[i]));
                }
                _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_fill.h"

static constexpr float REDUCE_SCALER = 1.0f;

template <std::uint32_t row_tiles>
__attribute__((always_inline)) inline void reduce_later_tiles(const ckernel::TensorShape& tensor_shape)
{
#pragma GCC unroll 8
    for (std::uint32_t i = 1; i < row_tiles; ++i)
    {
        _llk_math_mul_reduce_scalar_move_product_<true /* share_slots */>(i, row_tiles, tensor_shape);
        _llk_math_mul_reduce_column_<MATH_FIDELITY, false /* tile_setup */>(DST_INDEX, tensor_shape);
    }
}

// One MOVD2B: GAPOOL reads SrcB rows 0 to 3, and only row 0 reaches the result.
inline void fl3_scaler_one_move()
{
    math::srcb_bank_wait();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_BD);
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
}

__attribute__((always_inline)) inline void row_math(const std::uint32_t runtime_tile_cnt, const ckernel::TensorShape& tensor_shape)
{
    const std::uint32_t tile_cnt = RT > 0 ? RT : runtime_tile_cnt;
    if constexpr (TWIN == 13)
    {
        _llk_math_mul_reduce_scalar_interleaved_init_<is_fp32_dest_acc_en>();
#pragma GCC unroll 8
        for (std::uint32_t i = 0; i < tile_cnt; ++i)
        {
            _llk_math_mul_reduce_scalar_interleaved_tile_(i, tensor_shape, REDUCE_SCALER);
        }
        _llk_math_mul_reduce_scalar_interleaved_tail_(tile_cnt, tensor_shape);
        return;
    }
    if constexpr (TWIN == 9 || TWIN == 10)
    {
        // the init without its MOP: the multiply below issues the MOP's instructions itself
        eltwise_binary_configure_addrmod<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY>();
        TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
        math::reset_counters(p_setrwc::SET_ABD_F);
        math::_configure_default_zero_flag_state_();
    }
    else if constexpr (TWIN != 4)
    {
        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
            tensor_shape, 0 /* acc_to_dest */);
    }
    if constexpr (TWIN == 5)
    {
        const std::uint32_t nf = tensor_shape.total_num_faces();
        for (std::uint32_t i = 0; i < tile_cnt * nf; ++i)
        {
            _perf_math_clear_valid(ckernel::SrcA);
            _perf_math_clear_valid(ckernel::SrcB);
        }
    }
    else if constexpr (TWIN == 9 || TWIN == 10)
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
        for (std::uint32_t i = 0; i < tile_cnt; ++i)
        {
#pragma GCC unroll 4
            for (std::uint32_t f = 0; f < 4; ++f)
            {
                TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0);
                TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0);
                TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);
            }
        }
        math::clear_dst_reg_addr();
    }
    else if constexpr (TWIN == 6 || TWIN == 8)
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
        for (std::uint32_t i = 0; i < tile_cnt; ++i)
        {
            ckernel_template::run();
        }
        math::clear_dst_reg_addr();
    }
    else if constexpr (TWIN == 11 || TWIN == 12)
    {
        for (std::uint32_t i = 0; i < tile_cnt; ++i)
        {
            _llk_math_eltwise_binary_<
                EltwiseBinaryType::ELWMUL,
                BroadcastType::NONE,
                DST_SYNC,
                is_fp32_dest_acc_en,
                MATH_FIDELITY,
                EltwiseBinaryReuseDestType::NONE>(tensor_shape, 0, true /* clear_fp32_dst_acc */);
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < tile_cnt; ++i)
        {
            if (_llk_math_mul_reduce_scalar_shares_slot_(tensor_shape))
            {
                _llk_math_mul_reduce_scalar_mul_half_slot_<MATH_FIDELITY>(tensor_shape, i);
                continue;
            }
            _llk_math_eltwise_binary_<
                EltwiseBinaryType::ELWMUL,
                BroadcastType::NONE,
                DST_SYNC,
                is_fp32_dest_acc_en,
                MATH_FIDELITY,
                EltwiseBinaryReuseDestType::NONE>(tensor_shape, i, true /* clear_fp32_dst_acc */);
        }
    }
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();
    _llk_math_mul_reduce_scalar_move_product_<TWIN != 11 && TWIN != 12 /* share_slots */>(DST_INDEX, (TWIN == 11 || TWIN == 12) ? 1 : tile_cnt, tensor_shape);
    if constexpr (TWIN != 1)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, REDUCE_SCALER);
    }
    if constexpr (TWIN == 7 || TWIN == 8 || TWIN == 10 || TWIN == 12)
    {
        fl3_scaler_one_move();
    }
    else
    {
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(DST_INDEX);
    }
    _llk_math_mul_reduce_scalar_clear_pool_face_<is_fp32_dest_acc_en>(DST_INDEX);
    _llk_math_mul_reduce_column_<MATH_FIDELITY, false /* tile_setup */>(DST_INDEX, tensor_shape);
    if constexpr (TWIN != 3 && TWIN != 11 && TWIN != 12)
    {
        if constexpr (RT > 0)
        {
            reduce_later_tiles<RT>(tensor_shape);
        }
        else
        {
            for (std::uint32_t i = 1; i < tile_cnt; ++i)
            {
                _llk_math_mul_reduce_scalar_move_product_<true /* share_slots */>(i, tile_cnt, tensor_shape);
                _llk_math_mul_reduce_column_<MATH_FIDELITY, false /* tile_setup */>(DST_INDEX, tensor_shape);
            }
        }
    }
    if constexpr (TWIN != 2)
    {
        _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();
    }
    _llk_math_mul_reduce_scalar_clear_dvalid_();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const auto& TILE_CNT            = params.TILE_CNT;
    const auto& num_faces_r_dim_A   = params.num_faces_r_dim_A;
    const auto& num_faces_c_dim_A   = params.num_faces_c_dim_A;
#endif
    const std::uint32_t tile_cnt = TILE_CNT;
    // A compiled row length compiles its tile shape in too, as the API's operand shape is a constant.
    const ckernel::TensorShape tensor_shape =
        RT > 0 ? ckernel::TensorShape {static_cast<std::uint8_t>(FACE_R_DIM), static_cast<std::uint8_t>(FACE_C_DIM), ROW_FACES_R_DIM, 2}
                      : ckernel::TensorShape {
                            static_cast<std::uint8_t>(FACE_R_DIM),
                            static_cast<std::uint8_t>(FACE_C_DIM),
                            static_cast<std::uint8_t>(num_faces_r_dim_A),
                            static_cast<std::uint8_t>(num_faces_c_dim_A)};
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_sfpu_init_once_();
        if constexpr (TWIN == 4)
        {
            _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
                tensor_shape, 0 /* acc_to_dest */);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < tile_cnt * num_faces; ++i)
                {
                    _perf_math_clear_valid(ckernel::SrcA);
                    _perf_math_clear_valid(ckernel::SrcB);
                }
                _perf_math_clear_valid(ckernel::SrcA);
                _perf_math_clear_valid(ckernel::SrcB);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                row_math(tile_cnt, tensor_shape);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                row_math(tile_cnt, tensor_shape);
                _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const auto& num_faces_r_dim_A   = params.num_faces_r_dim_A;
    const auto& num_faces_c_dim_A   = params.num_faces_c_dim_A;
    const auto& buffer_Res          = params.buffer_Res;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face);
        _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
            formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
        _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(buffer_Res[0]));
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(buffer_Res[0]));
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
    _llk_pack_reduce_mask_clear_();
}

#endif
