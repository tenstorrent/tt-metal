// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the gate path of the Blackhole generalized MoE gate, one token per iteration. PERF_STAGE 0 runs the
// binary front end only, 1 adds sum_top2 and step0, 2 the whole gate; MATH_ISOLATE keeps the real unpack, drops the pack.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t NUM_DEST_TILES = 4;
constexpr std::uint32_t SCORES_TILE    = 0;
constexpr std::uint32_t IDS_TILE       = 1;
constexpr std::uint32_t KEYS_TILE      = 2;
constexpr std::uint32_t ID_FORMAT      = ckernel::to_underlying(DataFormat::UInt16);
constexpr bool STEP2_RUNS              = (PERF_STAGE == 2);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

constexpr auto GATE_UNPACK_TRANSPOSE = GMG_TRANSPOSE_OF_FACES ? ckernel::Transpose::Both : ckernel::Transpose::IntraFace;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const auto tensor_shape         = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces);
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                    ID_FORMAT, ID_FORMAT, params.TILE_SIZE_UNPACK_A);
                _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, tensor_shape, ID_FORMAT, ID_FORMAT);
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_A[1]), ID_FORMAT, ID_FORMAT);
                _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                    formats.unpack_A_src, formats.unpack_A_dst, params.TILE_SIZE_UNPACK_A);
                _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, GATE_UNPACK_TRANSPOSE);
                _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]));
                _llk_unpack_set_srcb_dummy_valid_();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"

using namespace ckernel;

#include "experimental/llk_math_generalized_moe_gate_eltwise_binary.h"
#include "experimental/llk_math_generalized_moe_gate_transpose_dest_single_face.h"
#include "experimental/llk_sfpu/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

constexpr GeneralizedMoeGateEltwiseBinaryMode BINARY_MODE =
    GMG_RELOAD ? GeneralizedMoeGateEltwiseBinaryMode::RELOAD : GeneralizedMoeGateEltwiseBinaryMode::COPY;

#define GMG_SFPU_CALL(FN, TEMPLATES, ...) \
    SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, FN, TEMPLATES, 0 /* dst_index */, VectorMode::RC_custom, ##__VA_ARGS__)

static inline void run_gate()
{
    GMG_SFPU_CALL(generalized_moe_gate_sum_top2, (APPROX_MODE, is_fp32_dest_acc_en));

    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_<false>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, false>();

    if constexpr (PERF_STAGE == 1)
    {
        return;
    }

    if constexpr (GMG_GROUPED)
    {
        GMG_SFPU_CALL(generalized_moe_gate_sort_top4_groups, (APPROX_MODE, is_fp32_dest_acc_en));
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_<false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, false>();
        GMG_SFPU_CALL(generalized_moe_gate_top8, (APPROX_MODE, is_fp32_dest_acc_en), GMG_EPS, GMG_SCALE);
    }
    else
    {
        _llk_math_generalized_moe_gate_copy4rows_init_<4, 8, false, 16>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false>();

        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<0, 0, false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, false>();
        GMG_SFPU_CALL(generalized_moe_gate_merge4_top8, (APPROX_MODE, is_fp32_dest_acc_en, 0, 0, 2));

        _llk_math_generalized_moe_gate_copy4rows_init_<0, 12, false, 20>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false>();
        _llk_math_generalized_moe_gate_copy4rows_init_<8, 4, false, 24>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false>();

        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<4, 0, false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, false>();
        GMG_SFPU_CALL(generalized_moe_gate_merge4_top8, (APPROX_MODE, is_fp32_dest_acc_en, 0, 4, 6));

        _llk_math_generalized_moe_gate_copy4rows_init_<12, 0, false, 28>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false>();

        GMG_SFPU_CALL(generalized_moe_gate_finalize_ungrouped, (APPROX_MODE, is_fp32_dest_acc_en, GMG_TOPK, GMG_SOFTMAX), GMG_EPS, GMG_SCALE);
    }

    _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_<false, GMG_OUTPUT_TILES>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, false>();
}

static inline void token_math(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(ID_FORMAT);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        params.num_faces, ID_FORMAT);
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        IDS_TILE, ID_FORMAT, ID_FORMAT);
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(formats.math);

    _llk_math_generalized_moe_gate_eltwise_binary_init_<ELTWISE_BINARY_OP, BINARY_MODE, MATH_FIDELITY>(params.num_faces, ACC_TO_DEST);
    _llk_math_generalized_moe_gate_eltwise_binary_<ELTWISE_BINARY_OP, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY>(params.num_faces, 0);

    _llk_math_generalized_moe_gate_transpose_dest_single_face_common_init_<false>();
    SFPU_UNARY_INIT_FN(unused, sfpu::generalized_moe_gate_topk_init, (APPROX_MODE, is_fp32_dest_acc_en));

    if constexpr (PERF_STAGE >= 1)
    {
        run_gate();
    }
    if constexpr (!STEP2_RUNS)
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                token_math(params);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<dest_sync>();
                token_math(params);
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            ID_FORMAT, ID_FORMAT, params.TILE_SIZE_PACK, FACE_R_DIM, TILE_C_DIM, params.num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false>(ID_FORMAT, FACE_R_DIM, TILE_C_DIM, params.num_faces);
        _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t tile = 0; tile < NUM_DEST_TILES; ++tile)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
