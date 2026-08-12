// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* deepseek_moe_gate LLK test.
   GATE mirrors the call sequence in compute_kernel_api/deepseek_moe_gate.h, in both its
   enable_sigmoid forms.
   BINARY drives the deepseek_moe_gate eltwise-binary FPU front-end alone.
   MOVE runs one transpose-dest MOP on a Dest image the test writes itself.
*/

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// The kernel doesn't support 32-bit dest.
constexpr bool IS_32BIT = false;

constexpr int MODE_GATE   = 0;
constexpr int MODE_BINARY = 1;
constexpr int MODE_MOVE   = 2;

constexpr int MOVE_STEP0 = 0;
constexpr int MOVE_STEP1 = 1;
constexpr int MOVE_STEP2 = 2;

// buffer_A is the Dest image, buffer_B the binary's SrcB operand.
// Every mode packs all four back out.
constexpr std::uint32_t NUM_DEST_TILES = 4;

// One Dest tile per region, in the order the SFPU walks them.
constexpr std::uint32_t SCORES_TILE = 0;
constexpr std::uint32_t IDS_TILE    = 1;
constexpr std::uint32_t KEYS_TILE   = 2;

// The id tile is uint16 both in L1 and in Dest, so it is unpacked under its own format.
constexpr std::uint32_t ID_FORMAT = ckernel::to_underlying(DataFormat::UInt16);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

constexpr auto GATE_UNPACK_TRANSPOSE = (DMG_MODE == MODE_GATE) ? ckernel::Transpose::Both : ckernel::Transpose::None;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const auto tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);

    // Every mode seeds Dest regions from raw L1 tiles before anything else, under the uint16 config
    // so nothing is reformatted on the way in.
    const auto unpack_raw_tile = [](const std::uint32_t l1_tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(l1_tile), ID_FORMAT, ID_FORMAT);
    };

    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
        ID_FORMAT, ID_FORMAT, params.TILE_SIZE_UNPACK_A);
    _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, ID_FORMAT, ID_FORMAT);

    if constexpr (DMG_MODE == MODE_GATE || DMG_MODE == MODE_BINARY)
    {
        unpack_raw_tile(params.buffer_A[1]); // expert ids
        if constexpr (DMG_MODE == MODE_BINARY)
        {
            unpack_raw_tile(params.buffer_A[2]); // keys
        }

        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
            formats.unpack_A_src, formats.unpack_A_dst, params.TILE_SIZE_UNPACK_A);

        if constexpr (DMG_MODE == MODE_GATE && DMG_SIGMOID)
        {
            _llk_unpack_A_init_<BroadcastType::NONE, true /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                1 /* transpose_of_faces */, 1 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
            _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);

            // The RELOAD variant takes SrcA back from Dest.
            _llk_unpack_A_init_<BroadcastType::NONE, true /* acc_to_dest */, EltwiseBinaryReuseDestType::DEST_TO_SRCA, unpack_to_dest>(
                0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
            _llk_unpack_A_<BroadcastType::NONE, true /* acc_to_dest */, EltwiseBinaryReuseDestType::DEST_TO_SRCA, unpack_to_dest>(
                L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
        }
        else
        {
            _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, GATE_UNPACK_TRANSPOSE);
            _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]));
        }
    }
    else
    {
        // MOVE
        for (std::uint32_t tile = 0; tile < NUM_DEST_TILES; ++tile)
        {
            unpack_raw_tile(params.buffer_A[tile]);
        }
    }

    if constexpr (DMG_MODE != MODE_BINARY)
    {
        _llk_unpack_set_srcb_dummy_valid_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"

using namespace ckernel;

// Required by topk.
#include "sfpu/ckernel_sfpu_converter.h"

// I know this looks awful.
// It's temporary (and so is the harness change that puts ttnn experimental on
// the include path) until this kernel is moved into Metal's experimental dir (#52837).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#ifdef ARCH_BLACKHOLE
#include "deepseek/moe/deepseek_moe_gate/device/kernel_includes/tt_llk/tt_llk_blackhole/llk_lib/llk_math_deepseek_moe_gate_eltwise_binary.h"
#include "deepseek/moe/deepseek_moe_gate/device/kernel_includes/tt_llk/tt_llk_blackhole/llk_lib/llk_math_deepseek_moe_gate_transpose_dest_single_face.h"
#include "deepseek/moe/deepseek_moe_gate/device/kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h"
#else
#include "deepseek/moe/deepseek_moe_gate/device/kernel_includes/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_deepseek_moe_gate_eltwise_binary.h"
#include "deepseek/moe/deepseek_moe_gate/device/kernel_includes/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_deepseek_moe_gate_transpose_dest_single_face.h"
#include "deepseek/moe/deepseek_moe_gate/device/kernel_includes/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h"
#endif
#pragma GCC diagnostic pop
#include "llk_sfpu/ckernel_sfpu_sigmoid.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

// The sigmoid front-end leaves its result in DEST, so the binary after it has to be RELOAD.
// That is the only combination the op ever instantiates RELOAD in.
constexpr DeepseekMoeGateEltwiseBinaryMode BINARY_MODE =
    (DMG_RELOAD || DMG_SIGMOID) ? DeepseekMoeGateEltwiseBinaryMode::RELOAD : DeepseekMoeGateEltwiseBinaryMode::COPY;

// One SFPU call on DEST tile 0; each gate functor walks its own region offsets from there.
#define DMG_SFPU_CALL(FN, TEMPLATES, ...) \
    SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, FN, TEMPLATES, 0 /* dst_index */, VectorMode::RC_custom, ##__VA_ARGS__)

// Reset dst_index to zero. The MOPs operate on whatever it was set to.
static inline void mop_dest_reset()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
}

static inline void run_gate()
{
    DMG_SFPU_CALL(deepseek_moe_gate_sum_top2, (APPROX_MODE, is_fp32_dest_acc_en));

    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_init_<IS_32BIT>();
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, IS_32BIT>();

    DMG_SFPU_CALL(deepseek_moe_gate_sort_top4_groups, (APPROX_MODE, is_fp32_dest_acc_en));

    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_init_<IS_32BIT>();
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, IS_32BIT>();

    DMG_SFPU_CALL(deepseek_moe_gate_top8, (APPROX_MODE, is_fp32_dest_acc_en), DMG_EPS, DMG_SCALE);

    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_init_<IS_32BIT>();
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, IS_32BIT>();
}

static inline void run_move()
{
    if constexpr (DMG_SUB_OP == MOVE_STEP0)
    {
        _llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_init_<IS_32BIT>();
        mop_dest_reset();
        _llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, IS_32BIT>();
    }
    else if constexpr (DMG_SUB_OP == MOVE_STEP1)
    {
        _llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_init_<IS_32BIT>();
        mop_dest_reset();
        _llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, IS_32BIT>();
    }
    else
    {
        _llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_init_<IS_32BIT>();
        mop_dest_reset();
        _llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, IS_32BIT>();
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<dest_sync>();

    {
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false /* to_from_int8 */>(ID_FORMAT);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(params.num_faces, ID_FORMAT);

        const auto copy_to_dest_tile = [](const std::uint32_t tile, const std::uint32_t format = ID_FORMAT)
        {
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, format, format);
        };

        // Each datacopy consumes one unpacked operand, so these have to match the unpacker's order:
        // the id tile (GATE), the score then key regions (BINARY), the whole image (MOVE).
        if constexpr (DMG_MODE == MODE_GATE)
        {
            copy_to_dest_tile(IDS_TILE);
        }
        else if constexpr (DMG_MODE == MODE_BINARY)
        {
            copy_to_dest_tile(SCORES_TILE); // RELOAD's SrcA
            copy_to_dest_tile(KEYS_TILE);   // ACC_TO_DEST's accumulator base
        }
        else
        {
            for (std::uint32_t tile = 0; tile < NUM_DEST_TILES; ++tile)
            {
                copy_to_dest_tile(tile);
            }
        }
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false /* to_from_int8 */>(formats.math);

        if constexpr (DMG_MODE == MODE_GATE && DMG_SIGMOID)
        {
            // The op's enable_sigmoid front-end: transpose_wh_tile then sigmoid_tile leave the activated
            // score in the score region, which the RELOAD binary below reads back through MOVD2A.
            _llk_math_eltwise_unary_datacopy_init_wrapper_<
                DataCopyType::A2D,
                is_fp32_dest_acc_en,
                BroadcastType::NONE,
                false /* is_int_fpu_en */,
                PackMode::Default>(params.num_faces, formats.math);
            copy_to_dest_tile(SCORES_TILE, formats.math);

            SFPU_UNARY_INIT_FN(sigmoid, sfpu::sigmoid_init, (false /* fast_and_approx */));
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_sigmoid,
                (false /* fast_and_approx */, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
                0,
                VectorMode::RC_custom);
        }
    }

    if constexpr (DMG_MODE == MODE_GATE || DMG_MODE == MODE_BINARY)
    {
        _llk_math_deepseek_moe_gate_eltwise_binary_init_<ELTWISE_BINARY_OP, BINARY_MODE, MATH_FIDELITY>(params.num_faces, ACC_TO_DEST);
        _llk_math_deepseek_moe_gate_eltwise_binary_<ELTWISE_BINARY_OP, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY>(
            params.num_faces, 0 /* dst_index */, true /* clear_fp32_dst_acc */);
    }

    if constexpr (DMG_MODE != MODE_BINARY)
    {
        _llk_math_deepseek_moe_gate_transpose_dest_single_face_common_init_<IS_32BIT>();
        SFPU_UNARY_INIT_FN(unused, sfpu::deepseek_moe_gate_topk_init, (APPROX_MODE, is_fp32_dest_acc_en));
    }

    if constexpr (DMG_MODE == MODE_GATE)
    {
        run_gate();
    }
    else if constexpr (DMG_MODE == MODE_MOVE)
    {
        run_move();
    }

    // step2 ends on a SETRWC CLR_AB. The paths that skip it have to hand the Src banks back themselves.
    constexpr bool step2_runs = (DMG_MODE == MODE_GATE) || (DMG_MODE == MODE_MOVE && DMG_SUB_OP == MOVE_STEP2);
    if constexpr (DMG_MODE != MODE_BINARY && !step2_runs)
    {
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD);
    }

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Every Dest tile is packed as uint16, as small ids are bf16 denormals which a float pack would flush.
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        ID_FORMAT, ID_FORMAT, params.TILE_SIZE_PACK, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(ID_FORMAT, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile = 0; tile < NUM_DEST_TILES; ++tile)
    {
        LLK_ASSERT(
            (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
