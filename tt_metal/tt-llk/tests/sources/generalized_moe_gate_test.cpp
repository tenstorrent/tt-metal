// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* generalized_moe_gate LLK test. GATE mirrors the call sequence in
   api/compute/experimental/generalized_moe_gate.h; BINARY drives its FPU front-end alone; MOVE and
   RUN work on a DEST image the test writes itself, one FPU MOP or SFPU op at a time or as the
   multi-block combine tail, so a contract can be checked without standing up the rest of the gate. */

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr int MODE_GATE   = 0; // Full gate, grouped or ungrouped.
constexpr int MODE_BINARY = 1; // The FPU binary front-end on its own.
constexpr int MODE_MOVE   = 2; // One transpose-dest / copy4rows MOP on a known DEST image.
constexpr int MODE_RUN    = 3; // SFPU run merges and placements on a known DEST image.

constexpr int MOVE_STEP0     = 0;
constexpr int MOVE_STEP1     = 1;
constexpr int MOVE_STEP1_HI  = 2;
constexpr int MOVE_STEP2     = 3;
constexpr int MOVE_COPY4ROWS = 4;

constexpr int RUN_MERGE4_TOP8       = 0;
constexpr int RUN_COPY_TOPK_RUN     = 1;
constexpr int RUN_PLACE_FIELD       = 2;
constexpr int RUN_MERGE16           = 3;
constexpr int RUN_COMBINE           = 4; // Combine tail, arriving run placed from intermediate.
constexpr int RUN_COMBINE_RELOCATED = 5; // Combine tail, arriving run relocated within DEST.

// buffer_A is the DEST image, buffer_B the binary's SrcB operand; every mode packs all four back out.
constexpr std::uint32_t NUM_DEST_TILES = 4;

// One DEST tile per region, in the order the SFPU's scores/indices/bias/interm offsets walk them.
constexpr std::uint32_t SCORES_TILE = 0;
constexpr std::uint32_t IDS_TILE    = 1;
constexpr std::uint32_t KEYS_TILE   = 2;

// The id tile is uint16 both in L1 and in DEST, so it is unpacked under its own format.
constexpr std::uint32_t ID_FORMAT = ckernel::to_underlying(DataFormat::UInt16);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const auto tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);

    if constexpr (GMG_MODE == MODE_GATE || GMG_MODE == MODE_BINARY)
    {
        // GATE uses the raw tile for the ids, mirroring the op's copy_tile before
        // generalized_moe_gate_init. BINARY takes two: buffer_A[1] seeds the score region so RELOAD
        // has a known SrcA to read back, buffer_A[2] the key region so ACC_TO_DEST accumulates onto
        // something the test picked rather than onto whatever DEST held.
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
            ID_FORMAT, ID_FORMAT, params.TILE_SIZE_UNPACK_A);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, ID_FORMAT, ID_FORMAT);
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[1]), ID_FORMAT, ID_FORMAT);
        if constexpr (GMG_MODE == MODE_BINARY)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[2]), ID_FORMAT, ID_FORMAT);
        }

        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
            formats.unpack_A_src, formats.unpack_A_dst, params.TILE_SIZE_UNPACK_A);
        _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::None);
        _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]));
    }
    else
    {
        // RUN starts from a DEST image the test builds, so every tile comes in raw under the
        // uint16 config.
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
            ID_FORMAT, ID_FORMAT, params.TILE_SIZE_UNPACK_A);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, ID_FORMAT, ID_FORMAT);
        for (std::uint32_t tile = 0; tile < NUM_DEST_TILES; ++tile)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[tile]), ID_FORMAT, ID_FORMAT);
        }
    }

    // The binary's mop end op (SETRWC CLR_AB) clears the SrcB valid, so this has to follow the
    // unpacks above.
    if constexpr (GMG_MODE != MODE_BINARY)
    {
        _llk_unpack_set_srcb_dummy_valid_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"

using namespace ckernel;

#define DST_SYNC_MODE  dest_sync
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "experimental/llk_sfpu/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#undef DST_SYNC_MODE
#undef DST_ACCUM_MODE

#include "experimental/llk_math_generalized_moe_gate_eltwise_binary.h"
#include "experimental/llk_math_generalized_moe_gate_transpose_dest_single_face.h"

// step2 ends on a SETRWC CLR_AB and is the only op here that does. The paths that skip it have to
// hand the Src banks back themselves, or the next kernel launched on this core stalls on its first
// UNPACR. BINARY is exempt: it issues no dummy valid, and the binary's own mop end op clears.
constexpr bool STEP2_RUNS = (GMG_MODE == MODE_GATE && !(GMG_PRODUCE_RUN && !GMG_GROUPED)) || (GMG_MODE == MODE_MOVE && GMG_SUB_OP == MOVE_STEP2);

constexpr GeneralizedMoeGateEltwiseBinaryMode BINARY_MODE =
    GMG_RELOAD ? GeneralizedMoeGateEltwiseBinaryMode::RELOAD : GeneralizedMoeGateEltwiseBinaryMode::COPY;

// One SFPU call on DEST tile 0; each gate functor walks its own region offsets from there.
#define GMG_SFPU_CALL(FN, TEMPLATES, ...) SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, FN, TEMPLATES, 0, VectorMode::RC_custom, ##__VA_ARGS__)

static inline void run_gate()
{
    GMG_SFPU_CALL(generalized_moe_gate_sum_top2, (APPROX_MODE, is_fp32_dest_acc_en));

    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_<false>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, false>();

    if constexpr (GMG_GROUPED)
    {
        GMG_SFPU_CALL(generalized_moe_gate_sort_top4_groups, (APPROX_MODE, is_fp32_dest_acc_en));
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_<false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, false>();
        GMG_SFPU_CALL(generalized_moe_gate_top8, (APPROX_MODE, is_fp32_dest_acc_en), GMG_EPS, GMG_SCALE);
    }
    else
    {
        // Groups 4-7 are parked in rows 8-11 while the low half is merged, because step1_hi with
        // d2b_dst=0 writes its run over rows 0-7. Each copy4rows takes its own SrcB window so a
        // later MOVB2D cannot read the previous copy's leftover.
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

        if constexpr (GMG_PRODUCE_RUN)
        {
            GMG_SFPU_CALL(generalized_moe_gate_merge16_to_run, (APPROX_MODE, is_fp32_dest_acc_en, GMG_TO_LO, GMG_TO_HI, GMG_IDX_OFFSET));
        }
        else
        {
            GMG_SFPU_CALL(generalized_moe_gate_finalize_ungrouped, (APPROX_MODE, is_fp32_dest_acc_en, GMG_TOPK, GMG_SOFTMAX), GMG_EPS, GMG_SCALE);
        }
    }

    // produce_run leaves the run in the SFPU layout for a later combine to consume, so the output
    // transpose is the one step the run-producing path skips.
    if constexpr (!(GMG_PRODUCE_RUN && !GMG_GROUPED))
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_<false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, false>();
    }
}

static inline void run_move()
{
    if constexpr (GMG_SUB_OP == MOVE_STEP0)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_<false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, false>();
    }
    else if constexpr (GMG_SUB_OP == MOVE_STEP1)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_<false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, false>();
    }
    else if constexpr (GMG_SUB_OP == MOVE_STEP1_HI)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<GMG_D2B_DST, GMG_B2D_BASE, false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, false>();
    }
    else if constexpr (GMG_SUB_OP == MOVE_STEP2)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_<false>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, false>();
    }
    else
    {
        _llk_math_generalized_moe_gate_copy4rows_init_<GMG_ROW_SRC, GMG_ROW_DST, false, GMG_SRCB>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false>();
        if constexpr (GMG_SECOND_COPY)
        {
            _llk_math_generalized_moe_gate_copy4rows_init_<GMG_ROW_SRC_2, GMG_ROW_DST_2, false, GMG_SRCB_2>();
            _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false>();
        }
    }
}

static inline void run_placement()
{
    if constexpr (GMG_PRE_COPY4ROWS)
    {
        // An FPU MOP leaves the Dst RWC advanced by +64 per tile. The SFPU ops below each reset it
        // on entry; without a MOP in front of them that reset is never needed and never tested.
        _llk_math_generalized_moe_gate_copy4rows_init_<GMG_ROW_SRC, GMG_ROW_DST, false, GMG_SRCB>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false>();
    }

    if constexpr (GMG_SUB_OP == RUN_MERGE4_TOP8)
    {
        GMG_SFPU_CALL(generalized_moe_gate_merge4_top8, (APPROX_MODE, is_fp32_dest_acc_en, GMG_READ_BASE, GMG_TO_LO, GMG_TO_HI));
    }
    else if constexpr (GMG_SUB_OP == RUN_COPY_TOPK_RUN)
    {
        GMG_SFPU_CALL(generalized_moe_gate_copy_topk_run, (APPROX_MODE, is_fp32_dest_acc_en, GMG_FROM_LO, GMG_FROM_HI, GMG_TO_LO, GMG_TO_HI));
    }
    else if constexpr (GMG_SUB_OP == RUN_PLACE_FIELD)
    {
        GMG_SFPU_CALL(
            generalized_moe_gate_place_field_from_interm, (APPROX_MODE, is_fp32_dest_acc_en, GMG_FIELD, GMG_FROM_LO, GMG_FROM_HI, GMG_TO_LO, GMG_TO_HI));
    }
    else if constexpr (GMG_SUB_OP == RUN_MERGE16)
    {
        GMG_SFPU_CALL(generalized_moe_gate_merge16_to_run, (APPROX_MODE, is_fp32_dest_acc_en, GMG_TO_LO, GMG_TO_HI, GMG_IDX_OFFSET));
    }
    else if constexpr (GMG_SUB_OP == RUN_COMBINE_RELOCATED)
    {
        // The same combine, but the arriving run is already in DEST at {8,10} and reaches the merge
        // slot by relocation instead. Whether a relocated run is still a run the merge accepts is
        // the run format's contract, and copy_topk_run's own test only checks that cells moved.
        GMG_SFPU_CALL(generalized_moe_gate_copy_topk_run, (APPROX_MODE, is_fp32_dest_acc_en, 8, 10, 4, 6));
        GMG_SFPU_CALL(generalized_moe_gate_merge16_to_run, (APPROX_MODE, is_fp32_dest_acc_en, GMG_TO_LO, GMG_TO_HI, GMG_IDX_OFFSET));
    }
    else
    {
        // The multi-block combine tail. One block's run is already resident at {0,2}; the other
        // arrives one field at a time through the intermediate region, then the two merge.
        //
        // The placement lands at {4,6} and the merge reads {0,2}+{4,6} because merge16 hardcodes
        // those four offsets, so neither is a free parameter. The three source pairs are spread
        // over the intermediate region only because a test kernel cannot re-unpack between fields
        // the way the op's copy_tile does; place_field's own test sweeps the source offsets.
        GMG_SFPU_CALL(generalized_moe_gate_place_field_from_interm, (APPROX_MODE, is_fp32_dest_acc_en, 0, 0, 2, 4, 6));
        GMG_SFPU_CALL(generalized_moe_gate_place_field_from_interm, (APPROX_MODE, is_fp32_dest_acc_en, 1, 4, 6, 4, 6));
        GMG_SFPU_CALL(generalized_moe_gate_place_field_from_interm, (APPROX_MODE, is_fp32_dest_acc_en, 2, 8, 10, 4, 6));
        GMG_SFPU_CALL(generalized_moe_gate_merge16_to_run, (APPROX_MODE, is_fp32_dest_acc_en, GMG_TO_LO, GMG_TO_HI, GMG_IDX_OFFSET));
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

        const auto copy_to_dest_tile = [](const std::uint32_t tile)
        {
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, ID_FORMAT, ID_FORMAT);
        };

        // Each datacopy consumes one unpacked operand, so these have to match the unpacker's order:
        // the id tile (GATE), the score then key regions (BINARY), the whole image (MOVE, RUN).
        if constexpr (GMG_MODE == MODE_GATE)
        {
            copy_to_dest_tile(IDS_TILE);
        }
        else if constexpr (GMG_MODE == MODE_BINARY)
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
    }

    if constexpr (GMG_MODE == MODE_GATE || GMG_MODE == MODE_BINARY)
    {
        _llk_math_generalized_moe_gate_eltwise_binary_init_<ELTWISE_BINARY_OP, BINARY_MODE, MATH_FIDELITY>(params.num_faces, ACC_TO_DEST);
        _llk_math_generalized_moe_gate_eltwise_binary_<ELTWISE_BINARY_OP, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY>(params.num_faces, 0 /* dst_index */);
    }

    if constexpr (GMG_MODE != MODE_BINARY)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_common_init_<false>();
        SFPU_UNARY_INIT_FN(unused, sfpu::generalized_moe_gate_topk_init, (APPROX_MODE, is_fp32_dest_acc_en));
    }

    if constexpr (GMG_MODE == MODE_GATE)
    {
        run_gate();
    }
    else if constexpr (GMG_MODE == MODE_MOVE)
    {
        run_move();
    }
    else if constexpr (GMG_MODE == MODE_RUN)
    {
        run_placement();
    }

    if constexpr (GMG_MODE != MODE_BINARY && !STEP2_RUNS)
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
    // Every DEST tile is packed as uint16: the test compares bit patterns, so nothing may be
    // reformatted on the way out. Small ids are bf16 denormals and a float pack path would flush them.
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
