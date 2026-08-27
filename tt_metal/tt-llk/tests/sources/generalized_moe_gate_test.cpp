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
constexpr int RUN_COMBINE_FINALIZE  = 6; // Combine tail through finalize + step2: the >256 output.

// buffer_A is the DEST image, buffer_B the binary's SrcB operand; every mode packs all four back out.
constexpr std::uint32_t NUM_DEST_TILES = 4;

// One DEST tile per region, in the order the SFPU's scores/indices/bias/intermediate offsets walk them.
constexpr std::uint32_t SCORES_TILE = 0;
constexpr std::uint32_t IDS_TILE    = 1;
constexpr std::uint32_t KEYS_TILE   = 2;

// The id tile is uint16 both in L1 and in DEST, so it is unpacked under its own format.
constexpr std::uint32_t ID_FORMAT = ckernel::to_underlying(DataFormat::UInt16);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

// generalized_moe_gate_init's non-sigmoid branch inits this unpack with ckernel::Transpose::Both,
// which sets Haloize on SEC0 only: SrcA (the payload) is transposed within the face and the faces
// are unpacked 0,2,1,3, while SrcB (the bias) arrives untransposed. So the gate scores payload.T
// against an untransposed bias, and since the id tile is unpacked separately with no transpose, the
// expert an id names is a transposed position. Driving Transpose::None here would leave the
// id-to-score association -- the thing most likely to break -- untested.
//
// BINARY stays at Transpose::None: it is here to pin the FPU mop's arithmetic across all four faces,
// and a transposed SrcA would only make its golden restate what the unpacker did.
constexpr auto GATE_UNPACK_TRANSPOSE = (GMG_MODE == MODE_GATE) ? ckernel::Transpose::Both : ckernel::Transpose::None;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const auto tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);

    // Each section feeds one DEST half. Under DstSync::Half the second lands at
    // get_dest_buffer_base() == DEST_REGISTER_HALF_SIZE, which is the only way to reach the upper
    // half at all — see the section loop on the math thread.
    for (std::uint32_t section = 0; section < GMG_SECTIONS; ++section)
    {
        if constexpr (GMG_MODE == MODE_GATE || GMG_MODE == MODE_BINARY)
        {
            // GATE uses the raw tile for the ids, mirroring the op's copy_tile before
            // generalized_moe_gate_init. BINARY takes two: buffer_A[1] seeds the score region so RELOAD
            // has a known SrcA to read back, buffer_A[2] the key region so ACC_TO_DEST accumulates onto
            // something the test picked rather than onto whatever DEST held.
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
                ID_FORMAT, ID_FORMAT, params.TILE_SIZE_UNPACK_A);
            _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, ID_FORMAT, ID_FORMAT);
            _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[1]), ID_FORMAT, ID_FORMAT);
            if constexpr (GMG_MODE == MODE_BINARY)
            {
                _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_A[2]), ID_FORMAT, ID_FORMAT);
            }

            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
                formats.unpack_A_src, formats.unpack_A_dst, params.TILE_SIZE_UNPACK_A);

            if constexpr (GMG_MODE == MODE_GATE && GMG_SIGMOID)
            {
                // transpose_wh_tile: the 32x32 transpose is the unpacker's, both flags set.
                //
                // acc_to_dest is true on the init and false on the call, which is the asymmetry
                // transpose_init/transpose_tile ship and neither value is arbitrary. The init ignores
                // it outright: the transpose_of_faces branch of the mop config never reads it. The
                // call does not -- acc_to_dest picks which unpacker's base address register receives
                // the L1 address, SEC0 (SrcA) when false and SEC1 (SrcB) when true. This mop issues
                // UNPACR on SrcA, so a true here would leave it unpacking whatever SEC0 last pointed
                // at, which is the id tile above.
                _llk_unpack_A_init_<BroadcastType::NONE, true /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    1 /* transpose_of_faces */, 1 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
                _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);

                // The RELOAD binary takes SrcA back from DEST, so the unpacker only feeds SrcB and
                // hands SrcA's bank over. This DEST_TO_SRCA reuse is the configuration the op ships
                // and the reason RELOAD needs the SRCA_VLD stall it carries.
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
            // RUN starts from a DEST image the test builds, so every tile comes in raw under the
            // uint16 config.
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false /* to_from_int8 */>(
                ID_FORMAT, ID_FORMAT, params.TILE_SIZE_UNPACK_A);
            _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, ID_FORMAT, ID_FORMAT);
            for (std::uint32_t tile = 0; tile < NUM_DEST_TILES; ++tile)
            {
                _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
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
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"

using namespace ckernel;

#include "experimental/llk_math_generalized_moe_gate_eltwise_binary.h"
#include "experimental/llk_math_generalized_moe_gate_transpose_dest_single_face.h"
#include "experimental/llk_sfpu/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

// step2 ends on a SETRWC CLR_AB and is the only op here that does. The paths that skip it have to
// hand the Src banks back themselves, or the next kernel launched on this core stalls on its first
// UNPACR. BINARY is exempt: it issues no dummy valid, and the binary's own mop end op clears.
constexpr bool STEP2_RUNS = (GMG_MODE == MODE_GATE && !(GMG_PRODUCE_RUN && !GMG_GROUPED)) || (GMG_MODE == MODE_MOVE && GMG_SUB_OP == MOVE_STEP2) ||
                            (GMG_MODE == MODE_RUN && GMG_SUB_OP == RUN_COMBINE_FINALIZE);

// The sigmoid front-end leaves its result in DEST, so the binary after it has to be RELOAD; that is
// the only combination the op ever instantiates RELOAD in.
constexpr GeneralizedMoeGateEltwiseBinaryMode BINARY_MODE =
    (GMG_RELOAD || GMG_SIGMOID) ? GeneralizedMoeGateEltwiseBinaryMode::RELOAD : GeneralizedMoeGateEltwiseBinaryMode::COPY;

// One SFPU call on DEST tile 0; each gate functor walks its own region offsets from there.
#define GMG_SFPU_CALL(FN, TEMPLATES, ...) \
    SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, FN, TEMPLATES, 0 /* dst_index */, VectorMode::RC_custom, ##__VA_ARGS__)

// The MOP runners take no dst_index, they address whatever tile DEST_TARGET_REG_CFG_MATH_Offset holds.
// In the op that is tile 0, because the eltwise binary ahead of them runs at dst_index 0 and leaves
// it there, which is why run_gate does not call this. MOVE and RUN skip the binary and reach the MOPs
// straight out of datacopies that walk all four tiles, so the offset would sit at tile 3. Put it back
// so those modes exercise the MOPs under the offset the op actually gives them.
static inline void mop_dest_reset()
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
}

// Determinism sanitize (test-only). The gate's answer is row 0, columns 0..7 of the SCORES and IDS
// tiles -- the op's documented output (generalized_moe_gate_nanobind.cpp: "Only the first k entries of
// row 0 are valid ... the rest of the tile is padding"), and all _gate_output reads. Every other
// packed lane (row 0 cols 8..15, rows 1..15, the KEYS/intermediate scratch tiles, faces 1..3) holds
// uninitialized-SFPU-LReg residue: the sort
// and the sum_top2 "replicate down the column" broadcast SFPSTORE full 32-lane rows whose non-rank
// lanes were never written this run, so they carry whatever the previous kernel left in the LReg file
// (see the op's own "junk lanes ... harmless" note). That residue is bit-reproducible run-to-run only
// because it is a fixed point once warm; run 0 (cold) differs, so the bit-exact re-run check flags it
// though the answer is stable. Rather than scrub every SFPU stage, keep only the answer here (row 0,
// cols 0..7 of SCORES and IDS) and zero every other DEST row the packer ships.
static inline void gmg_sanitize_scratch()
{
    // Address everything through sfpi dst_reg -- the one mapping proven to land on the right rows here
    // (raw SFP offsets / ZEROACC start mid-tile). dst_reg[k] maps to TTI address 2k, so each packed DEST
    // tile is 32 dst_reg rows and the 4 tiles span dst_reg[0..127]. A packed 16-col row is split across
    // TWO dst_reg rows -- even columns in dst_reg[2r], odd columns in dst_reg[2r+1] -- so the answer
    // (packed row 0 of SCORES/IDS) lives in dst_reg pairs {0,1} (SCORES) and {32,33} (IDS). Keep those
    // four; zero every other dst_reg row.
    constexpr int DREG_PER_TILE = ckernel::sfpu::dst_tile_offset / 2; // 32
    constexpr int SCORES_LO     = SCORES_TILE * DREG_PER_TILE;        // 0  (even cols of SCORES row 0)
    constexpr int SCORES_HI     = SCORES_LO + 1;                      // 1  (odd  cols)
    constexpr int IDS_LO        = IDS_TILE * DREG_PER_TILE;           // 32 (even cols of IDS row 0)
    constexpr int IDS_HI        = IDS_LO + 1;                         // 33 (odd  cols)
    constexpr int NUM_DREG      = NUM_DEST_TILES * DREG_PER_TILE;     // 128
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int k = 0; k < NUM_DREG; ++k)
    {
        if (k == SCORES_LO || k == SCORES_HI || k == IDS_LO || k == IDS_HI)
        {
            continue;
        }
        sfpi::dst_reg[k] = 0.0f;
    }
    // Mask each answer row-half to the top-8: with columns split even/odd, cols 0..7 are the first four
    // even and first four odd columns -> lanes 0..3 -> vConstTileId < 8; zero the residue past them.
    // (Same vFloat mod-0 raw-bit round-trip finalize_ungrouped uses for the idx row -- kept ids survive.)
    for (const int dreg : {SCORES_LO, SCORES_HI, IDS_LO, IDS_HI})
    {
        sfpi::vFloat v = sfpi::dst_reg[dreg];
        v_if (sfpi::vConstTileId >= 8)
        {
            v = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[dreg] = v;
    }
}

static inline void run_gate()
{
    GMG_SFPU_CALL(generalized_moe_gate_sum_top2, (APPROX_MODE, is_fp32_dest_acc_en));

    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_<false /* is_32bit */>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, false /* is_32bit */>();

    if constexpr (GMG_GROUPED)
    {
        GMG_SFPU_CALL(generalized_moe_gate_sort_top4_groups, (APPROX_MODE, is_fp32_dest_acc_en));
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_<false /* is_32bit */>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, false /* is_32bit */>();
        GMG_SFPU_CALL(generalized_moe_gate_top8, (APPROX_MODE, is_fp32_dest_acc_en), GMG_EPS, GMG_SCALE);
    }
    else
    {
        // Groups 4-7 are parked in rows 8-11 while the low half is merged, because step1_hi with
        // d2b_dst=0 writes its run over rows 0-7. Each copy4rows takes its own SrcB window so a
        // later MOVB2D cannot read the previous copy's leftover.
        _llk_math_generalized_moe_gate_copy4rows_init_<4 /* src */, 8 /* dst */, false /* is_32bit */, 16 /* srcb */>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false /* is_32bit */>();

        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<0 /* d2b_dst */, 0 /* b2d_base */, false /* is_32bit */>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, false /* is_32bit */>();
        GMG_SFPU_CALL(generalized_moe_gate_merge4_top8, (APPROX_MODE, is_fp32_dest_acc_en, 0 /* read_base */, 0 /* store_lo */, 2 /* store_hi */));

        _llk_math_generalized_moe_gate_copy4rows_init_<0 /* src */, 12 /* dst */, false /* is_32bit */, 20 /* srcb */>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false /* is_32bit */>();
        _llk_math_generalized_moe_gate_copy4rows_init_<8 /* src */, 4 /* dst */, false /* is_32bit */, 24 /* srcb */>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false /* is_32bit */>();

        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<4 /* d2b_dst */, 0 /* b2d_base */, false /* is_32bit */>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, false /* is_32bit */>();
        GMG_SFPU_CALL(generalized_moe_gate_merge4_top8, (APPROX_MODE, is_fp32_dest_acc_en, 0 /* read_base */, 4 /* store_lo */, 6 /* store_hi */));

        _llk_math_generalized_moe_gate_copy4rows_init_<12 /* src */, 0 /* dst */, false /* is_32bit */, 28 /* srcb */>();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false /* is_32bit */>();

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
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_<false /* is_32bit */>();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, false /* is_32bit */>();
    }
}

static inline void run_move()
{
    if constexpr (GMG_SUB_OP == MOVE_STEP0)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_<false /* is_32bit */>();
        mop_dest_reset();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, false /* is_32bit */>();
    }
    else if constexpr (GMG_SUB_OP == MOVE_STEP1)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_<false /* is_32bit */>();
        mop_dest_reset();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, false /* is_32bit */>();
    }
    else if constexpr (GMG_SUB_OP == MOVE_STEP1_HI)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<GMG_D2B_DST, GMG_B2D_BASE, false /* is_32bit */>();
        mop_dest_reset();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, false /* is_32bit */>();
    }
    else if constexpr (GMG_SUB_OP == MOVE_STEP2)
    {
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_<false /* is_32bit */>();
        mop_dest_reset();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, false /* is_32bit */>();
    }
    else
    {
        _llk_math_generalized_moe_gate_copy4rows_init_<GMG_ROW_SRC, GMG_ROW_DST, false /* is_32bit */, GMG_SRCB>();
        mop_dest_reset();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false /* is_32bit */>();
        if constexpr (GMG_SECOND_COPY)
        {
            _llk_math_generalized_moe_gate_copy4rows_init_<GMG_ROW_SRC_2, GMG_ROW_DST_2, false /* is_32bit */, GMG_SRCB_2>();
            mop_dest_reset();
            _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false /* is_32bit */>();
        }
    }
}

static inline void run_placement()
{
    if constexpr (GMG_PRE_COPY4ROWS)
    {
        // An FPU MOP leaves the Dst RWC advanced by +64 per tile. The SFPU ops below each reset it
        // on entry; without a MOP in front of them that reset is never needed and never tested.
        _llk_math_generalized_moe_gate_copy4rows_init_<GMG_ROW_SRC, GMG_ROW_DST, false /* is_32bit */, GMG_SRCB>();
        mop_dest_reset();
        _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, false /* is_32bit */>();
    }

    if constexpr (GMG_SUB_OP == RUN_MERGE4_TOP8)
    {
        GMG_SFPU_CALL(generalized_moe_gate_merge4_top8, (APPROX_MODE, is_fp32_dest_acc_en, GMG_READ_BASE, GMG_TO_LO, GMG_TO_HI));
    }
    else if constexpr (GMG_SUB_OP == RUN_COPY_TOPK_RUN)
    {
        GMG_SFPU_CALL(generalized_moe_gate_copy_topk_run, (APPROX_MODE, GMG_FROM_LO, GMG_FROM_HI, GMG_TO_LO, GMG_TO_HI));
    }
    else if constexpr (GMG_SUB_OP == RUN_PLACE_FIELD)
    {
        GMG_SFPU_CALL(
            generalized_moe_gate_place_field_from_interm, (APPROX_MODE, GMG_FIELD, GMG_FROM_LO, GMG_FROM_HI, GMG_TO_LO, GMG_TO_HI));
    }
    else if constexpr (GMG_SUB_OP == RUN_MERGE16)
    {
        GMG_SFPU_CALL(generalized_moe_gate_merge16_to_run, (APPROX_MODE, is_fp32_dest_acc_en, GMG_TO_LO, GMG_TO_HI, GMG_IDX_OFFSET));
    }
    else if constexpr (GMG_SUB_OP == RUN_COMBINE_FINALIZE)
    {
        // generalized_moe_gate_combine_finalize, whole. The arriving run is placed at {4,6}, then
        // the pair at {0,2}+{4,6} is sorted, normalized and transposed to the output layout. Note
        // finalize does its own merge, so unlike the RUN_COMBINE tail there is no merge16 here.
        GMG_SFPU_CALL(
            generalized_moe_gate_place_field_from_interm,
            (APPROX_MODE, 0 /* field */, 0 /* src_lo */, 2 /* src_hi */, 4 /* dst_lo */, 6 /* dst_hi */));
        GMG_SFPU_CALL(
            generalized_moe_gate_place_field_from_interm,
            (APPROX_MODE, 1 /* field */, 4 /* src_lo */, 6 /* src_hi */, 4 /* dst_lo */, 6 /* dst_hi */));
        GMG_SFPU_CALL(
            generalized_moe_gate_place_field_from_interm,
            (APPROX_MODE, 2 /* field */, 8 /* src_lo */, 10 /* src_hi */, 4 /* dst_lo */, 6 /* dst_hi */));
        GMG_SFPU_CALL(generalized_moe_gate_finalize_ungrouped, (APPROX_MODE, is_fp32_dest_acc_en, GMG_TOPK, GMG_SOFTMAX), GMG_EPS, GMG_SCALE);
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_<false /* is_32bit */>();
        mop_dest_reset();
        _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, false /* is_32bit */>();
    }
    else if constexpr (GMG_SUB_OP == RUN_COMBINE_RELOCATED)
    {
        // The same combine, but the arriving run is already in DEST at {8,10} and reaches the merge
        // slot by relocation instead. Whether a relocated run is still a run the merge accepts is
        // the run format's contract, and copy_topk_run's own test only checks that cells moved.
        GMG_SFPU_CALL(generalized_moe_gate_copy_topk_run, (APPROX_MODE, 8 /* from_lo */, 10 /* from_hi */, 4 /* to_lo */, 6 /* to_hi */));
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
        GMG_SFPU_CALL(
            generalized_moe_gate_place_field_from_interm,
            (APPROX_MODE, 0 /* field */, 0 /* src_lo */, 2 /* src_hi */, 4 /* dst_lo */, 6 /* dst_hi */));
        GMG_SFPU_CALL(
            generalized_moe_gate_place_field_from_interm,
            (APPROX_MODE, 1 /* field */, 4 /* src_lo */, 6 /* src_hi */, 4 /* dst_lo */, 6 /* dst_hi */));
        GMG_SFPU_CALL(
            generalized_moe_gate_place_field_from_interm,
            (APPROX_MODE, 2 /* field */, 8 /* src_lo */, 10 /* src_hi */, 4 /* dst_lo */, 6 /* dst_hi */));
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
    // Two sections walk both DEST halves under DstSync::Half. Only the second reaches a non-zero
    // get_dest_buffer_base(), which every set_dst_write_addr in the op adds to its tile offset.
    for (std::uint32_t section = 0; section < GMG_SECTIONS; ++section)
    {
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

            if constexpr (GMG_MODE == MODE_GATE && GMG_SIGMOID)
            {
                // The op's enable_sigmoid front-end: transpose_wh_tile then sigmoid_tile leave the
                // activated score in the score region, which the RELOAD binary below reads back through
                // MOVD2A instead of taking SrcA from the unpacker. The transpose itself is the
                // unpacker's; math only datacopies what it produced.
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

        if constexpr (GMG_MODE == MODE_GATE || GMG_MODE == MODE_BINARY)
        {
            _llk_math_generalized_moe_gate_eltwise_binary_init_<ELTWISE_BINARY_OP, BINARY_MODE, MATH_FIDELITY>(params.num_faces, ACC_TO_DEST);
            _llk_math_generalized_moe_gate_eltwise_binary_<ELTWISE_BINARY_OP, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY>(
                params.num_faces, 0 /* dst_index */);
        }

        if constexpr (GMG_MODE != MODE_BINARY)
        {
            _llk_math_generalized_moe_gate_transpose_dest_single_face_common_init_<false /* is_32bit */>();
            SFPU_UNARY_INIT_FN(unused, sfpu::generalized_moe_gate_topk_init, (APPROX_MODE, is_fp32_dest_acc_en));
        }

        if constexpr (GMG_MODE == MODE_GATE)
        {
            run_gate();
            // Only the normalized answer path leaves the top-8 in row 0; produce_run leaves a
            // re-mergeable run the test reads by cell, so it must not be sanitized.
            if constexpr (!(GMG_PRODUCE_RUN && !GMG_GROUPED))
            {
                gmg_sanitize_scratch();
            }
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

    // Section s packs to buffer_Res[4s..4s+3], so the two DEST halves come back separately and can
    // be compared against each other.
    for (std::uint32_t section = 0; section < GMG_SECTIONS; ++section)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < NUM_DEST_TILES; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[section * NUM_DEST_TILES + tile]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
