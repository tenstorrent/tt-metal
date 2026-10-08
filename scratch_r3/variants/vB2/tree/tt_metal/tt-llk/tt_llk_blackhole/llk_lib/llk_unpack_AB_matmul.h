// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// Reuse direction (ct_dim >= rt_dim) the unpack MOP was programmed for; written and read only under LLK asserts.
inline bool unpack_matmul_init_reuse_a = true;

/**
 * @brief Set the SrcA address step used to stream matmul in1 columns.
 *
 * tile_size is in unpacker L1 address units (16-byte words), not bytes. Only the
 * column-streaming path (ct_dim >= rt_dim, no in1 kernel broadcast) uses this step.
 * A stride of one restores the normal contiguous-tile step.
 */
inline void _llk_unpack_AB_matmul_set_in1_column_stride_(const std::uint32_t tile_size, const std::uint32_t stride_tiles)
{
    LLK_ASSERT(tile_size > 0 && tile_size <= 0xffff, "Matmul tile size must fit the SrcA address-step register");
    LLK_ASSERT(stride_tiles > 0 && stride_tiles <= 0xffff / tile_size, "Matmul column stride must fit the SrcA address-step register");
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size * stride_tiles), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
}

/**
 * @brief Whether a format holds 8 bits per datum or less (block float, fp8 and 8-bit integer formats).
 *
 * The unpacker moves such a tile in half the time of a 16-bit one, so the GPR address advance of the streamed operand
 * (RDCFG, ADDDMAREG, STALLWAIT, WRCFG) would set the rate; these formats get the CFGSHIFTMASK advance instead. 16-bit and
 * 32-bit formats keep the GPR advance.
 *
 * @param unpack_src_format: Unpacker input (L1) data format.
 */
inline constexpr bool _llk_unpack_AB_matmul_narrow_format_(const std::uint32_t unpack_src_format)
{
    switch (unpack_src_format)
    {
        case to_underlying(DataFormat::Bfp8):
        case to_underlying(DataFormat::Bfp8_b):
        case to_underlying(DataFormat::Bfp4):
        case to_underlying(DataFormat::Bfp4_b):
        case to_underlying(DataFormat::Bfp2):
        case to_underlying(DataFormat::Bfp2_b):
        case to_underlying(DataFormat::Lf8):
        case to_underlying(DataFormat::Fp8_e4m3):
        case to_underlying(DataFormat::Int8):
        case to_underlying(DataFormat::UInt8):
            return true;
        default:
            return false;
    }
}

/**
 * @brief Whether the operand a matmul streams holds 8 bits per datum or less: the value of the stream_narrow argument of
 *        @ref _llk_unpack_AB_matmul_init_.
 *
 * The streamed operand is in1 (unpacked into SrcA) when ct_dim >= rt_dim and in0 (SrcB) otherwise.
 *
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param unpA_src_format: Unpacker input (L1) data format of in1, the operand unpacked into SrcA, as given to
 *                         @ref _llk_unpack_hw_configure_.
 * @param unpB_src_format: Unpacker input (L1) data format of in0, the operand unpacked into SrcB.
 * @param fp32_dest_acc_en: Whether DEST holds 32-bit data; such a kernel keeps the GPR advance for every format (the
 *                          narrow body's THCON wait slows its pack-bound blocks).
 */
inline constexpr bool _llk_unpack_AB_matmul_stream_narrow_(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const bool fp32_dest_acc_en)
{
    static_cast<void>(fp32_dest_acc_en);
    return _llk_unpack_AB_matmul_narrow_format_((ct_dim >= rt_dim) ? unpA_src_format : unpB_src_format);
}

/**
 * @brief Record the replay body for one streamed tile of a matmul row: the UNPACR group, the base address advance of that
 *        unpacker and the NOP that covers the config write.
 *
 * A narrow operand (8 bits per datum or less) advances with one CFGSHIFTMASK that adds SCRATCH_SEC0_val to CFG_REG; the
 * other formats read CFG_REG into a GPR, add STRIDE_GPR and write it back, as before this change. Under kernel broadcast
 * the advance is replaced by NOPs so the same tile is re-read.
 *
 * @tparam SRC: SrcA or SrcB, the source register (and unpacker) of the streamed operand.
 * @tparam CFG_REG: Base address register of the streamed operand's unpacker in the config context this copy serves.
 * @tparam STRIDE_GPR: GPR holding the streamed tile stride for the GPR advance.
 * @tparam ADVANCE: False under kernel broadcast (the base address is not advanced).
 * @param partial_face: Whether the streamed operand is unpacked face-by-face.
 * @param narrow: Whether the streamed operand's format holds 8 bits per datum or less.
 */
template <std::uint32_t SRC, std::uint32_t CFG_REG, std::uint32_t STRIDE_GPR, bool ADVANCE>
inline void _llk_unpack_AB_matmul_stream_tile_body_(const bool partial_face, const bool narrow)
{
    if (partial_face)
    {
        TTI_UNPACR_NOP(SRC, 0, 0, 0 /*Set Dvalid*/, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR(SRC, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        TTI_UNPACR(SRC, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
        TTI_SETADCZW((SRC == SrcA) ? p_setadc::UNP_A : p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
    }
    else
    {
        TTI_UNPACR(SRC, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
    }

    if constexpr (!ADVANCE)
    {
        // keep the length of the address advance
        TTI_NOP;
        TTI_NOP;
        TTI_NOP;
        if (!narrow)
        {
            TTI_NOP;
        }
    }
    else if (narrow)
    {
        // SCRATCH_SEC0_val = STRIDE_GPR, then CFG_REG += SCRATCH_SEC0_val (0b011 = add, 32-bit mask, scratch_sel 0): the stride is
        // read from the GPR every tile as before, without the RDCFG and ADDDMAREG round trip
        TTI_NOP; // the call waits for THCON before its MOP
        TTI_WRCFG(STRIDE_GPR, p_cfg::WRCFG_32b, SCRATCH_SEC0_val_ADDR32);
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0, CFG_REG);
    }
    else
    {
        TTI_RDCFG(p_gpr_unpack::TMP0, CFG_REG);
        TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, STRIDE_GPR);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, 0, CFG_REG);
    }
    // The config write takes two cycles and the next UNPACR must see the new base address.
    TTI_NOP;
}

/**
 * @brief Program the unpacker MOP/replay buffer for a matmul operand unpack.
 *
 * Builds a replay buffer that unpacks the streamed (non-reused) operand and advances its L1 base
 * address by one tile each step. Which operand is reused versus streamed is chosen by comparing
 * ct_dim and rt_dim (reuse_a = ct_dim >= rt_dim); operand A maps to SrcB and operand B to SrcA.
 * The address advance is suppressed under kernel broadcast.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 * @param stream_narrow: Whether the streamed operand's format holds 8 bits per datum or less (CFGSHIFTMASK address advance).
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_mop_config_(
    const std::uint32_t ct_dim, const std::uint32_t rt_dim, const bool unpA_partial_face, const bool unpB_partial_face, const bool stream_narrow = false)
{
    // in0/inA - loaded to SrcB
    // in1/inB - loaded to SrcA

    const bool reuse_a             = ct_dim >= rt_dim;
    const bool stream_partial_face = reuse_a ? unpA_partial_face : unpB_partial_face;
    LLK_ASSERT_BLOCK(unpack_matmul_init_reuse_a = reuse_a);
    // two copies of the streamed tile body, one per config context: the UNPACR group (4 instructions for a partial face,
    // 1 otherwise), the address advance (3 instructions for a narrow format, 4 otherwise) and the NOP
    const std::uint32_t replay_buf_run_len  = (stream_partial_face ? 4 : 1) + (stream_narrow ? 3 : 4) + 1;
    const std::uint32_t replay_buf_prog_len = 2 * replay_buf_run_len;

    if (reuse_a)
    {
        static_assert(kernel_broadcast_b <= 1, "kernel_broadcast>1 on matmul input 1 is not supported with reuse enabled");
        constexpr bool advance = (kernel_broadcast_b != 1);
        load_replay_buf(
            0,
            replay_buf_prog_len,
            // Lambda function to set up replay buffer
            [unpA_partial_face, stream_narrow]
            {
                _llk_unpack_AB_matmul_stream_tile_body_<SrcA, THCON_SEC0_REG3_Base_address_ADDR32, p_gpr_unpack::TILE_SIZE_A, advance>(
                    unpA_partial_face, stream_narrow);
                _llk_unpack_AB_matmul_stream_tile_body_<SrcA, THCON_SEC0_REG3_Base_cntx1_address_ADDR32, p_gpr_unpack::TILE_SIZE_A, advance>(
                    unpA_partial_face, stream_narrow);
            });
    }
    else
    {
        static_assert(kernel_broadcast_a <= 1, "kernel_broadcast>1 on matmul input 0 is not supported with reuse enabled");
        constexpr bool advance = (kernel_broadcast_a != 1);
        load_replay_buf(
            0,
            replay_buf_prog_len,
            // Lambda function to set up replay buffer
            [unpB_partial_face, stream_narrow]
            {
                _llk_unpack_AB_matmul_stream_tile_body_<SrcB, THCON_SEC1_REG3_Base_address_ADDR32, p_gpr_unpack::TMP_LO, advance>(
                    unpB_partial_face, stream_narrow);
                _llk_unpack_AB_matmul_stream_tile_body_<SrcB, THCON_SEC1_REG3_Base_cntx1_address_ADDR32, p_gpr_unpack::TMP_LO, advance>(
                    unpB_partial_face, stream_narrow);
            });
    }

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                    // src B
        false,                                    // halo - just used for 4 unpacks
        lltt::replay_insn(0, replay_buf_run_len), // runs when context is 0
        0,
        0,
        0,
        lltt::replay_insn(replay_buf_run_len, replay_buf_run_len), // runs when context is 1
        0,
        0);

    tmp.program();
}

/**
 * @brief Initialize the unpacker for a matmul (A x B) operation.
 *
 * Re-enables within-face transpose if needed, programs per-unpacker datum counts (full-tile or
 * face-by-face for partial faces), stashes kt_dim into a GPR for tile-size scaling, and programs the matmul MOP
 * with the address advance the streamed operand's format calls for.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param transpose: Nonzero to enable within-face (16x16) transpose for SrcA.
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param kt_dim: Number of tiles along the contraction (K) dimension.
 * @param unpA_face_r_dim: Rows per face for operand A.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @param unpA_num_faces: Number of faces for operand A, valid values = <1, 2, 4>.
 * @param unpB_num_faces: Number of faces for operand B, valid values = <1, 2, 4>.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 * @param stream_narrow: Whether the streamed operand holds 8 bits per datum or less, from
 *                       @ref _llk_unpack_AB_matmul_stream_narrow_; such an operand is streamed at its data rate (the replay
 *                       advances its address with one CFGSHIFTMASK), every other one as before.
 * @note Call @ref _llk_unpack_AB_matmul_uninit_ to restore the modified datum-count state.
 * @ref _llk_unpack_AB_matmul_ is the matching execute call.
 * @ref _llk_math_matmul_init_ is the matching init on the math thread (consumes SrcA/SrcB).
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
__attribute__((always_inline)) inline void _llk_unpack_AB_matmul_init_(
    const std::uint32_t transpose       = 0,
    const std::uint32_t ct_dim          = 1,
    const std::uint32_t rt_dim          = 1,
    const std::uint32_t kt_dim          = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4,
    const bool unpA_partial_face        = false,
    const bool unpB_partial_face        = false,
    const bool stream_narrow            = false)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");
    // 16x16 matmul not supported - no dedicated math path; falls to 32x32 default which is incorrect for < 4 faces
    LLK_ASSERT(!(unpA_num_faces == 1 && unpB_num_faces == 1), "16x16 by 16x16 matmul is not supported");

    // also turn on within_face_16x16_transpose if it was turned off by datacopy at runtime
    // on WH, the unpacker performs both transpose of faces as well as transpose each face.
    // the former is configured in mop, the latter is configured in cfg register in hw_configure
    // in large matmul, datacopy will disable the transpose of faces, so we need it turn it back on for matmul.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    if (unpA_partial_face)
    {
        // Do face by face unpacking. Need to program correct face dim
        // to compute address of the next face
        config_unpacker_x_end<p_setadc::UNP_A>(unpA_face_r_dim);
    }
    else
    {
        const std::uint32_t unpA_x_end = unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
        TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    }

    if (unpB_partial_face)
    {
        // Do face by face unpacking. Need to program correct face dim
        // to compute address of the next face
        config_unpacker_x_end<p_setadc::UNP_B>(unpB_face_r_dim);
    }
    else
    {
        // Do full tile unpacking. No need to program face dim
        // as address counter pointing to the face is not incremented
        const std::uint32_t unpB_x_end = unpB_num_faces * unpB_face_r_dim * FACE_C_DIM - 1;
        TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);
    }

    TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(p_gpr_unpack::KT_DIM)); // store kt_dim to gpr for scaling tile size

    _llk_unpack_AB_matmul_mop_config_<kernel_broadcast_a, kernel_broadcast_b>(ct_dim, rt_dim, unpA_partial_face, unpB_partial_face, stream_narrow);
}

/**
 * @brief No-op after a matmul operation.
 *
 * x-start/x-end is transient and reprogrammed by the next operation's init (see tt-llk#1036), so
 * there is nothing to restore here.
 *
 * @note Call @ref _llk_unpack_AB_matmul_init_ before this function.
 */
inline void _llk_unpack_AB_matmul_uninit_()
{
}

/**
 * @brief Unpack the operand tiles for a matmul (A x B) into SrcA and SrcB.
 *
 * Iterates over the reused dimension, computing per-tile L1 addresses (with optional kernel-
 * broadcast wraparound and kt_dim striding), and unpacks operand A to SrcB / operand B to SrcA
 * for each step while synchronizing through the unpack semaphore and config-context switching.
 *
 * @tparam kernel_broadcast_a: Tile count to wrap operand A around for kernel broadcast (0 = disabled).
 * @tparam kernel_broadcast_b: Tile count to wrap operand B around for kernel broadcast (0 = disabled).
 * @param base_address_a: L1 base address of operand A's tile buffer.
 * @param base_address_b: L1 base address of operand B's tile buffer.
 * @param tile_index_a: Starting tile index into operand A.
 * @param tile_index_b: Starting tile index into operand B.
 * @param tile_size_a: Size of one operand A tile, used to compute per-tile offsets.
 * @param tile_size_b: Size of one operand B tile, used to compute per-tile offsets.
 * @param unpA_partial_face: Whether operand A is unpacked face-by-face (partial faces).
 * @param unpB_partial_face: Whether operand B is unpacked face-by-face (partial faces).
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param kt_dim: Number of tiles along the contraction (K) dimension.
 * @note Call @ref _llk_unpack_AB_matmul_init_ with matching template args before this function, and
 *       @ref _llk_unpack_AB_matmul_uninit_ after it to restore modified state.
 * @ref _llk_math_matmul_ on the math thread consumes the SrcA/SrcB tiles unpacked here.
 */
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const bool unpA_partial_face = false,
    const bool unpB_partial_face = false,
    std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim   = 1,
    const std::uint32_t kt_dim   = 1,
    const bool stream_narrow     = false)
{
    // In0/InA -> srcB (supports partial face)
    // In1/InB -> srcA

    volatile std::uint32_t *cfg = get_cfg_pointer(); // get pointer to registers for current state ID

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
    LLK_ASSERT(reuse_a == unpack_matmul_init_reuse_a, "matmul: ct_dim >= rt_dim differs from the init's; re-init for a block of the other reuse direction");

    if (!reuse_a)
    {
        TTI_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM);
    }
    if (stream_narrow)
    {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON); // the 8-bit body's WRCFG reads the stride GPR, and on Blackhole it can pass a THCON write
    }

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        std::uint32_t offset_address_a = tile_size_a * (tile_index_a + (reuse_a ? (t * kt_dim) : (0)));
        std::uint32_t offset_address_b = tile_size_b * (tile_index_b + (reuse_a ? (0) : (t)));
        if constexpr (kernel_broadcast_a > 0)
        {
            offset_address_a = tile_size_a * ((tile_index_a + (reuse_a ? ((t * kt_dim)) : (0))) % kernel_broadcast_a);
        }
        if constexpr (kernel_broadcast_b > 0)
        {
            offset_address_b = tile_size_b * ((tile_index_b + (reuse_a ? (0) : (t))) % kernel_broadcast_b);
        }

        std::uint32_t address_a = base_address_a + offset_address_a;
        std::uint32_t address_b = base_address_b + offset_address_b;

        // Wait for free context
        wait_for_next_context(2);

        // Validate and configure addresses (note: address_b goes to SEC0, address_a to SEC1 for matmul)
        _llk_unpack_configure_addresses_(address_b, address_a, cfg);

        semaphore_post(semaphore::UNPACK_SYNC); // Trisc::SEMPOST for context acquire

        // Stall unpacker until pending CFG writes from Trisc have completed
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        if (reuse_a)
        {
            if (unpB_partial_face)
            {
                TTI_UNPACR_NOP(SrcB, 0, 0, 0 /*Set Dvalid*/, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                // Do face by face unpacking
                TTI_UNPACR(
                    SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                TTI_UNPACR(
                    SrcB, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
            }
            else
            {
                TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            }
        }
        else
        {
            if (unpA_partial_face)
            {
                // Do face by face unpacking
                TTI_UNPACR_NOP(SrcA, 0, 0, 0 /*Set Dvalid*/, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                TTI_UNPACR(
                    SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 0 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                TTI_UNPACR(
                    SrcA, 0b00010001, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
                TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101); // Set ch0_z=0, ch1_z=0
            }
            else
            {
                TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0 /* Set ContextIdInc */, 0, 0, 1);
            }
        }

        TT_MOP(0, (reuse_a ? ct_dim : rt_dim) - 1, unp_cfg_context == 0 ? 0 : 0xffff); // Run the MOP

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }
}
