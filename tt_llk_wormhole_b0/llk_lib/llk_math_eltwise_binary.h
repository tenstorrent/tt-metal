// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "lltt.h"

using namespace ckernel;

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, std::uint32_t FIDELITY_INCREMENT>
inline void eltwise_binary_configure_addrmod()
{
    constexpr std::uint32_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? 8 : 0;
    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 0, .cr = 1}, .fidelity = {.incr = FIDELITY_INCREMENT}}
        .set(ADDR_MOD_2);

    addr_mod_t {
        .srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1}, .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_3);
}

// Helper template to select the appropriate eltwise binary operation
template <EltwiseBinaryType eltwise_binary_type>
inline auto eltwise_binary_func(std::uint8_t clr_src, std::uint8_t acc_to_dest, std::uint8_t broadcast_type, std::uint8_t addr_mod)
{
    if constexpr (eltwise_binary_type == ELWADD)
    {
        return TT_OP_ELWADD(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
    }
    else if constexpr (eltwise_binary_type == ELWSUB)
    {
        return TT_OP_ELWSUB(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
    }
    else
    {
        return TT_OP_ELWMUL(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
    }
}

template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void eltwise_binary_reuse_dest_as_src()
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        move_d2a_fixed_face(ADDR_MOD_1);
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        move_d2b_fixed_face(ADDR_MOD_1);
    }
}

// Helper to run the eltwise binary loop with optional dest reuse and face clearing
template <bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_helper_func(
    const std::uint32_t loop_count,
    const std::uint32_t face_offset, // 0 for faces 0&1, 2 for faces 2&3
    const bool clear_fp32_dst_acc,
    const std::uint32_t dst_index)
{
    constexpr std::uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;

#pragma GCC unroll 0
    for (std::uint32_t n = 0; n < loop_count; n++)
    {
        eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();

        // Clear DEST face-by-face when reusing dest as source
        if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE)
        {
            auto base_address = (get_dest_buffer_base() >> 4) + (dst_index << ((is_fp32_dest_acc_en && clear_fp32_dst_acc) ? 3 : 2));
            // fp32 zeroacc can only clear 8x16 datums at a time, need to call twice per 16x16 face
            if (is_fp32_dest_acc_en && clear_fp32_dst_acc)
            {
                const std::uint32_t face_offset_fp32 = face_offset * 2;
                TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, base_address + (face_offset_fp32 + n * 2));         // Clear lower half
                TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, base_address + (face_offset_fp32 + ((n * 2) + 1))); // Clear upper half
            }
            else
            {
                TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, base_address + (face_offset + n));
            }
        }

        ckernel_template::run();
    }
}

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam Dst: Destination sync mode
 * @tparam is_fp32_dest_acc_en: Enable FP32 destination accumulator
 * @tparam NUM_FIDELITY_PHASES: Number of fidelity phases for high-fidelity math
 * @tparam binary_reuse_dest: Reuse destination as source type
 * @param num_faces: Number of faces to process (1, 2, or 4)
 * @param dst_index: Tile index into the destination register
 * @param clear_fp32_dst_acc: Whether to clear FP32 destination accumulator
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    int NUM_FIDELITY_PHASES                      = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_(const std::uint32_t num_faces, std::uint32_t dst_index, const bool clear_fp32_dst_acc)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool high_fidelity          = (NUM_FIDELITY_PHASES > 0);
    constexpr std::uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice for full tile size
            constexpr std::uint32_t outerloop = (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? 2 : 1;
#pragma GCC unroll 0
            for (std::uint32_t n = 0; n < outerloop; n++)
            {
                eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                ckernel_template::run();
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            if (num_faces == 4)
            {
#pragma GCC unroll 0
                for (std::uint32_t n = 0; n < outerloop; n++)
                {
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                    ckernel_template::run();
                }
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            const std::uint32_t outerloop = (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? num_faces : 1;
#pragma GCC unroll 0
            for (std::uint32_t n = 0; n < outerloop; n++)
            {
                eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                ckernel_template::run();
            }
            // Manually clear B once mop is done for scaler bcast
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    else if constexpr (eltwise_binary_type == ELWMUL)
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // Mop for col broadcast only does 2 outerloops.  Needs to clear B manually and call twice for full tile size
            constexpr std::uint32_t outerloop = (high_fidelity) ? 2 : ((binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? 2 : 1);
            eltwise_binary_reuse_dest_helper_func<is_fp32_dest_acc_en, binary_reuse_dest>(outerloop, 0 /*face_base_offset*/, clear_fp32_dst_acc, dst_index);
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);

            if (num_faces == 4)
            {
                eltwise_binary_reuse_dest_helper_func<is_fp32_dest_acc_en, binary_reuse_dest>(outerloop, 2 /*face_base_offset*/, clear_fp32_dst_acc, dst_index);
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // Row and no broadcasted behaves similarly
            const std::uint32_t outerloop = (high_fidelity) ? num_faces : ((binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? num_faces : 1);
            eltwise_binary_reuse_dest_helper_func<is_fp32_dest_acc_en, binary_reuse_dest>(outerloop, 0 /*face_base_offset*/, clear_fp32_dst_acc, dst_index);

            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    math::clear_dst_reg_addr();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType bcast_type,
    int NUM_FIDELITY_PHASES                      = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void eltwise_binary_configure_mop(const std::uint32_t acc_to_dest = 0, const std::uint32_t num_faces = 4)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool high_fidelity      = (NUM_FIDELITY_PHASES > 0);
    const std::uint32_t addr_mod      = ADDR_MOD_0;
    constexpr std::uint32_t innerloop = 16 >> 3; // 8 rows per eltwise op at a time.

    // Consolidated outerloop calculation: binary_reuse_dest has highest priority, then COL, then default
    const std::uint32_t outerloop = (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) ? 1 : (bcast_type == BroadcastType::COL) ? 2 : num_faces;

    constexpr auto broadcast_type = (bcast_type == BroadcastType::COL)      ? p_elwise::SRCB_BCAST_COL
                                    : (bcast_type == BroadcastType::ROW)    ? p_elwise::SRCB_BCAST_ROW
                                    : (bcast_type == BroadcastType::SCALAR) ? p_elwise::SRCB_BCAST_ALL
                                                                            : p_elwise::SRCB_NO_BCAST;

    // Scalar and Col broadcast should not Clear B within a mop.  This is controlled outside of MOP.
    constexpr auto CLR_SRC = (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) ? p_setrwc::CLR_A : p_setrwc::CLR_AB;

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        ckernel_template tmp(outerloop, innerloop, eltwise_binary_func<eltwise_binary_type>(0, acc_to_dest, broadcast_type, addr_mod));
        tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    else if constexpr (eltwise_binary_type == ELWMUL)
    {
        ckernel_template tmp(high_fidelity ? NUM_FIDELITY_PHASES : outerloop, innerloop, eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, addr_mod));
        if constexpr (high_fidelity)
        {
            tmp.set_last_inner_loop_instr(eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, ADDR_MOD_2)); // Incr fidelity last inst of inner loop
            tmp.set_last_outer_loop_instr(eltwise_binary_func<ELWMUL>(CLR_SRC, 0, broadcast_type, ADDR_MOD_3));
        }
        else
        {
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        }
        tmp.program();
    }
}

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam MATH_FIDELITY_DESC: Math fidelity descriptor for controlling precision
 * @tparam binary_reuse_dest: Reuse destination as source type
 * @param num_faces: Number of faces to process (1, 2, or 4)
 * @param acc_to_dest: Accumulate to destination flag
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int MATH_FIDELITY_DESC                       = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_(const std::uint32_t num_faces, const std::uint32_t acc_to_dest)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");
    constexpr int MATH_FIDELITY_PHASES    = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr int MATH_FIDELITY_INCREMENT = get_math_fidelity_increment(MATH_FIDELITY_DESC);

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, MATH_FIDELITY_INCREMENT>();
    eltwise_binary_configure_mop<eltwise_binary_type, src_b_bcast_type, MATH_FIDELITY_PHASES, binary_reuse_dest>(acc_to_dest, num_faces);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Uninitialize/cleanup after elementwise binary operations
 * Restores any modified state to defaults
 */
inline void _llk_math_eltwise_binary_uninit_()
{
    // No state to restore - all states are transient or default
}

/*************************************************************************
 * LLK eltwise_bcast_row_tile math implementation for SDPA

 These LLKs are meant to be used with unpacker that does unpack broadcast row
 on srcA register _llk_unpack_bcastA_B_. Using them with other unpack LLKs will lead to hangs
 since toggling of dvalid signal is different in both cases.

 *************************************************************************/
inline void eltwise_binary_configure_mop(std::uint32_t srca_reuse_count = 4)
{
    /*

        MOP configuration is following. In innerloop single tile is processed via TT_OP_REPLAY.
        After all innerloop iterations are finished dvalid for SrcA is cleared signaling
        the unpacker to load new tile in SrcA.

    */

    std::uint32_t innerloop           = srca_reuse_count;
    constexpr std::uint32_t outerloop = 1;

    ckernel_template tmp(outerloop, innerloop, TT_OP_REPLAY(0, 10, 0, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_AB)); // Clearing src A dvalid
    tmp.program();
}

inline void eltwise_binary_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 8},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_1);
}

template <EltwiseBinaryType eltwise_binary_type, int NUM_FIDELITY_PHASES = 0>
inline void _llk_math_eltwise_binary_init_(std::uint32_t srca_reuse_count = 4)
{
    eltwise_binary_configure_addrmod();

    /*
        Loading of instructions into replay buffer. First 4 operate on F0 and F1,
        and second 4 operate on F2 and F3. Each pair of instructions operates on 8 rows
        of the tile. The last instruction clears B dvalid which means unpacker
        will load following B tile while still keeping same A tile in srcA.
        After F0 and F1 A counter is cleared which allows it to reuse
        boradcasted data.
    */

    auto eltwise_op = [](std::uint8_t addr_mod)
    {
        if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWSUB)
        {
            TTI_ELWSUB(0, 0, 0, addr_mod, 0);
        }
        else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWADD)
        {
            TTI_ELWADD(0, 0, 0, addr_mod, 0);
        }
        else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
        {
            TTI_ELWMUL(0, 0, 0, addr_mod, 0);
        }
    };

    // Setup eltwise operation for one tile
    // TTI_REPLAY(0, 10, 0, 1);
    lltt::record<lltt::NoExec>(0, 10);

    // Dest address is always incremented by 8 in address mode
    eltwise_op(ADDR_MOD_0); // srca_increment -> 0 | srcb_increment -> 8
    eltwise_op(ADDR_MOD_1); // srca_increment -> 8 | srcb_increment -> 8

    eltwise_op(ADDR_MOD_0); // srca_increment -> 0 | srcb_increment -> 8
    eltwise_op(ADDR_MOD_1); // srca_increment -> 8 | srcb_increment -> 8

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_A);

    eltwise_op(ADDR_MOD_0); // srca_increment -> 0 | srcb_increment -> 8
    eltwise_op(ADDR_MOD_1); // srca_increment -> 8 | srcb_increment -> 8

    eltwise_op(ADDR_MOD_0); // srca_increment -> 0 | srcb_increment -> 8
    eltwise_op(ADDR_MOD_1); // srca_increment -> 8 | srcb_increment -> 8

    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_AB); // Clearing B dvalid

    eltwise_binary_configure_mop(srca_reuse_count);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_eltwise_binary_(std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AB);

    // Run the MOP
    ckernel_template::run();

    math::clear_dst_reg_addr();
}
