// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "../../common/tensor_shape.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "lltt.h"

using namespace ckernel;

/*************************************************************************
 * Common Helpers
 *************************************************************************/
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
    constexpr std::uint8_t srcb_incr           = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? MAX_FPU_ROWS : 0;
    addr_mod_t {
        .srca = {.incr = MAX_FPU_ROWS},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = MAX_FPU_ROWS},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 0, .cr = 1}, .fidelity = {.incr = fidelity_increment}}
        .set(ADDR_MOD_2);

    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1},
        .srcb     = {.incr = 0, .clr = 1},
        .dest     = {.incr = MAX_FPU_ROWS, .clr = 0, .cr = 0, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 1}}
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

/**
 * @brief Configure MOP for standard eltwise binary operations (no dest reuse)
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting
 * @param tensor_shape: Tensor shape describing tile dimensions
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_standard(const std::uint32_t acc_to_dest, const ckernel::TensorShape &tensor_shape)
{
    validate_tensor_shape_tile_dependent_ops_(tensor_shape);
    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    const bool narrow_tile              = tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim;
    constexpr bool high_fidelity        = is_high_fidelity(math_fidelity);
    constexpr std::uint8_t addr_mod     = ADDR_MOD_0;

    // Inner loop: number of MAX_FPU_ROWS (8-row) operations per face
    // Even if face_r_dim < 16, we still process at least 1 inner loop iteration
    const std::uint8_t innerloop = tensor_shape.face_r_dim > MAX_FPU_ROWS ? (tensor_shape.face_r_dim >> MAX_FPU_ROWS_LOG2) : 1;

    // Outer loop depends on broadcast type:
    // - COL broadcast: MOP processes num_faces_c_dim faces (one row of faces)
    //                  Runtime calls MOP num_faces_r_dim times (one call per row)
    // - Other broadcasts: MOP processes all num_faces in one call
    const std::uint32_t outerloop = (bcast_type == BroadcastType::COL) ? num_faces_c_dim : num_faces;

    constexpr auto broadcast_type = (bcast_type == BroadcastType::COL)      ? p_elwise::SRCB_BCAST_COL
                                    : (bcast_type == BroadcastType::ROW)    ? p_elwise::SRCB_BCAST_ROW
                                    : (bcast_type == BroadcastType::SCALAR) ? p_elwise::SRCB_BCAST_ALL
                                                                            : p_elwise::SRCB_NO_BCAST;

    // Scalar and Col broadcast should not Clear B within a MOP - B is cleared outside of MOP
    constexpr auto CLR_SRC = (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) ? p_setrwc::CLR_A : p_setrwc::CLR_AB;

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        ckernel_template tmp(outerloop, innerloop, eltwise_binary_func<eltwise_binary_type>(0, acc_to_dest, broadcast_type, addr_mod));
        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // For partial faces (face_r_dim < 16), we still need to increment counters by MAX_FPU_ROWS
            // to maintain proper 16-row spacing between faces
            tmp.set_loop_op1(TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
        }
        tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    else if constexpr (eltwise_binary_type == ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            ckernel_template tmp(to_underlying(math_fidelity), innerloop, eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, addr_mod));
            tmp.set_last_inner_loop_instr(eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, ADDR_MOD_2));
            tmp.set_last_outer_loop_instr(eltwise_binary_func<ELWMUL>(CLR_SRC, 0, broadcast_type, ADDR_MOD_3));
            tmp.program();
        }
        else if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // Partial faces: INCRWC as loop_op1 via two-arg constructor to maintain 16-row face spacing.
            // Must use two-arg constructor so m_loop0/1_last_instr = INCRWC, preventing the
            // last-iteration override from replacing INCRWC with a second ELWMUL instruction.
            ckernel_template tmp(
                outerloop, innerloop, eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, addr_mod), TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, addr_mod));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
    }
}

/**
 * @brief Initialize FPU for standard elementwise binary operations (no dest reuse)
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_init_(const ckernel::TensorShape &tensor_shape, const std::uint32_t acc_to_dest)
{
    validate_tensor_shape_tile_dependent_ops_(tensor_shape);
    LLK_ASSERT(math_fidelity == MathFidelity::LoFi || eltwise_binary_type == ELWMUL, "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_ASSERT(
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_standard<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Perform standard elementwise binary operation (no dest reuse)
 * Output = SrcA [+, -, *] SrcB
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam Dst: Destination sync mode, values = <Half, Full>
 * @tparam is_fp32_dest_acc_en: Enable FP32 mode in destination register
 * @tparam math_fidelity: Math fidelity for controlling precision
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param dst_index: Tile index into the destination register
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(const ckernel::TensorShape &tensor_shape, std::uint32_t dst_index)
{
    validate_tensor_shape_tile_dependent_ops_(tensor_shape);
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    LLK_ASSERT(math_fidelity == MathFidelity::LoFi || eltwise_binary_type == ELWMUL, "Math fidelity larger than LoFi only works with Eltwise multiply");
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    // Dest counter always jumps by 32x32 tile spacing regardless of actual tile size
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast: MOP processes num_faces_c_dim faces (one row of faces)
            // Runtime calls MOP num_faces_r_dim times (once per row of faces)
            // After each row, CLR_B to allow next B column to be loaded
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
                ckernel_template::run();
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: MOP handles all faces in one call
            ckernel_template::run();
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
            // COL broadcast: MOP processes fidelity phases for one face (HiFi) or all face columns (LoFi)
            // With high fidelity, call MOP once per face column per face row
            const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
            const std::uint32_t fidelity_loop   = high_fidelity ? num_faces_c_dim : 1;
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
#pragma GCC unroll 0
                for (std::uint32_t i = 0; i < fidelity_loop; i++)
                {
                    ckernel_template::run();
                    if constexpr (high_fidelity)
                    {
                        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
                        {
                            // HiFi: only advance dest and carry register, src was cleared by ADDR_MOD_3
                            TTI_INCRWC(MAX_FPU_ROWS, MAX_FPU_ROWS, 0, 0);
                        }
                    }
                    // LoFi: MOP handles face spacing internally via loop_op1, no runtime INCRWC needed
                }
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: MOP handles all faces, fidelity requires multiple runs
            const std::uint32_t num_faces     = tensor_shape.total_num_faces();
            const std::uint32_t fidelity_loop = high_fidelity ? num_faces : 1;
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < fidelity_loop; i++)
            {
                ckernel_template::run();
                if constexpr (high_fidelity)
                {
                    if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
                    {
                        // HiFi: only advance dest and carry register, src was cleared by ADDR_MOD_3
                        TTI_INCRWC(MAX_FPU_ROWS, MAX_FPU_ROWS, 0, 0);
                    }
                }
                // LoFi: MOP handles face spacing internally via loop_op1, no runtime INCRWC needed
            }
            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    math::clear_dst_reg_addr();
}

/*************************************************************************
 * Eltwise Binary WITH Dest Reuse
 * Complex: Read dest -> Move to src -> Compute -> Store
 *************************************************************************/

template <EltwiseBinaryReuseDestType binary_reuse_dest>
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

/**
 * @brief Configure MOP for eltwise binary operations with dest reuse
 * MOP outer loop = 1 face, called multiple times externally with ZEROACC between calls
 * This processes one face at a time because we need to clear dest before each face computation
 * @tparam eltwise_binary_type: Type of eltwise binary op
 * @tparam bcast_type: Broadcast type for source B
 * @tparam math_fidelity: Math fidelity for controlling precision
 * @param acc_to_dest: Accumulate result to destination register
 * @param tensor_shape: Tensor shape describing tile dimensions
 */
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void eltwise_binary_configure_mop_with_dest_reuse(const std::uint32_t acc_to_dest, const ckernel::TensorShape &tensor_shape)
{
    validate_tensor_shape_tile_dependent_ops_(tensor_shape);
    constexpr bool high_fidelity    = is_high_fidelity(math_fidelity);
    constexpr std::uint8_t addr_mod = ADDR_MOD_0;

    // Inner loop: number of MAX_FPU_ROWS (8-row) operations per face
    const std::uint8_t innerloop = tensor_shape.face_r_dim > MAX_FPU_ROWS ? (tensor_shape.face_r_dim >> MAX_FPU_ROWS_LOG2) : 1;

    // For dest reuse: MOP processes 1 face at a time (outer loop = 1)
    // Runtime calls MOP multiple times with move_d2a/d2b + ZEROACC between calls
    constexpr std::uint32_t outerloop = 1;

    constexpr auto broadcast_type = (bcast_type == BroadcastType::COL)      ? p_elwise::SRCB_BCAST_COL
                                    : (bcast_type == BroadcastType::ROW)    ? p_elwise::SRCB_BCAST_ROW
                                    : (bcast_type == BroadcastType::SCALAR) ? p_elwise::SRCB_BCAST_ALL
                                                                            : p_elwise::SRCB_NO_BCAST;

    // Scalar and Col broadcast should not Clear B within MOP - B is cleared outside of MOP
    constexpr auto CLR_SRC = (bcast_type == BroadcastType::COL || bcast_type == BroadcastType::SCALAR) ? p_setrwc::CLR_A : p_setrwc::CLR_AB;

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        ckernel_template tmp(outerloop, innerloop, eltwise_binary_func<eltwise_binary_type>(0, acc_to_dest, broadcast_type, addr_mod));
        if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // For partial faces, still increment by MAX_FPU_ROWS to maintain 16-row face spacing
            tmp.set_loop_op1(TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
        }
        tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    else if constexpr (eltwise_binary_type == ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            ckernel_template tmp(to_underlying(math_fidelity), innerloop, eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, addr_mod));
            tmp.set_last_inner_loop_instr(eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, ADDR_MOD_2));
            tmp.set_last_outer_loop_instr(eltwise_binary_func<ELWMUL>(CLR_SRC, 0, broadcast_type, ADDR_MOD_3));

            if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
            {
                // HiFi: only advance dest and carry register, src was cleared by ADDR_MOD_3
                tmp.set_end_op(TT_OP_INCRWC(MAX_FPU_ROWS, MAX_FPU_ROWS, 0, 0));
            }
            tmp.program();
        }
        else if (tensor_shape.face_r_dim <= MAX_FPU_ROWS)
        {
            // Partial faces: INCRWC as loop_op1 via two-arg constructor to maintain 16-row face spacing.
            ckernel_template tmp(
                outerloop, innerloop, eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, addr_mod), TT_OP_INCRWC(0, MAX_FPU_ROWS, MAX_FPU_ROWS, MAX_FPU_ROWS));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
        else
        {
            ckernel_template tmp(outerloop, innerloop, eltwise_binary_func<ELWMUL>(0, 0, broadcast_type, addr_mod));
            tmp.set_end_op(TT_OP_SETRWC(CLR_SRC, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
            tmp.program();
        }
    }
}

/**
 * @brief Initialize FPU for elementwise binary operations with dest reuse
 * @tparam eltwise_binary_type: Type of eltwise binary op
 * @tparam src_b_bcast_type: Broadcast type for source B
 * @tparam math_fidelity: Math fidelity for controlling precision
 * @tparam binary_reuse_dest: Reuse destination as source type
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param acc_to_dest: Accumulate result to destination register
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
inline void _llk_math_eltwise_binary_with_dest_reuse_init_(const ckernel::TensorShape &tensor_shape, const std::uint32_t acc_to_dest)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Use _llk_math_eltwise_binary_standard_init_ for no dest reuse");
    validate_tensor_shape_tile_dependent_ops_(tensor_shape);
    LLK_ASSERT(math_fidelity == MathFidelity::LoFi || eltwise_binary_type == ELWMUL, "Math fidelity larger than LoFi only works with Eltwise multiply");
    LLK_ASSERT(
        (eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB) || (eltwise_binary_type == ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");

    eltwise_binary_configure_addrmod<eltwise_binary_type, src_b_bcast_type, math_fidelity>();
    eltwise_binary_configure_mop_with_dest_reuse<eltwise_binary_type, src_b_bcast_type, math_fidelity>(acc_to_dest, tensor_shape);

    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Helper to run the eltwise binary loop with dest reuse and face clearing
template <bool is_fp32_dest_acc_en, EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_run_with_dest_reuse(
    const std::uint32_t loop_count, const std::uint32_t face_offset, const bool clear_fp32_dst_acc, const std::uint32_t dst_index)
{
    constexpr std::uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;

#pragma GCC unroll 0
    for (std::uint32_t n = 0; n < loop_count; n++)
    {
        eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();

        // Clear DEST face-by-face when reusing dest as source
        auto base_address = (get_dest_buffer_base() >> 4) + (dst_index << ((is_fp32_dest_acc_en && clear_fp32_dst_acc) ? 3 : 2));
        if (is_fp32_dest_acc_en && clear_fp32_dst_acc)
        {
            const std::uint32_t face_offset_fp32 = face_offset * 2;
            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, base_address + (face_offset_fp32 + n * 2));
            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, base_address + (face_offset_fp32 + ((n * 2) + 1)));
        }
        else
        {
            TT_ZEROACC(ZERO_ACC_MODE, ADDR_MOD_1, base_address + (face_offset + n));
        }

        ckernel_template::run();
    }
}

/**
 * @brief Perform elementwise binary operation with dest reuse
 * Output = SrcA [+, -, *] SrcB, where one src comes from dest register
 * @tparam eltwise_binary_type: Type of eltwise binary op
 * @tparam src_b_bcast_type: Broadcast type for source B
 * @tparam Dst: Destination sync mode
 * @tparam is_fp32_dest_acc_en: Enable FP32 mode in destination register
 * @tparam math_fidelity: Math fidelity for controlling precision
 * @tparam binary_reuse_dest: Reuse destination as source type
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param dst_index: Tile index into the destination register
 * @param clear_fp32_dst_acc: Clears index in destination register when float32 mode is enabled
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest>
inline void _llk_math_eltwise_binary_with_dest_reuse_(const ckernel::TensorShape &tensor_shape, std::uint32_t dst_index, const bool clear_fp32_dst_acc)
{
    static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, "Use _llk_math_eltwise_binary_standard_ for no dest reuse");
    validate_tensor_shape_tile_dependent_ops_(tensor_shape);
    const std::uint32_t num_faces       = tensor_shape.total_num_faces();
    const std::uint32_t num_faces_r_dim = tensor_shape.num_faces_r_dim;
    const std::uint32_t num_faces_c_dim = tensor_shape.num_faces_c_dim;
    LLK_ASSERT(math_fidelity == MathFidelity::LoFi || eltwise_binary_type == ELWMUL, "Math fidelity larger than LoFi only works with Eltwise multiply");
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    // Dest counter always jumps by 32x32 tile spacing regardless of actual tile size
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB))
    {
        if constexpr (src_b_bcast_type == BroadcastType::COL)
        {
            // COL broadcast with dest reuse:
            // For each face row: process num_faces_c_dim faces, then CLR_B
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
#pragma GCC unroll 0
                for (std::uint32_t face_col = 0; face_col < num_faces_c_dim; face_col++)
                {
                    eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                    ckernel_template::run();
                }
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: process all faces sequentially
#pragma GCC unroll 0
            for (std::uint32_t n = 0; n < num_faces; n++)
            {
                eltwise_binary_reuse_dest_as_src<binary_reuse_dest>();
                ckernel_template::run();
            }
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
            // COL broadcast with dest reuse and multiply:
            // Process num_faces_c_dim faces per row with ZEROACC
#pragma GCC unroll 0
            for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++)
            {
                // face_offset = face_row * num_faces_c_dim (position in face array)
                const std::uint32_t face_offset = face_row * num_faces_c_dim;
                eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest>(num_faces_c_dim, face_offset, clear_fp32_dst_acc, dst_index);
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        }
        else
        {
            // NONE/ROW/SCALAR: process all faces with ZEROACC
            eltwise_binary_run_with_dest_reuse<is_fp32_dest_acc_en, binary_reuse_dest>(num_faces, 0 /*face_offset*/, clear_fp32_dst_acc, dst_index);

            if constexpr (src_b_bcast_type == BroadcastType::SCALAR)
            {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_D);
            }
        }
    }
    math::clear_dst_reg_addr();
}

/*************************************************************************
 * Public API - Wrapper Functions (Backward Compatible)
 *************************************************************************/

/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * Dispatches to standard or dest-reuse implementation based on binary_reuse_dest template parameter
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <NONE, DEST_TO_SRCA, DEST_TO_SRCB>
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param acc_to_dest: Accumulate result to destination register instead of overwriting
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape &tensor_shape, const std::uint32_t acc_to_dest)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        _llk_math_eltwise_binary_standard_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity>(tensor_shape, acc_to_dest);
    }
    else
    {
        _llk_math_eltwise_binary_with_dest_reuse_init_<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest>(tensor_shape, acc_to_dest);
    }
}

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * Dispatches to standard or dest-reuse implementation based on binary_reuse_dest template parameter
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam Dst: Destination sync mode, values = <Half, Full>
 * @tparam is_fp32_dest_acc_en: Enable FP32 mode in destination register
 * @tparam math_fidelity: Math fidelity for controlling precision
 * @tparam binary_reuse_dest: Reuse destination as source type, values = <NONE, DEST_TO_SRCA, DEST_TO_SRCB>
 * @param tensor_shape: Tensor shape describing tile dimensions
 * @param dst_index: Tile index into the destination register
 * @param clear_fp32_dst_acc: Clears index in destination register when float32 mode is enabled
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_(const ckernel::TensorShape &tensor_shape, std::uint32_t dst_index, const bool clear_fp32_dst_acc = false)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE)
    {
        _llk_math_eltwise_binary_standard_<eltwise_binary_type, src_b_bcast_type, Dst, is_fp32_dest_acc_en, math_fidelity>(tensor_shape, dst_index);
    }
    else
    {
        _llk_math_eltwise_binary_with_dest_reuse_<eltwise_binary_type, src_b_bcast_type, Dst, is_fp32_dest_acc_en, math_fidelity, binary_reuse_dest>(
            tensor_shape, dst_index, clear_fp32_dst_acc);
    }
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

template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_init_(std::uint32_t srca_reuse_count = 4)
{
    eltwise_binary_configure_addrmod();

    /*
        Loading of instructions into replay buffer. First 4 operate on F0 and F1,
        and second 4 operate on F2 and F3. Each pair of instructions operates on 8 rows
        of the tile. The last instruction clears B dvalid which means unpacker
        will load following B tile while still keeping same A tile in srcA.
        After F0 and F1 A counter is cleared which allows it to reuse
        broadcasted data.
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
