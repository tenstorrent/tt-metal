// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

/*************************************************************************
 * LLK MUL REDUCE SCALAR - Low-level reduce operations for fused mul+reduce
 *************************************************************************/

// Helper macros for moving destination to source registers
#define MOVD2A_4_ROWS(base_offset, row_offset) \
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + (row_offset) * 4, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, (base_offset) + (row_offset) * 4);

#define MOVD2A_8_ROWS(base_offset, row_offset) \
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + (row_offset) * 8, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, (base_offset) + (row_offset) * 8);

#define MOVD2A_16_ROWS(base_offset) \
    MOVD2A_8_ROWS(base_offset, 0)   \
    MOVD2A_8_ROWS(base_offset, 1)   \
    MOVD2A_8_ROWS(base_offset, 2)   \
    MOVD2A_8_ROWS(base_offset, 3)   \
    MOVD2A_8_ROWS(base_offset, 4)   \
    MOVD2A_8_ROWS(base_offset, 5)   \
    MOVD2A_8_ROWS(base_offset, 6)   \
    MOVD2A_8_ROWS(base_offset, 7)

#define MOVD2A_FACE(base_offset, face)         \
    MOVD2A_4_ROWS(base_offset, (face) * 4 + 0) \
    MOVD2A_4_ROWS(base_offset, (face) * 4 + 1) \
    MOVD2A_4_ROWS(base_offset, (face) * 4 + 2) \
    MOVD2A_4_ROWS(base_offset, (face) * 4 + 3)

// A whole slot moves in eight-row steps; a one- or two-face tile keeps four-row moves, which read faster there.
#define MOVD2A_FACES(base_offset, num_faces) \
    if ((num_faces) > 2)                     \
    {                                        \
        MOVD2A_16_ROWS(base_offset)          \
    }                                        \
    else                                     \
    {                                        \
        MOVD2A_FACE(base_offset, 0)          \
        if ((num_faces) > 1)                 \
        {                                    \
            MOVD2A_FACE(base_offset, 1)      \
        }                                    \
    }

/**
 * @brief Execute GAPOOL operations with optional high fidelity phases
 *
 * Executes the appropriate number of GAPOOL instructions based on fidelity level.
 * For high fidelity (math_fidelity > LoFi), executes Number of Fidelity Phases - 1
 * iterations with ADDR_MOD_2, followed by one final GAPOOL with ADDR_MOD_0.
 *
 * @tparam math_fidelity: MathFidelity value LoFi, HiFi2, HiFi3, HiFi4
 */
template <MathFidelity math_fidelity, std::uint32_t last_addr_mod = ADDR_MOD_0>
inline void execute_high_fidelity_gapool()
{
    if constexpr (math_fidelity >= MathFidelity::HiFi2)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }
    if constexpr (math_fidelity >= MathFidelity::HiFi3)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }
    if constexpr (math_fidelity >= MathFidelity::HiFi4)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }

    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, last_addr_mod, p_gpool::INDEX_DIS, 0);
}

/**
 * @brief Move destination tile to source registers for mul_reduce_scalar
 *
 * Moves a whole tile to source A, or its first 16 rows to source B.
 *
 * @tparam binary_reuse_dest Direction: DEST_TO_SRCA or DEST_TO_SRCB
 * @param idst Destination tile index (0-7)
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_mul_reduce_scalar_move_dest_to_src_([[maybe_unused]] std::uint32_t idst = 0)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        if (idst == 0)
        {
            TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        }

        math::srca_bank_wait();

        switch (idst)
        {
            case 0:
                MOVD2A_16_ROWS(0);
                break;
            case 1:
                MOVD2A_16_ROWS(64);
                break;
            case 2:
                MOVD2A_16_ROWS(128);
                break;
            case 3:
                MOVD2A_16_ROWS(192);
                break;
            case 4:
                MOVD2A_16_ROWS(256);
                break;
            case 5:
                MOVD2A_16_ROWS(320);
                break;
            case 6:
                MOVD2A_16_ROWS(384);
                break;
            case 7:
                MOVD2A_16_ROWS(448);
                break;
            default:
                break;
        }
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        math::srcb_bank_wait();
        // dst may have been written by the SFPU just before this move (e.g. a fill). The MATH bit in the
        // bank wait only tracks FPU writes, so also drain the SFPU before the FPU copies dst into SrcB.
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);

        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_BD);

        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 4, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 4);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 8, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 8);
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 12, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 12);
    }
}

inline void _llk_math_mul_reduce_scalar_move_faces_to_srca_(const std::uint32_t idst, const ckernel::TensorShape tensor_shape)
{
    // Later moves find the bank still valid, and MOVD2A itself waits for pending DEST writes.
    if (idst == 0)
    {
        // dest_offset_id is 0 or 1 and flips every section: a shift, not a select, keeps it off the branch predictor.
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dest_offset_id * DEST_REGISTER_HALF_SIZE);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        math::srca_bank_wait();
    }

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    switch (idst)
    {
        case 0:
            MOVD2A_FACES(0, num_faces);
            break;
        case 1:
            MOVD2A_FACES(64, num_faces);
            break;
        case 2:
            MOVD2A_FACES(128, num_faces);
            break;
        case 3:
            MOVD2A_FACES(192, num_faces);
            break;
        case 4:
            MOVD2A_FACES(256, num_faces);
            break;
        case 5:
            MOVD2A_FACES(320, num_faces);
            break;
        case 6:
            MOVD2A_FACES(384, num_faces);
            break;
        case 7:
            MOVD2A_FACES(448, num_faces);
            break;
        default:
            break;
    }
}

// A 16x32 product takes 32 rows, so two fit one 64-row DEST slot.
inline bool _llk_math_mul_reduce_scalar_shares_slot_(const ckernel::TensorShape tensor_shape)
{
    return tensor_shape.face_r_dim == ckernel::MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 2;
}

// Multiplies one 16x32 product into rows 32 * (product_index % 2) of DEST slot product_index / 2.
template <MathFidelity math_fidelity>
inline void _llk_math_mul_reduce_scalar_mul_half_slot_(const ckernel::TensorShape tensor_shape, const std::uint32_t product_index)
{
    LLK_ASSERT(_llk_math_mul_reduce_scalar_shares_slot_(tensor_shape), "the half-slot multiply is for 16x32 products");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_mul_reduce_scalar_mul_half_slot_", tensor_shape);
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "mul_half_slot: Src zero-substitution flag is not in DEFAULT state");
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::SrcRegs>(product_index);
    const std::uint32_t fidelity_loop = is_high_fidelity(math_fidelity) ? tensor_shape.total_num_faces() : 1;
#pragma GCC unroll 0
    for (std::uint32_t i = 0; i < fidelity_loop; i++)
    {
        ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}

// share_slots must match how the products were written.
template <bool share_slots>
inline void _llk_math_mul_reduce_scalar_move_product_(const std::uint32_t i, const std::uint32_t num_products, const ckernel::TensorShape tensor_shape)
{
    if (!share_slots || !_llk_math_mul_reduce_scalar_shares_slot_(tensor_shape))
    {
        _llk_math_mul_reduce_scalar_move_faces_to_srca_(i, tensor_shape);
    }
    else if ((i & 1) == 0)
    {
        _llk_math_mul_reduce_scalar_move_faces_to_srca_(i / 2, i + 1 < num_products ? ckernel::DEFAULT_TENSOR_SHAPE : tensor_shape);
    }
}

/**
 * @brief Configure address modifiers for mul_reduce_scalar operations
 *
 * Sets up address modifiers for source A, source B, destination, and fidelity
 * registers based on the specified math fidelity level.
 *
 * @tparam MathFidelity math_fidelity -> MathFidelity value LoFi, HiFi2, HiFi3, HiFi4
 */
template <MathFidelity math_fidelity>
inline void mul_reduce_scalar_configure_addrmod()
{
    constexpr std::uint32_t FIDELITY_INCREMENT = 1;
    constexpr bool HIGH_FIDELITY               = is_high_fidelity(math_fidelity);

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
    addr_mod_t {.srca = {.incr = 16}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.clr = 1}}.set(ADDR_MOD_1);

    if constexpr (HIGH_FIDELITY)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = FIDELITY_INCREMENT}}.set(ADDR_MOD_2);
    }
}

/**
 * @brief Initialize reduce for mul_reduce_scalar operation
 *
 * Configures address modifiers and resets counters for the fused
 * multiply-reduce-scalar operation.
 *
 * @tparam is_fp32_dest_acc_en If true, enables FP32 destination accumulation
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 * @tparam enforce_fp32_accumulation If true, enforces FP32 accumulation (requires is_fp32_dest_acc_en)
 */
template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void _llk_math_mul_reduce_scalar_init_()
{
    mul_reduce_scalar_configure_addrmod<math_fidelity>();

    if constexpr (enforce_fp32_accumulation)
    {
        static_assert(is_fp32_dest_acc_en, "FP32 Dest must be enabled for FP32 accumulation");
    }
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
}

/**
 * @brief Perform column reduction for mul_reduce_scalar (accumulates across tiles)
 *
 * Performs column-wise reduction that accumulates results into the specified
 * destination tile. Used in a loop to accumulate multiple input tiles.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 * @param tensor_shape Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile)
 * @note Without tile_setup, the SFPU must have drained and the DEST address must already be at the accumulation tile.
 */
template <MathFidelity math_fidelity, bool tile_setup = true>
inline void _llk_math_mul_reduce_column_(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    const std::uint32_t num_row_tiles = tensor_shape.num_faces_r_dim;

    // dst[dst_index] may have just been zeroed through the SFPU; GAPOOL accumulates into it, so drain first.
    if constexpr (tile_setup)
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }

    // ADDR_MOD_1 steps SrcA to the next face after the last fidelity pass, in face order.
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
#pragma GCC unroll 4
    for (std::uint32_t face = 0; face < num_faces; face++)
    {
        execute_high_fidelity_gapool<math_fidelity, ADDR_MOD_1>();
    }
    if (num_row_tiles > 1)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AD);
    }
    else
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// Zeroes the face the column passes pool into; call once the SFPU has drained.
template <bool is_fp32_dest_acc_en>
inline void _llk_math_mul_reduce_scalar_clear_pool_face_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    constexpr std::uint32_t tiles_per_bank = is_fp32_dest_acc_en ? MAX_TILES_IN_HALF_DEST >> 1 : MAX_TILES_IN_HALF_DEST;
    TT_ZEROACC(p_zeroacc::CLR_16, is_fp32_dest_acc_en, 0, ADDR_MOD_0, get_dest_index_in_faces(dst_index & (tiles_per_bank - 1), 0));
}

/**
 * @brief Perform final scalar reduction for mul_reduce_scalar
 *
 * Collapses a column-reduced tile into a single scalar value stored in the
 * first element of the output tile.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 */
template <MathFidelity math_fidelity>
inline void _llk_math_mul_reduce_scalar_()
{
    math::srcb_bank_wait();

    // Copy row 0 from dest to srcB (rows 16-31 as scratch) and transpose
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_GATESRCRST(0b1, 0b1);
    TTI_TRNSPSRCB;
    TTI_GATESRCRST(0b1, 0b1);

    // Copy all 16 rows from srcB to srcA
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
    TTI_GATESRCRST(0b1, 0b1);

    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 0);

    execute_high_fidelity_gapool<math_fidelity>();
}

/**
 * @brief Clear data valid flags after mul_reduce_scalar operation
 *
 * Clears DVALID flags and resets all counters. Should be called after
 * mul_reduce_scalar operations are complete.
 */
inline void _llk_math_mul_reduce_scalar_clear_dvalid_()
{
    TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}

/*************************************************************************
 * Interleaved row, LoFi: each face's SrcA/SrcB bank turn multiplies that face of tile i and pools the same face of
 * product i - 1, which MOVD2A puts in SrcA rows 16 to 31 beside the unpacked face (rows 0 to 15). The scaler sits in
 * SrcB rows 16 to 19 of both banks. The pooling order is the per-tile row's: product by product, faces in order.
 *************************************************************************/

// Faces of 16 rows, two (16x32, products 32 rows apart as in the shared slots) or four (32x32), and two or more tiles.
inline bool _llk_math_mul_reduce_scalar_interleaves_(const ckernel::TensorShape tensor_shape, const std::uint32_t num_tiles)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    return num_tiles >= 2 && tensor_shape.face_r_dim == ckernel::MAX_FACE_R_DIM && tensor_shape.num_faces_c_dim == 2 &&
           (num_faces == 4 || (num_faces == 2 && tensor_shape.num_faces_r_dim == 1));
}

template <bool is_fp32_dest_acc_en>
inline void _llk_math_mul_reduce_scalar_interleaved_init_()
{
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MathFidelity::LoFi>();
    addr_mod_t {.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 0}}.set(ADDR_MOD_3);
    addr_mod_t {.srca = {.incr = 24}, .srcb = {.incr = 8}, .dest = {.incr = 0}}.set(ADDR_MOD_4);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dest_offset_id * DEST_REGISTER_HALF_SIZE);
}

template <std::uint32_t cur>
inline void _llk_math_mul_reduce_scalar_mul_face_()
{
    TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, cur);
    TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, cur + 8);
}

template <std::uint32_t prev>
inline void _llk_math_mul_reduce_scalar_move_face_()
{
    TTI_MOVD2A(0, 16, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev);
    TTI_MOVD2A(0, 24, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 8);
}

inline void _llk_math_mul_reduce_scalar_pool_face_()
{
    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
}

// One face turn: pools face f of the product at DEST row prev and multiplies face f of the tile at row cur.
template <std::uint32_t prev, std::uint32_t cur, std::uint32_t f>
inline void _llk_math_mul_reduce_scalar_turn_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    _llk_math_mul_reduce_scalar_move_face_<prev + 16 * f>();
    _llk_math_mul_reduce_scalar_mul_face_<cur + 16 * f>();
    _llk_math_mul_reduce_scalar_pool_face_();
}

template <std::uint32_t num_faces, std::uint32_t i>
inline void _llk_math_mul_reduce_scalar_interleaved_tile_ct_(const float scaler)
{
    constexpr std::uint32_t rows = 16 * num_faces;
    constexpr std::uint32_t cur  = rows * i;
    if constexpr (i == 0)
    {
        _llk_math_mul_reduce_scalar_mul_face_<0>();
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        _llk_math_mul_reduce_scalar_mul_face_<16>();
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        if constexpr (num_faces == 4)
        {
            _llk_math_mul_reduce_scalar_mul_face_<32>();
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
            _llk_math_mul_reduce_scalar_mul_face_<48>();
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
    }
    else
    {
        constexpr std::uint32_t prev = rows * (i - 1);
        if constexpr (i == 1)
        {
            // The scaler goes to rows 0 to 3 of the pool tile, once product 0's face 0 has left them, and to rows 8 to 11
            // for the other bank and the last product; only pool row 0 reaches the result.
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
            _llk_math_mul_reduce_scalar_move_face_<prev>();
            _llk_math_eltwise_sfpu_start_(0);
            ckernel::sfpu::_calculate_fill_<false, 2>(scaler);
            TTI_INCRWC(0, 4, 0, 0);
            ckernel::sfpu::_calculate_fill_<false, 2>(scaler);
            _llk_math_eltwise_sfpu_done_();
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 0);
            _llk_math_mul_reduce_scalar_mul_face_<cur>();
            _llk_math_mul_reduce_scalar_pool_face_();

            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
            TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 8);
            _llk_math_mul_reduce_scalar_move_face_<prev + 16>();
            _llk_math_mul_reduce_scalar_mul_face_<cur + 16>();
            _llk_math_mul_reduce_scalar_pool_face_();
        }
        else
        {
            _llk_math_mul_reduce_scalar_turn_<prev, cur, 0>();
            _llk_math_mul_reduce_scalar_turn_<prev, cur, 1>();
        }
        if constexpr (num_faces == 4)
        {
            _llk_math_mul_reduce_scalar_turn_<prev, cur, 2>();
            _llk_math_mul_reduce_scalar_turn_<prev, cur, 3>();
        }
    }
}

template <std::uint32_t num_faces>
inline void _llk_math_mul_reduce_scalar_interleaved_tile_n_(const std::uint32_t i, const float scaler)
{
    switch (i)
    {
        case 0: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<num_faces, 0>(scaler); break;
        case 1: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<num_faces, 1>(scaler); break;
        case 2: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<num_faces, 2>(scaler); break;
        case 3: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<num_faces, 3>(scaler); break;
        case 4: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<num_faces, 4>(scaler); break;
        case 5: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<num_faces, 5>(scaler); break;
        case 6: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<num_faces, 6>(scaler); break;
        case 7: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<num_faces, 7>(scaler); break;
        default:
            if constexpr (num_faces == 2)
            {
                switch (i)
                {
                    case 8: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<2, 8>(scaler); break;
                    case 9: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<2, 9>(scaler); break;
                    case 10: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<2, 10>(scaler); break;
                    case 11: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<2, 11>(scaler); break;
                    case 12: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<2, 12>(scaler); break;
                    case 13: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<2, 13>(scaler); break;
                    case 14: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<2, 14>(scaler); break;
                    case 15: _llk_math_mul_reduce_scalar_interleaved_tile_ct_<2, 15>(scaler); break;
                    default: break;
                }
            }
            break;
    }
}

// Multiplies tile i and, from tile 1 on, pools product i - 1 in the same face turns.
inline void _llk_math_mul_reduce_scalar_interleaved_tile_(const std::uint32_t i, const ckernel::TensorShape tensor_shape, const float scaler)
{
    if (tensor_shape.total_num_faces() == 4)
    {
        _llk_math_mul_reduce_scalar_interleaved_tile_n_<4>(i, scaler);
    }
    else
    {
        _llk_math_mul_reduce_scalar_interleaved_tile_n_<2>(i, scaler);
    }
}

// Pools the last product in the bank the unpacker zeroed, then collapses the pool row to the scalar.
inline void _llk_math_mul_reduce_scalar_interleaved_tail_(const std::uint32_t num_tiles, const ckernel::TensorShape tensor_shape)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 8);
    const std::uint32_t last = (num_faces == 4 ? 64 : 32) * (num_tiles - 1);
    if (num_faces == 4)
    {
        switch (last)
        {
            case 64: MOVD2A_16_ROWS(64); break;
            case 128: MOVD2A_16_ROWS(128); break;
            case 192: MOVD2A_16_ROWS(192); break;
            case 256: MOVD2A_16_ROWS(256); break;
            case 320: MOVD2A_16_ROWS(320); break;
            case 384: MOVD2A_16_ROWS(384); break;
            case 448: MOVD2A_16_ROWS(448); break;
            default: break;
        }
    }
    else
    {
        switch (last)
        {
            case 32: MOVD2A_FACE(32, 0) MOVD2A_FACE(32, 1) break;
            case 64: MOVD2A_FACE(64, 0) MOVD2A_FACE(64, 1) break;
            case 96: MOVD2A_FACE(96, 0) MOVD2A_FACE(96, 1) break;
            case 128: MOVD2A_FACE(128, 0) MOVD2A_FACE(128, 1) break;
            case 160: MOVD2A_FACE(160, 0) MOVD2A_FACE(160, 1) break;
            case 192: MOVD2A_FACE(192, 0) MOVD2A_FACE(192, 1) break;
            case 224: MOVD2A_FACE(224, 0) MOVD2A_FACE(224, 1) break;
            case 256: MOVD2A_FACE(256, 0) MOVD2A_FACE(256, 1) break;
            case 288: MOVD2A_FACE(288, 0) MOVD2A_FACE(288, 1) break;
            case 320: MOVD2A_FACE(320, 0) MOVD2A_FACE(320, 1) break;
            case 352: MOVD2A_FACE(352, 0) MOVD2A_FACE(352, 1) break;
            case 384: MOVD2A_FACE(384, 0) MOVD2A_FACE(384, 1) break;
            case 416: MOVD2A_FACE(416, 0) MOVD2A_FACE(416, 1) break;
            case 448: MOVD2A_FACE(448, 0) MOVD2A_FACE(448, 1) break;
            case 480: MOVD2A_FACE(480, 0) MOVD2A_FACE(480, 1) break;
            default: break;
        }
    }
    _llk_math_mul_reduce_column_<MathFidelity::LoFi, false>(0, tensor_shape);
    _llk_math_mul_reduce_scalar_<MathFidelity::LoFi>();
    _llk_math_mul_reduce_scalar_clear_dvalid_();
}

// v2 (scratch): faces f and f + 2 move together in turn f, so turns 2 and 3 pool rows 32 to 47 with no move.
template <std::uint32_t prev, std::uint32_t cur, std::uint32_t f>
inline void _llk_math_mul_reduce_scalar_turn2_move_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    TTI_MOVD2A(0, 16, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 16 * f);
    TTI_MOVD2A(0, 24, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 16 * f + 8);
    TTI_MOVD2A(0, 32, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 16 * f + 32);
    TTI_MOVD2A(0, 40, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 16 * f + 40);
    _llk_math_mul_reduce_scalar_mul_face_<cur + 16 * f>();
    _llk_math_mul_reduce_scalar_pool_face_();
}

template <std::uint32_t cur, std::uint32_t f>
inline void _llk_math_mul_reduce_scalar_turn2_pool_()
{
    TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, cur + 16 * f);
    TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_4, cur + 16 * f + 8);
    _llk_math_mul_reduce_scalar_pool_face_();
}

template <std::uint32_t i>
inline void _llk_math_mul_reduce_scalar_interleaved2_tile_ct_(const float scaler)
{
    constexpr std::uint32_t cur = 64 * i;
    if constexpr (i == 0)
    {
        _llk_math_mul_reduce_scalar_interleaved_tile_ct_<4, 0>(scaler);
    }
    else
    {
        constexpr std::uint32_t prev = 64 * (i - 1);
        if constexpr (i == 1)
        {
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
            TTI_MOVD2A(0, 16, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev);
            TTI_MOVD2A(0, 24, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 8);
            TTI_MOVD2A(0, 32, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 32);
            TTI_MOVD2A(0, 40, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 40);
            _llk_math_eltwise_sfpu_start_(0);
            ckernel::sfpu::_calculate_fill_<false, 2>(scaler);
            TTI_INCRWC(0, 4, 0, 0);
            ckernel::sfpu::_calculate_fill_<false, 2>(scaler);
            _llk_math_eltwise_sfpu_done_();
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 0);
            _llk_math_mul_reduce_scalar_mul_face_<cur>();
            _llk_math_mul_reduce_scalar_pool_face_();

            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD | p_stall::SRCB_VLD);
            TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 8);
            TTI_MOVD2A(0, 16, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 16);
            TTI_MOVD2A(0, 24, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 24);
            TTI_MOVD2A(0, 32, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 48);
            TTI_MOVD2A(0, 40, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 56);
            _llk_math_mul_reduce_scalar_mul_face_<cur + 16>();
            _llk_math_mul_reduce_scalar_pool_face_();
        }
        else
        {
            _llk_math_mul_reduce_scalar_turn2_move_<prev, cur, 0>();
            _llk_math_mul_reduce_scalar_turn2_move_<prev, cur, 1>();
        }
        _llk_math_mul_reduce_scalar_turn2_pool_<cur, 2>();
        _llk_math_mul_reduce_scalar_turn2_pool_<cur, 3>();
    }
}

inline void _llk_math_mul_reduce_scalar_interleaved2_tile_(const std::uint32_t i, const float scaler)
{
    switch (i)
    {
        case 0: _llk_math_mul_reduce_scalar_interleaved2_tile_ct_<0>(scaler); break;
        case 1: _llk_math_mul_reduce_scalar_interleaved2_tile_ct_<1>(scaler); break;
        case 2: _llk_math_mul_reduce_scalar_interleaved2_tile_ct_<2>(scaler); break;
        case 3: _llk_math_mul_reduce_scalar_interleaved2_tile_ct_<3>(scaler); break;
        case 4: _llk_math_mul_reduce_scalar_interleaved2_tile_ct_<4>(scaler); break;
        case 5: _llk_math_mul_reduce_scalar_interleaved2_tile_ct_<5>(scaler); break;
        case 6: _llk_math_mul_reduce_scalar_interleaved2_tile_ct_<6>(scaler); break;
        case 7: _llk_math_mul_reduce_scalar_interleaved2_tile_ct_<7>(scaler); break;
        default: break;
    }
}
