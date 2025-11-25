// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

template <EltwiseBinaryType ELTWISE_BINARY_TYPE, uint8_t CLR_SRC, uint8_t EN_DST_ACCUM, uint8_t SRCB_BROADCAST_TYPE, uint8_t ADDR_MOD>
constexpr uint eltwise_binary_func()
{
    if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWADD)
    {
        return TT_OP_ELWADD(CLR_SRC, EN_DST_ACCUM, SRCB_BROADCAST_TYPE, ADDR_MOD, 0);
    }
    else if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWSUB)
    {
        return TT_OP_ELWSUB(CLR_SRC, EN_DST_ACCUM, SRCB_BROADCAST_TYPE, ADDR_MOD, 0);
    }
    else
    {
        return TT_OP_ELWMUL(CLR_SRC, EN_DST_ACCUM, SRCB_BROADCAST_TYPE, ADDR_MOD, 0);
    }
}

//----------------------
// Direct Indexing Method
//----------------------
template <EltwiseBinaryType ELTWISE_BINARY_TYPE>
inline uint eltwise_di_binary_func(
    uint8_t CLR_SRC, uint8_t EN_DST_ACCUM, uint8_t SRCB_BROADCAST_TYPE, uint8_t SRCB_ADDR, uint8_t SRCA_ADDR, uint8_t ADDR_MOD, uint8_t DST_ADDR)
{
    uint8_t INSTR_MOD = ((SRCB_BROADCAST_TYPE << 0) | (EN_DST_ACCUM << 2));
    if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWADD)
    {
        return TT_ELWADDDI(CLR_SRC, INSTR_MOD, SRCB_ADDR, SRCA_ADDR, ADDR_MOD, DST_ADDR);
    }
    else if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWSUB)
    {
        return TT_ELWSUBDI(CLR_SRC, INSTR_MOD, SRCB_ADDR, SRCA_ADDR, ADDR_MOD, DST_ADDR);
    }
    else
    {
        return TT_ELWMULDI(CLR_SRC, INSTR_MOD, SRCB_ADDR, SRCA_ADDR, ADDR_MOD, DST_ADDR);
    }
}

//----------------------
/**
 * @brief Sets up mop config for elementwise binary operations
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam MATH_FIDELITY: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when input is Tf32 format
 * @param tile_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_mop_config_(const TileShape& tile_shape)
{
    const uint32_t total_num_rows_per_tile = tile_shape.num_faces * tile_shape.face_r_dim;
    const uint32_t MOP_OUTER_LOOP          = (total_num_rows_per_tile >> math_rows_log2(ELTWISE_MATH_ROWS));
    constexpr uint32_t MOP_INNER_LOOP      = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : static_cast<uint32_t>(MATH_FIDELITY_TYPE);
    constexpr bool math_fidelity_enable    = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    static_assert(!(math_fidelity_enable && ELTWISE_BINARY_TYPE != EltwiseBinaryType::ELWMUL), "Math fidelity larger than LoFi only works with Eltwise MUL");
    const uint32_t EN_DST_ACC_EN = math_fidelity_enable;

    constexpr uint8_t addrmod_fid = math_fidelity_enable ? ADDR_MOD_2 : ADDR_MOD_0;
    constexpr static uint eltwise_binary_op =
        eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_NONE, EN_DST_ACC_EN, p_elwise::SRCB_NO_BCAST, addrmod_fid>();
    constexpr static uint eltwise_binary_op_clr_valid =
        eltwise_binary_func<ELTWISE_BINARY_TYPE, p_setrwc::CLR_AB, EN_DST_ACC_EN, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1>();
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, eltwise_binary_op);
    temp.set_last_outer_loop_instr(eltwise_binary_op_clr_valid);

    if (math_fidelity_enable)
    {
        constexpr static uint eltwise_binary_op_clr_fidelity =
            eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_NONE, EN_DST_ACC_EN, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0>();
        temp.set_last_inner_loop_instr(eltwise_binary_op_clr_fidelity); // clear math fidelity
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

//----------------------
// Direct Indexing Method
//----------------------
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_di_binary_mop_config_(const TileShape& tile_shape)
{
    const uint32_t total_num_rows_per_tile = tile_shape.num_faces * tile_shape.face_r_dim;
    const uint32_t REPLAY_BUF_LEN          = (total_num_rows_per_tile >> math_rows_log2(ELTWISE_MATH_ROWS));
    const uint32_t MOP_INNER_LOOP          = static_cast<uint32_t>(MATH_FIDELITY_TYPE) + 1;
    constexpr bool math_fidelity_enable    = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    static_assert(!(math_fidelity_enable && ELTWISE_BINARY_TYPE != EltwiseBinaryType::ELWMUL), "Math fidelity larger than LoFi only works with Eltwise MUL");
    const uint32_t EN_DST_ACC_EN = math_fidelity_enable;

    load_replay_buf(
        0u,
        REPLAY_BUF_LEN,
        false,
        0,
        0,
        [&]()
        {
            for (uint32_t i = 0; i < REPLAY_BUF_LEN - 1; ++i)
            {
                eltwise_di_binary_func<ELTWISE_BINARY_TYPE>(
                    p_elwise::CLR_NONE,
                    EN_DST_ACC_EN,
                    p_elwise::SRCB_NO_BCAST,
                    (i * ELTWISE_MATH_ROWS) >> 2, // Srcb addr
                    (i * ELTWISE_MATH_ROWS) >> 2, // Srca addr
                    0x0,
                    (i * ELTWISE_MATH_ROWS) >> 2); // Dest addr
            }

            if (math_fidelity_enable)
            {
                eltwise_di_binary_func<ELTWISE_BINARY_TYPE>(
                    p_elwise::CLR_NONE,
                    EN_DST_ACC_EN,
                    p_elwise::SRCB_NO_BCAST,
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2,  // Srcb addr
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2,  // Srca addr
                    ADDR_MOD_1,                                       // Increment Fidelty
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2); // Dest addr
            }
            else
            {
                eltwise_di_binary_func<ELTWISE_BINARY_TYPE>(
                    p_setrwc::CLR_AB,
                    EN_DST_ACC_EN,
                    p_elwise::SRCB_NO_BCAST,
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2,  // Srcb addr
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2,  // Srca addr
                    ADDR_MOD_1,                                       // Increment Fidelty
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2); // Dest addr
            }
        });

    ckernel_template temp(1 /* outer loop */, MOP_INNER_LOOP, TT_OP_REPLAY(0, REPLAY_BUF_LEN, 0, 0, 0, 0));

    if (math_fidelity_enable)
    {
        temp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, p_setrwc::SET_ABD_F));
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

//----------------------

/**
 * @brief Sets up addrmods for elementwise binary operations
 * @tparam MATH_FIDELITY: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when input is Tf32 format
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_addrmod_()
{
    constexpr bool math_fidelity_enable = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    // For ELWADD/SUB/MUL, can increment source
    //  and dest registers
    addr_mod_t {
        .srca     = {.incr = ELTWISE_MATH_ROWS},
        .srcb     = {.incr = ELTWISE_MATH_ROWS},
        .dest     = {.incr = ELTWISE_MATH_ROWS},
        .fidelity = {.incr = 0, .clr = math_fidelity_enable}}
        .set(ADDR_MOD_0);

    // Reset Src counters, inc dest
    addr_mod_t {.srca = {.clr = 1}, .srcb = {.clr = 1}, .dest = {.incr = ELTWISE_MATH_ROWS}, .fidelity = {.incr = 0, .clr = math_fidelity_enable}}.set(
        ADDR_MOD_1);

    if constexpr (math_fidelity_enable)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 1}}.set(ADDR_MOD_2);
    }
}

//----------------------
// Direct Indexing Method
//----------------------
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_di_binary_addrmod_()
{
    constexpr bool math_fidelity_enable = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT    = math_fidelity_enable ? 1 : 0;
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_1);
}

//----------------------
/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam MATH_FIDELITY: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when input is Tf32 format
 * @param tile_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE, bool EN_DI = false>
inline void _llk_math_eltwise_binary_init_(const TileShape& tile_shape)
{
    if constexpr (EN_DI)
    {
        _llk_math_eltwise_di_binary_addrmod_<MATH_FIDELITY_TYPE>();
        _llk_math_eltwise_di_binary_mop_config_<ELTWISE_BINARY_TYPE, MATH_FIDELITY_TYPE>(tile_shape);
    }
    else
    {
        _llk_math_eltwise_binary_addrmod_<MATH_FIDELITY_TYPE>();
        _llk_math_eltwise_binary_mop_config_<ELTWISE_BINARY_TYPE, MATH_FIDELITY_TYPE>(tile_shape);
    }

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @param tile_idx: Tile index into the destination register.
 * If dest reg in float16 mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in full mode
 * If dest reg in float32 mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 */
inline void _llk_math_eltwise_binary_(const uint32_t tile_idx)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);

    // Run MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
