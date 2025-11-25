// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

template <PoolType POOL_TYPE, uint8_t CLR_SRC, uint8_t POOL_SIZE, uint8_t ADDR_MOD, uint8_t MAX_POOL_IDX_EN, uint8_t DST_ADDR>
void tti_pool_instr_func()
{
    if constexpr (POOL_TYPE == PoolType::MAX)
    {
        TTI_GMPOOL(CLR_SRC, POOL_SIZE, ADDR_MOD, MAX_POOL_IDX_EN, DST_ADDR);
    }
    else
    {
        TTI_GAPOOL(CLR_SRC, POOL_SIZE, ADDR_MOD, MAX_POOL_IDX_EN, DST_ADDR);
    }
}

/**
 * @brief Sets up mop config for reduce column operations
 *For reduce Col, in a 32 x 32 tile, faces layout would be the following:
 *--------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * In order to get 1x32 row output (which means 2 output faces, 1x16 row each), then Face 0 + Face 2 are pooled together, and Face 1 and Face 3 are pooled
 *together
 * @tparam POOL_TYPE: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types, shows how many loops
 * to use full precision with of Source register datums with multiplies, values = [LoFi, HiFi2, HiFi3, HiFi4]
 * @param tile_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_col_mop_config_(const TileShape& tile_shape)
{
    // So Face 0 reduce, dest counter += 16, Face 1 reduce, dest counter reset to 0
    // then Face 2 reduce (which includes Face 0 reduce result in dest), dest counter += 16, Face 3 reduce(which includes Face 1 reduce result in dest at index
    // 16)
    const uint32_t MOP_OUTER_LOOP      = 1;
    const uint32_t MOP_INNER_LOOP      = (tile_shape.num_faces >= 2) ? (tile_shape.num_faces >> 1) : tile_shape.num_faces;
    constexpr uint NUM_FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 0 : static_cast<uint32_t>(MATH_FIDELITY_TYPE) - 1;
    constexpr bool RUN_FID_LOOPS       = (MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi && (POOL_TYPE == PoolType::AVG || POOL_TYPE == PoolType::SUM));
    const uint replay_buf_len          = 2 + (2 * NUM_FIDELITY_PHASES);

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        []
        {
            if constexpr (RUN_FID_LOOPS)
            {
                for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0x0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0x0>();

            if constexpr (RUN_FID_LOOPS)
            {
                for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0x0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0x0>();
        });

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up mop config for reduce row operations
 *For reduce row, in a 32 x 32 tile, faces layout would be the following:
 *--------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * In order to get 32x1 column output (which means 2 output faces, 16x1 col each),
 * then all faces are transposed, Face 0 + Face 1 are pooled together, and Face 2 & 3 are pooled together
 * @tparam POOL_TYPE: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types, shows how many loops
 * to use full precision with of Source register datums with multiplies, values = [LoFi, HiFi2, HiFi3, HiFi4]
 * @param tile_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_row_mop_config_(const TileShape& tile_shape)
{
    constexpr bool RUN_FID_LOOPS       = (MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi && (POOL_TYPE == PoolType::AVG || POOL_TYPE == PoolType::SUM));
    constexpr uint NUM_FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 0 : static_cast<uint32_t>(MATH_FIDELITY_TYPE) - 1;
    constexpr uint32_t MOP_OUTER_LOOP  = 1;
    constexpr uint32_t MOP_INNER_LOOP  = 1;
    // Replay buf max len is 32, NUM_FIDELITY_PHASES will be larger than 3, hypothetical limit of 19 + 12 = 31
    constexpr uint replay_buf_len = 19 + (4 * NUM_FIDELITY_PHASES);

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [tile_shape]
        {
            // Each face is transposed in the unpacker, and then faces 0 & 1 are pooled together
            if constexpr (RUN_FID_LOOPS)
            {
                for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

            if constexpr (RUN_FID_LOOPS)
            {
                for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

            // This will clear AB counters to 0, and cr d is also 0
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);

            // Src B can only transpose rows [16-31], and output them at [32-47]
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, 0);

            // Required for accumulating on multiple tiles at a time, accumulation can only work
            // on row not column
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);

            // Copy transposed rows in SrcB from [32 - 47] to dest rows [0 - 16]
            TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);

            // Increment dest by 32
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 32, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_A, p_setrwc::CR_D, 0, p_setrwc::SET_B);

            /////////////////////
            // Second face Row //
            /////////////////////
            // Each face is transposed in the unpacker, and then faces 0 & 1 are pooled together
            if constexpr (RUN_FID_LOOPS)
            {
                for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

            if constexpr (RUN_FID_LOOPS)
            {
                for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();

            // This will clear AB counters to 0, and cr d is 32
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, p_setrwc::SET_AB);

            // Src B can only transpose rows [16-31], and output them at [32-47]
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, 0);

            // Required for accumulating on multiple tiles at a time, accumulation can only work
            // on row not column
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0, 0);

            // Copy transposed rows in SrcB from [32 - 47] to dest rows [32 - 48]
            TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);
            TTI_ELWADDDI(p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, ADDR_MOD_1, 0x0);
            // Set counters back to 0
            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_BD);
        });

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up mop config for reduce scalar operations
 * For reduce scalar, in a 32 x 32 tile, faces layout would be the following:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * All 4 faces will be pooled together , result will be a single reduce datum placed in datum 0 of the tile idx
 * @tparam POOL_TYPE: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types, shows how many loops
 * to use full precision with of Source register datums with multiplies, values = [LoFi, HiFi2, HiFi3, HiFi4]
 * @param tile_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_scalar_mop_config_(const TileShape& tile_shape)
{
    constexpr uint32_t MOP_OUTER_LOOP  = 1;
    constexpr uint32_t MOP_INNER_LOOP  = 1;
    constexpr uint NUM_FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 0 : static_cast<uint32_t>(MATH_FIDELITY_TYPE) - 1;
    const uint replay_buf_len          = 6 + tile_shape.num_faces - 1 + ((tile_shape.num_faces - 1) * NUM_FIDELITY_PHASES) + (2 * NUM_FIDELITY_PHASES);
    constexpr bool RUN_FID_LOOPS       = (MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi && (POOL_TYPE == PoolType::AVG || POOL_TYPE == PoolType::SUM));

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [tile_shape]
        {
            // Set up a dest addr to output temp results into, has to be less than 64 (to not write into next tile)
            // but also has to be greater than 0 (where results are expected)
            constexpr uint32_t scratch_dst_addr = 16;

            // Pool all faces together (default 4 faces), this will generate 1x16 row of result at dst index scratch_dst_addr
            // No src/dest counters are incremented
            for (uint face = 0; face < tile_shape.num_faces - 1; face++)
            {
                if constexpr (RUN_FID_LOOPS)
                {
                    for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                    {
                        tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, scratch_dst_addr>();
                    }
                }
                tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, scratch_dst_addr>();
            }

            if constexpr (RUN_FID_LOOPS)
            {
                for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, scratch_dst_addr>();
                }
            }
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, scratch_dst_addr>();

            // Only SrcB transpose can be done from Math
            // Rows [16:31] of SrcB are transposed, then written back into rows [32:47] of SrcB
            // Following will move 1x16 pool result to SrcB to be transposed into 16 rows
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 1, scratch_dst_addr);

            // copy over all 16 rows from B to A
            TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_8_ROWS, p_movb2a::SRCB_ROW32_OFFSET + 0);
            TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_8_ROWS, p_movb2a::SRCB_ROW32_OFFSET + 8);

            // zero out scratch in dest
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, scratch_dst_addr);

            if constexpr (RUN_FID_LOOPS)
            {
                for (uint fid_phase_idx = 0; fid_phase_idx < NUM_FIDELITY_PHASES; fid_phase_idx++)
                {
                    // Do final pool of the transposed rows to generate a single pool datum
                    tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0>();
                }
            }
            // Do final pool of the transposed rows to generate a single pool datum
            tti_pool_instr_func<POOL_TYPE, p_gpool::CLR_SRCA_VLD, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0>();
        });

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods for reduce operations
 * @tparam REDUCE_DIM: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types, shows how many loops
 * to use full precision with of Source register datums with multiplies, values = [LoFi, HiFi2, HiFi3, HiFi4]
 */
template <ReduceDim REDUCE_DIM, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_addrmod_()
{
    constexpr bool high_fidelity     = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT = high_fidelity ? 1 : 0;

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ((REDUCE_DIM == ReduceDim::REDUCE_COL) ? 16 : 0)}, .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_0);

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = FIDELITY_INCREMENT}}.set(ADDR_MOD_2);

    if constexpr (REDUCE_DIM == ReduceDim::REDUCE_COL)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0, .clr = 1}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    }
    else if constexpr (REDUCE_DIM == ReduceDim::REDUCE_ROW)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = ELTWISE_MATH_ROWS},
            .dest = {.incr = ELTWISE_MATH_ROWS},
        }
            .set(ADDR_MOD_1);
    }
}

/**
 * @brief Sets up mop config for reduce operations
 * @tparam POOL_TYPE: Type of reduce pool op, values = [MAX, SUM, AVG]
 * @tparam REDUCE_DIM: Sets the reduce dimension, values = [REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR]
 * @tparam MATH_FIDELITY_TYPE: Only works for AVG/SUM pool types, shows how many loops
 * to use full precision with of Source register datums with multiplies, values = [LoFi, HiFi2, HiFi3, HiFi4]
 * @param tile_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIM, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_init_(const TileShape& tile_shape)
{
    _llk_math_reduce_addrmod_<REDUCE_DIM, MATH_FIDELITY_TYPE>();

    if constexpr (REDUCE_DIM == ReduceDim::REDUCE_COL)
    {
        _llk_math_reduce_col_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tile_shape);
    }
    else if constexpr (REDUCE_DIM == ReduceDim::REDUCE_ROW)
    {
        _llk_math_reduce_row_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tile_shape);
    }
    else if constexpr (REDUCE_DIM == ReduceDim::REDUCE_SCALAR)
    {
        _llk_math_reduce_scalar_mop_config_<POOL_TYPE, MATH_FIDELITY_TYPE>(tile_shape);
    }

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Perform a reduce operation
 * @param tile_idx: Tile index into the destination register.
 * If dest reg in float16 mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in full mode
 * If dest reg in float32 mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 */
inline void _llk_math_reduce_(const uint32_t tile_idx)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);
    // Run MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    // Since only 1 face of srcB is used for constant values,
    // can clear data valid after all operations are done
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}
