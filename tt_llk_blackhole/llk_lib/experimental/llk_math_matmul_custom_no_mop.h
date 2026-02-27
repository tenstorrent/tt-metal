// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

using namespace ckernel;

// Helper functions for math fidelity
constexpr int get_math_num_fidelity_phases(const MathFidelity math_fidelity)
{
    // LoFi = 0 has 0 fidelity phases
    // HiFi2 = 1 has 1 phase, HiFi3 = 2 has 2 phases, HiFi4 = 3 has 3 phases
    return ckernel::to_underlying(math_fidelity);
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL>
inline void matmul_configure_addrmod_no_mop(
    const bool transpose,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    static_assert(THROTTLE_LEVEL >= 0 && THROTTLE_LEVEL <= 5, "THROTTLE_LEVEL must be in range [0, 5]");
    constexpr bool high_fidelity     = math_fidelity != MathFidelity::LoFi;
    constexpr int fidelity_increment = high_fidelity ? 1 : 0;

    // MVMUL does D = B*A

    // Inner Loop --> 32/8 = 4 times for the full 32x16 face
    // DEST -- 8 rows are calculated each time
    // SRCB -- 8 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[8,16] = B[8,16] * A[16,16]
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // reset all, increment fidelity if we have more fidelity phases
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 1},
        .srcb     = {.incr = 0, .clr = 1, .cr = 1},
        .dest     = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = fidelity_increment, .clr = 0},
    }
        .set(ADDR_MOD_5);

    if constexpr (THROTTLE_LEVEL)
    {
        // reset all, including fidelity
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 1},
            .srcb     = {.incr = 0, .clr = 1, .cr = 1},
            .dest     = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_6);
    }

    if (transpose)
    {
        addr_mod_t {
            .srca = {.incr = 32, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    }
    else
    {
        addr_mod_t {
            //.srca = {.incr = srca_increment, .clr = 0, .cr = 0},
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    }

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 32, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);

    if (transpose)
    {
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 1},
            .srcb = {.incr = 48, .clr = 0, .cr = 1}, // cr=32 before, cr+48=16 after wrapping
            .dest = {.incr = 0, .clr = 0, .cr = 1},
            // .bias = {.incr = 1},
        }
            .set(ADDR_MOD_4);
    }
    else
    {
        addr_mod_t {
            .srca = {.incr = 32, .clr = 0, .cr = 1},
            //.srca = {.incr = srca_set, .clr = 0, .cr = 1},
            .srcb = {.incr = 48, .clr = 0, .cr = 1}, // cr=32 before, cr+48=16 after wrapping
            .dest = {.incr = 0, .clr = 0, .cr = 1},
            // .bias = {.incr = 1},
        }
            .set(ADDR_MOD_4);
    }
}

template <MathFidelity math_fidelity = MathFidelity::LoFi, int THROTTLE_LEVEL = 0>
inline void matmul_configure_addrmod_reinit(const bool transpose = false)
{
    // Reinit must restore the full matmul address-modifier contract used by replay.
    // In particular, transpose affects ADDR_MOD_1/4 and fidelity/throttle use ADDR_MOD_5/6.
    matmul_configure_addrmod_no_mop<math_fidelity, THROTTLE_LEVEL>(transpose);
}

template <MathFidelity math_fidelity>
inline void matmul_configure_mop_custom(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    // Col major layout in dest only impacs destination address increment
    // if col major layout faces are ordered as f0,f2,f1,f3

    constexpr int num_fidelity_phases = get_math_num_fidelity_phases(math_fidelity);
    constexpr bool high_fidelity      = math_fidelity != MathFidelity::LoFi;

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    const std::uint32_t replay_buf_len = 16;

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [high_fidelity, reuse_a]
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16/32, srcb=0, dest+=8  // srca+=32 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8  // A1 -> A2 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1 // srca=0,    srcb=32,  dest+=8  // A1 -> A2 if transposed

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=srca, srcb+=8,  dest+=8 // A1 -> A2 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8 // A2 -> A1 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3 // srca=32,   srcb=48,  dest+=8

            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2 // srca+=16,  srcb=0,   dest+=8 // A2 -> A1 if transposed
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3 // srca=srca, srcb+=8,  dest+=8

            if constexpr (high_fidelity)
            {
                // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5)
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5)
            }
            else
            {
                if (reuse_a)
                {
                    TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0); // B3A3 or B3A2 // reset srca/srcb/dest, increment phase (addr_mod_5), clear src A
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0); // B3A3 or B2A1 // reset srca/srcb/dest, increment phase (addr_mod_5), clear src A
                }
            }
        });

    // MOP template programming removed - will use direct replay calls
}

template <int Level>
void run_throttled_sequence_no_mop();

template <>
void run_throttled_sequence_no_mop<1>()
{
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
}

template <>
void run_throttled_sequence_no_mop<2>()
{
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
}

template <>
void run_throttled_sequence_no_mop<3>()
{
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
}

template <>
void run_throttled_sequence_no_mop<4>()
{
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_NOP;
}

template <>
void run_throttled_sequence_no_mop<5>()
{
    TTI_NOP;
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_NOP;
    TTI_NOP;
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_NOP;
    TTI_NOP;
}

/*
 * Programming of the MOP for the case we limit matmul compute throughput
 * Done by inserting NOP instructions between MVMUL instructions of matmul kernel
 *
 * Valid range of THROTTLE_LEVEL is {1,2,3,4,5}
 * Each value corresponds to level of throttling as:
 * Level 1: throttle to 73% of max
 * Level 2: throttle to 67% of max
 * Level 3: throttle to 50% of max
 * Level 4: throttle to 40% of max
 * Level 5: throttle to 33% of max
 */
template <MathFidelity math_fidelity, int THROTTLE_LEVEL>
inline void matmul_configure_mop_throttled_no_mop(
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    // Col major layout in dest only impacts destination address increment
    // if col major layout faces are ordered as f0,f2,f1,f3

    constexpr int num_fidelity_phases = get_math_num_fidelity_phases(math_fidelity);
    constexpr bool high_fidelity      = math_fidelity != MathFidelity::LoFi;
    static_assert((THROTTLE_LEVEL > 0) && (THROTTLE_LEVEL <= 5), "MM throttling only enabled for THROTTLE_LEVEL={1,2,3,4,5}");
    LLK_ASSERT(
        (in0_tile_r_dim == TILE_R_DIM) && (in0_tile_c_dim == TILE_C_DIM) && (in1_tile_r_dim == TILE_R_DIM) && (in1_tile_c_dim == TILE_C_DIM) && !partial_face,
        "MM throttling only enabled for full 32x32 tile size");

    const bool reuse_a = ct_dim >= rt_dim;

    constexpr std::uint32_t replay_buf_len = (THROTTLE_LEVEL > 3) ? (1 + THROTTLE_LEVEL * 2) : ((THROTTLE_LEVEL > 1) ? (3 + THROTTLE_LEVEL * 4) : 10);

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [] { run_throttled_sequence_no_mop<THROTTLE_LEVEL>(); });

    // MOP template programming removed - will use direct replay calls
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_init_no_mop_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t transpose      = 0,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1)
{
    matmul_configure_addrmod_no_mop<math_fidelity, THROTTLE_LEVEL>(transpose, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    if constexpr (THROTTLE_LEVEL > 0)
    {
        matmul_configure_mop_throttled_no_mop<math_fidelity, THROTTLE_LEVEL>(
            ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }
    else
    {
        matmul_configure_mop_custom<math_fidelity>(ct_dim, rt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void _llk_math_matmul_uninit_no_mop_()
{
    // No state to restore - all states are transient or default
}

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_no_mop_(
    std::uint32_t dst_index,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    const bool reuse_a                = ct_dim >= rt_dim;
    const std::uint32_t t_dim         = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim       = reuse_a ? ct_dim : rt_dim; // reuse-dim
    constexpr int num_fidelity_phases = get_math_num_fidelity_phases(math_fidelity);
    constexpr bool high_fidelity      = math_fidelity != MathFidelity::LoFi;

    // Compute replay buffer length based on tile dimensions (same logic as in matmul_configure_mop)
    std::uint32_t replay_buf_len;
    if constexpr (THROTTLE_LEVEL > 0)
    {
        replay_buf_len = (THROTTLE_LEVEL > 3) ? (1 + THROTTLE_LEVEL * 2) : ((THROTTLE_LEVEL > 1) ? (3 + THROTTLE_LEVEL * 4) : 10);
    }
    else
    {
        replay_buf_len = 16;
    }

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));

            if constexpr (THROTTLE_LEVEL > 0)
            {
                // Throttled execution
                if constexpr (THROTTLE_LEVEL > 3)
                {
                    // THROTTLE_LEVEL 4 or 5: outer_loops = 2
                    if constexpr (high_fidelity)
                    {
                        // outer loop for fidelity phases
                        for (std::uint32_t phase = 0; phase < num_fidelity_phases; phase++)
                        {
                            // inner loop (2 iterations for standard tiles)
                            for (std::uint32_t inner = 0; inner < 2; inner++)
                            {
                                lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
                                if (inner < 1)
                                {
                                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // inner loop continuation
                                }
                                else if (phase < num_fidelity_phases - 1)
                                {
                                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // last inner, not last outer
                                }
                                else
                                {
                                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0); // last inner, last outer
                                }
                            }
                        }
                        // Final clear
                        if (reuse_a)
                        {
                            TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                        }
                        else
                        {
                            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                        }
                    }
                    else
                    {
                        // Not high fidelity, outer_loops = 2
                        for (std::uint32_t inner = 0; inner < 2; inner++)
                        {
                            lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
                            if (inner < 1)
                            {
                                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                            }
                            else
                            {
                                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
                            }
                        }
                        for (std::uint32_t inner = 0; inner < 2; inner++)
                        {
                            lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
                            if (inner < 1)
                            {
                                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                            }
                            else
                            {
                                TTI_MVMUL(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);
                            }
                        }
                    }
                }
                else
                {
                    // THROTTLE_LEVEL 1, 2, or 3
                    if constexpr (high_fidelity)
                    {
                        // outer loop is num_fidelity_phases
                        for (std::uint32_t phase = 0; phase < num_fidelity_phases; phase++)
                        {
                            for (std::uint32_t inner = 0; inner < 2; inner++)
                            {
                                lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
                                if (inner < 1)
                                {
                                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // inner loop continuation
                                }
                                else if (phase < num_fidelity_phases - 1)
                                {
                                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0); // last inner, not last outer
                                }
                                else
                                {
                                    TTI_MVMUL(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, ADDR_MOD_6, 0); // last inner, last outer
                                }
                            }
                        }
                    }
                    else
                    {
                        // Not high fidelity, outer_loops = 1
                        for (std::uint32_t inner = 0; inner < 2; inner++)
                        {
                            lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
                            if (inner < 1)
                            {
                                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
                            }
                            else
                            {
                                TTI_MVMUL(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);
                            }
                        }
                    }
                }
            }
            else
            {
                // Non-throttled execution - use replay
                if constexpr (high_fidelity)
                {
                    // Replay num_fidelity_phases times
                    for (std::uint32_t phase = 0; phase < num_fidelity_phases; phase++)
                    {
                        lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
                    }
                    // Final clear after all fidelity phases
                    if (reuse_a)
                    {
                        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                    }
                    else
                    {
                        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                    }
                }
                else
                {
                    // Just replay once
                    lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
                }
            }

            // Clear srcB or srcA at end of reuse (once per u block row)
            if (rut == (rut_dim - 1))
            {
                if (reuse_a)
                {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                }
                else
                {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                }
            }
        }
    }
}
