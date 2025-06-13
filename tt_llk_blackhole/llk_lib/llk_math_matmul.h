// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"

#ifndef HF
#define HF 0
#endif

using namespace ckernel;

template <int MATH_FIDELITY_DESC, DstTileFaceLayout FaceLayout = DstTileFaceLayout::ColMajor, int THROTTLE_LEVEL>
inline void matmul_configure_addrmod(
    const bool transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    constexpr int NUM_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr bool high_fidelity      = (NUM_FIDELITY_PHASES > 0);
    constexpr int FIDELITY_INCREMENT  = high_fidelity ? get_math_fidelity_increment(MATH_FIDELITY_DESC) : 0;

    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);

    static_assert(FaceLayout == DstTileFaceLayout::RowMajor, "FaceLayout must be RowMajor");

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
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
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

    if ((is_in0_16x32 && (!is_in1_32x16)) || is_in0_32x16)
    {
        if (transpose)
        {
            addr_mod_t {
                .srca = {.incr = 32, .clr = 0, .cr = 0},
                .srcb = {.incr = 0, .clr = 0, .cr = 1}, // cr=16 before
                .dest = {.incr = 8, .clr = 0, .cr = 0},
            }
                .set(ADDR_MOD_1);
        }
        else
        {
            addr_mod_t {
                .srca = {.incr = 16, .clr = 0, .cr = 0},
                .srcb = {.incr = 0, .clr = 0, .cr = 1}, // cr=16 before
                .dest = {.incr = 8, .clr = 0, .cr = 0},
            }
                .set(ADDR_MOD_1);
        }
    }
    else
    {
        if (is_in1_32x16)
        {
            addr_mod_t {
                .srca = {.incr = 16, .clr = 0, .cr = 0},
                .srcb = {.incr = 8, .clr = 0, .cr = 0},
                .dest = {.incr = 0, .clr = 0, .cr = 1},
            }
                .set(ADDR_MOD_1);
        }
        else
        {
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
        }
    }

    if (is_in1_32x16)
    {
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 8, .clr = 0, .cr = 0}, .dest = {.incr = 0, .clr = 0, .cr = 1}, // cr=16
        }
            .set(ADDR_MOD_2);
    }
    else if (is_in0_16x32 || is_in0_32x16)
    {
        if (partial_face)
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 32, .clr = 0, .cr = 0}, .srcb = {.incr = 0, .clr = 0, .cr = 0}, .dest = {.incr = 16, .clr = 0, .cr = 0},
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_2);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 0, .clr = 0, .cr = 0}, .dest = {.incr = 16, .clr = 0, .cr = 0},
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_2);
            }
        }
        else
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 32, .clr = 0, .cr = 0},
                    .srcb = {.incr = 0, .clr = 0, .cr = 1},
                    .dest = {.incr = 8, .clr = 0, .cr = 0},
                }
                    .set(ADDR_MOD_2);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0},
                    .srcb = {.incr = 0, .clr = 0, .cr = 1},
                    .dest = {.incr = 8, .clr = 0, .cr = 0},
                }
                    .set(ADDR_MOD_2);
            }
        }
    }
    else
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1},
            .srcb = {.incr = 32, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_2);
    }

    if (is_in0_16x32)
    {
        if (partial_face)
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 1}, // srca=16
                    .srcb = {.incr = 16, .clr = 0, .cr = 0},
                    .dest = {.incr = 0, .clr = 1, .cr = 0},
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_4);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 16, .clr = 0, .cr = 0}, .dest = {.incr = 0, .clr = 1, .cr = 0},
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_4);
            }
        }
        else
        {
            if (transpose)
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 1}, // srca=16
                    .srcb = {.incr = 16, .clr = 0, .cr = 1},
                    .dest = {.incr = 0, .clr = 0, .cr = 1},
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_4);
            }
            else
            {
                addr_mod_t {
                    .srca = {.incr = 16, .clr = 0, .cr = 0}, .srcb = {.incr = 16, .clr = 0, .cr = 1}, .dest = {.incr = 0, .clr = 0, .cr = 1},
                    // .bias = {.incr = 1},
                }
                    .set(ADDR_MOD_4);
            }
        }
    }
    else if (is_in0_32x16)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1}, .srcb = {.incr = 16, .clr = 0, .cr = 1}, .dest = {.incr = 8, .clr = 0, .cr = 0},
            // .bias = {.incr = 1},
        }
            .set(ADDR_MOD_4);
    }
    else if (is_in1_32x16)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1}, .srcb = {.incr = 8, .clr = 0, .cr = 0}, .dest = {.incr = 16, .clr = 0, .cr = 1},
            // .bias = {.incr = 1},
        }
            .set(ADDR_MOD_4);
    }
    else
    {
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
}

template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout = DstTileFaceLayout::ColMajor>
inline void matmul_configure_mop(
    bool transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim,
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

    constexpr bool high_fidelity = NUM_FIDELITY_PHASES > 0;

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_16x32 = (in1_tile_r_dim <= FACE_R_DIM) && (in1_tile_c_dim > FACE_C_DIM);

    const std::uint32_t replay_buf_len =
        (is_in0_16x32 && is_in1_32x16) ? 4 : ((is_in0_16x32 || is_in1_32x16 || is_in0_32x16 || is_in1_16x32) ? (partial_face ? 4 : 8) : 16);

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [high_fidelity, reuse_a, partial_face, is_in1_32x16, is_in0_16x32, is_in0_32x16, is_in1_16x32, t_dim]
        {
            if (is_in1_32x16)
            {
                if (is_in0_16x32)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16,  srcb+=8,  dest=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A1 // srca=srca, srcb+=8,  dest=+8, bias=1
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16,  srcb+=8,  dest=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A1 // srca=srca, srcb+=8,  dest=+8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B1A1 // srca=0,    srcb+=8,  dest=16 (addr_mod_4), bias=0

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B2A0 // srca+=16,  srcb+=8,  dest=16
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A1 // srca=srca, srcb+=8,  dest+=8
                }
            }
            else if (is_in0_16x32 || is_in0_32x16)
            {
                if (partial_face)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A0 // srca+=16,  srcb=0,   dest=+16
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B0A1 // srca+=16,  srcb+=16,  dest=0 (addr_mod_4), bias=0
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A2 // srca+=16,  srcb=0,  dest=+16
                }
                else
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A0 // srca+=16,  srcb=0,   dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(
                        p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B0A1 // srca+=16/=0,  srcb=16,  dest=0/+=8 (addr_mod_4), bias=0 // srca=0 dest+=8 if in0_32x16

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8/24 // dest+=24 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
                }
            }
            else
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16/32, srcb=0, dest+=8  // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8  // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1 // srca=0,    srcb=32,  dest+=8  // A1 -> A2 if transposed

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=srca, srcb+=8,  dest+=8 // A1 -> A2 if transposed
                if (!is_in1_16x32)
                {
                    // TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B2A1 // srca=32/16,srcb=16,  dest=0 (addr_mod_4) // A1 -> A2 && srca=16 if transposed

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3 // srca=32,   srcb=48,  dest+=8

                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2 // srca+=16,  srcb=0,   dest+=8 // A2 -> A1 if transposed
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3 // srca=srca, srcb+=8,  dest+=8
                }
            }

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

    // TODO: can we commonize this?
    constexpr uint inner_loops = high_fidelity ? NUM_FIDELITY_PHASES : 1;
    ckernel_template tmp(1 /* outer loop */, inner_loops, lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len));

    if constexpr (high_fidelity)
    {
        if (reuse_a)
        {
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
        else
        {
            tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
    }
    tmp.program(instrn_buffer);
}

template <int Level>
void run_throttled_sequence();

template <>
void run_throttled_sequence<1>()
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
void run_throttled_sequence<2>()
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
void run_throttled_sequence<3>()
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
void run_throttled_sequence<4>()
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
void run_throttled_sequence<5>()
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
template <int NUM_FIDELITY_PHASES, DstTileFaceLayout FaceLayout = DstTileFaceLayout::ColMajor, int THROTTLE_LEVEL>
inline void matmul_configure_mop_throttled(
    bool transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim,
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

    constexpr bool high_fidelity = NUM_FIDELITY_PHASES > 0;
    static_assert((THROTTLE_LEVEL > 0) && (THROTTLE_LEVEL <= 5), "MM throttling only enabled for THROTTLE_LEVEL={1,2,3,4,5}");

    const bool reuse_a        = ct_dim >= rt_dim;
    const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;

    const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
    const bool is_in1_32x16 = (in1_tile_r_dim > FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);
    const bool is_in0_32x16 = (in0_tile_r_dim > FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
    const bool is_in1_16x32 = (in1_tile_r_dim <= FACE_R_DIM) && (in1_tile_c_dim > FACE_C_DIM);

    constexpr std::uint32_t replay_buff_len_throttle = (THROTTLE_LEVEL > 3) ? (1 + THROTTLE_LEVEL * 2) : ((THROTTLE_LEVEL > 1) ? (3 + THROTTLE_LEVEL * 4) : 10);
    const std::uint32_t replay_buf_len =
        (is_in0_16x32 && is_in1_32x16) ? 4
                                       : ((is_in0_16x32 || is_in1_32x16 || is_in0_32x16 || is_in1_16x32) ? (partial_face ? 4 : 8) : replay_buff_len_throttle);

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        // Lambda function to load reply buffer
        [is_in1_32x16, is_in1_16x32, is_in0_32x16, is_in0_16x32]
        {
            if (!is_in1_32x16 && !is_in1_16x32 && !is_in0_32x16 && !is_in0_16x32)
            {
                run_throttled_sequence<THROTTLE_LEVEL>();
            }
        });

    constexpr uint outer_loops            = (THROTTLE_LEVEL > 3) ? 2 : (high_fidelity ? NUM_FIDELITY_PHASES : 1);
    const uint inner_loops                = (!is_in1_16x32) ? 2 : 1;
    constexpr uint8_t addr_mod_inner_loop = (THROTTLE_LEVEL > 3) ? ADDR_MOD_2 : ADDR_MOD_4;
    ckernel_template tmp(
        outer_loops,
        inner_loops,
        lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len),
        TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, addr_mod_inner_loop, 0));

    if (!is_in1_32x16 && !is_in1_16x32 && !is_in0_32x16 && !is_in0_16x32)
    {
        if constexpr (high_fidelity && THROTTLE_LEVEL > 3)
        {
            tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0));
            tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0));
        }
        else if constexpr (high_fidelity)
        {
            tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0));
            tmp.set_last_outer_loop_instr(TT_OP_MVMUL(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, ADDR_MOD_6, 0));
        }
        else
        {
            if constexpr (THROTTLE_LEVEL > 3)
            {
                tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0));
            }
            tmp.set_last_outer_loop_instr(TT_OP_MVMUL(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, ADDR_MOD_5, 0));
        }
    }

    tmp.program(instrn_buffer);
}

template <int MATH_FIDELITY_DESC, DstTileFaceLayout FaceLayout = DstTileFaceLayout::ColMajor, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_init_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t transpose      = 0,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1,
    const std::uint32_t kt_dim         = 1)
{
    matmul_configure_addrmod<MATH_FIDELITY_DESC, FaceLayout, THROTTLE_LEVEL>(
        transpose, ct_dim, rt_dim, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    constexpr int MATH_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    if constexpr (THROTTLE_LEVEL > 0)
    {
        matmul_configure_mop_throttled<MATH_FIDELITY_PHASES, FaceLayout, THROTTLE_LEVEL>(
            transpose > 0, ct_dim, rt_dim, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }
    else
    {
        matmul_configure_mop<MATH_FIDELITY_PHASES, FaceLayout>(
            transpose > 0, ct_dim, rt_dim, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <int MATH_FIDELITY_DESC, DstTileFaceLayout FaceLayout = DstTileFaceLayout::ColMajor, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_(
    uint dst_index, const bool transpose = false, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1, const std::uint32_t kt_dim = 1)
{
    const bool reuse_a                = ct_dim >= rt_dim;
    const std::uint32_t t_dim         = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim       = reuse_a ? ct_dim : rt_dim; // reuse-dim
    constexpr int NUM_FIDELITY_PHASES = get_math_num_fidelity_phases(MATH_FIDELITY_DESC);
    constexpr bool high_fidelity      = NUM_FIDELITY_PHASES > 0;

    for (uint t = 0; t < t_dim; t++)
    {
        for (uint rut = 0; rut < rut_dim; rut++)
        {
            math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));

            if constexpr (THROTTLE_LEVEL > 3 && high_fidelity)
            {
                for (uint phase = 0; phase < NUM_FIDELITY_PHASES; phase++)
                {
                    ckernel_template::run(instrn_buffer);
                }
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
                ckernel_template::run(instrn_buffer);
            }

            // Clear srcB or srcA at end of re-use (once per u block row)
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
