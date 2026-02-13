// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file llk_unpack_AB_reduce.h
 * @brief Unpacker kernel for reduction operations
 *
 * This file provides template functions for unpacking and configuring reduction operations
 * on TT-Blackhole. It supports various pooling types and reduction dimensions,
 * handling both standard and tiny tile configurations.
 */

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

/**
 * @brief Configures the unpacker MOP for reduction operations. Handles both tiny tiles (face_r_dim < 16) and standard tiles.
 *
 * @tparam pool_type The type of pooling operation (MAX, SUM, AVG)
 * @tparam reduce_dim The dimension along which to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 *
 * @param face_r_dim The number of rows per face (must be either 1, 2, 4, 8, or 16)
 * @param num_faces The number of faces to process (must be 1, 2, or 4)
 *
 * @note For tiny tiles (face_r_dim < 16), padding is applied to prevent incorrect outputs
 * @note For REDUCE_SCALAR operations, SrcA is cleared before unpacking because SrcA is clobbered in the Math kernel.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_mop_config_(const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16, "face_r_dim must be either 1, 2, 4, 8, or 16");

    // Data valid for clear instructions is set to 0 since the MATH kernel should not process this data.
    // pool_type == PoolType::MAX sets the clear value to neginf if the pool-type is MAX and 0 if the pool-type is AVG/SUM
    static constexpr std::uint32_t clear_pool_dep_srca =
        TT_OP_UNPACR_NOP(Srcs::SrcA, 0, 0, 0 /* dvalid */, 0, 0 /* Stall_Clr_Cntrl */, 0, pool_type == PoolType::MAX /* 0 or neginf */, p_unpacr_nop::CLR_SRC);
    static constexpr std::uint32_t clear_zero_srca =
        TT_OP_UNPACR_NOP(Srcs::SrcA, 0, 0, 0 /* dvalid */, 0, 0 /* Stall_Clr_Cntrl */, 0, p_unpacr_nop::CLR_SRC_0, p_unpacr_nop::CLR_SRC);

    constexpr std::uint32_t REPLAY_BUF_LEN = 2;

    load_replay_buf(
        0,
        REPLAY_BUF_LEN,
        []
        {
            // Configure unpacker instruction for Src{A,B}. These instructions always increment L1 by 1 face.
            TTI_UNPACR(Srcs::SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Unpack SrcA
            TTI_UNPACR(Srcs::SrcB, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); // Unpack SrcB
        });

    // MOP constants
    constexpr std::uint32_t outerloop = 1;
    const std::uint32_t innerloop     = num_faces;

    // Padding should only be done when using tiny tiles otherwise the entire face overwrites the data read in Math
    if (face_r_dim < FACE_R_DIM) // Using tiny faces
    {
        // Fill SrcA with pool-type dependent padding value for tiny tiles before unpacking a face
        ckernel_template tmp(outerloop, innerloop, clear_pool_dep_srca, lltt::replay_insn(0, REPLAY_BUF_LEN));
        tmp.program();
    }
    else // Using standard faces (face_r_dim = FACE_R_DIM)
    {
        if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR)
        {
            // For scalar reduction, clear SrcA to zero before unpacking. SrcA is clobbered in Math kernel.
            ckernel_template tmp(outerloop, innerloop, clear_zero_srca, lltt::replay_insn(0, REPLAY_BUF_LEN));
            tmp.program();
        }
        else
        {
            // For row/column reduction, no clearing needed
            ckernel_template tmp(outerloop, innerloop, lltt::replay_insn(0, REPLAY_BUF_LEN));
            tmp.program();
        }
    }
}

/**
 * @brief Initialize the unpacker for reduce operations
 *
 * Configures the unpacker hardware registers and MOP settings
 * for reduction operations. This includes:
 * - Setting up haloize mode for row reductions (transpose)
 * - Configuring unpacker X dimension endpoints
 * - Calling the MOP configuration routine
 *
 * @tparam pool_type The type of pooling operation (MAX, SUM, AVG)
 * @tparam reduce_dim The dimension along which to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 *
 * @param face_r_dim The number of rows per face (must be either 1, 2, 4, 8, or 16)
 * @param num_faces The number of faces to process (must be 1, 2, or 4)
 *
 * @note For REDUCE_ROW operations, the face is transposed using haloize mode
 * @note Unpacker 0 (SrcA) reads face_r_dim*FACE_R_DIM datums
 * @note Unpacker 1 (SrcB) reads one row (FACE_R_DIM datums)
 */
template <PoolType pool_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
inline void _llk_unpack_AB_reduce_init_(const std::uint32_t face_r_dim, const std::uint32_t num_faces)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16, "face_r_dim must be either 1, 2, 4, 8, or 16");

    if constexpr (enforce_fp32_accumulation)
    {
        // Set necessary config regs for MOVB2D hi16/lo16 to work
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    }

    // Enable transpose (haloize mode) if reducing along rows
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(reduce_dim == ReduceDim::REDUCE_ROW);

    // Sets up Unpacker 0 to read face_r_dim*16 datums into SrcA register
    config_unpacker_x_end<p_setadc::UNP_A>(face_r_dim);

    // Sets up Unpacker 1 to read one row (16 datums) into SrcB register
    config_unpacker_x_end<p_setadc::UNP_B>(1);

    // Configure unpack MOP
    _llk_unpack_AB_reduce_mop_config_<pool_type, reduce_dim>(face_r_dim, num_faces);
}

/**
 * @brief Execute the unpacker for reduction operations
 *
 * Performs the actual unpacking of data for reduction operations by:
 * 1. Resetting address counters
 * 2. Programming source A and B base addresses in hardware registers
 * 3. Synchronizing with Trisc using semaphores
 * 4. Running the configured MOP
 * 5. Switching unpacker configuration context
 *
 * @tparam pool_type The type of pooling operation (MAX, SUM, AVG)
 * @tparam reduce_dim The dimension along which to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 *
 * @param address_a Base address for source A data in L1 memory
 * @param address_b Base address for source B data in L1 memory
 *
 * @note This function manages dual-context switching for pipelined execution
 * @note Semaphores ensure proper synchronization between Trisc and unpacker
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    // Reset address counters for both unpackers
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    // Get pointer to configuration registers for current state ID
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure addresses
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Execute the configured MOP
    ckernel::ckernel_template::run();

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
