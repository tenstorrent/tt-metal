// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

using namespace ckernel;
using namespace ckernel::unpacker;

// CUSTOM_MM
// Custom matmul that uses MOP to loop both srcA and srcB along inner dim. Output height
// and width should be single tile with tile shape [1, 32]. Further work will uplift the
// custom mm to support for tiles along the width.
//
// K-dimension optimized implementation with MOP looping over kt_dim
//
// Implementation assumptions:
// - ct_dim = 1, rt_dim = 1 (single output tile, optimized for K-dimension reduction)
// - MOP replay buffer unpacks both SrcA and SrcB, incrementing both addresses per iteration
// - kernel_broadcast_a = 0 (no broadcast)
// - kernel_broadcast_b = 0 (no broadcast)
inline void _llk_unpack_AB_custom_mm_mop_config_(const bool unpB_partial_face) {
    // in0/inA - loaded to SrcB
    // in1/inB - loaded to SrcA

    // MOP replay buffer now updates both SrcA and SrcB addresses for K-dimension loop
    const std::uint32_t replay_buf_run_len = unpB_partial_face ? 5 : 4;
    const std::uint32_t replay_buf_prog_len = replay_buf_run_len * 2;

    load_replay_buf(
        0,
        replay_buf_prog_len,
        // Lambda function to set up replay buffer
        [unpB_partial_face] {
            // === Context 0 ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1/inB)
            TTI_UNPACR(
                SrcA,
                0b00000000,
                0,
                0,
                0,
                1 /*Set OvrdThreadId*/,
                1 /*Set Dvalid*/,
                p_unpacr::RAREFYB_DISABLE,
                0,
                0 /* Set ContextIdInc */,
                0,
                0,
                1);

            // Unpack SrcB (in0/inA)
            if (unpB_partial_face) {
                TTI_UNPACR(
                    SrcB,
                    0b00010001,
                    0,
                    0,
                    0,
                    1 /*Set OvrdThreadId*/,
                    0 /*Set Dvalid*/,
                    p_unpacr::RAREFYB_DISABLE,
                    0,
                    0 /* Set ContextIdInc */,
                    0,
                    0,
                    1);
                TTI_UNPACR(
                    SrcB,
                    0b00110001,
                    0,
                    0,
                    0,
                    1 /*Set OvrdThreadId*/,
                    1 /*Set Dvalid*/,
                    p_unpacr::RAREFYB_DISABLE,
                    0,
                    0 /* Set ContextIdInc */,
                    0,
                    0,
                    1);
            } else {
                TTI_UNPACR(
                    SrcB,
                    0b00001010,
                    0,
                    0,
                    0,
                    1 /*Set OvrdThreadId*/,
                    1 /*Set Dvalid*/,
                    p_unpacr::RAREFYB_DISABLE,
                    0,
                    0 /* Set ContextIdInc */,
                    0,
                    0,
                    1);
            }

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            // === Context 1 ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1/inB)
            TTI_UNPACR(
                SrcA,
                0b00000000,
                0,
                1,
                0,
                1 /*Set OvrdThreadId*/,
                1 /*Set Dvalid*/,
                p_unpacr::RAREFYB_DISABLE,
                0,
                0 /* Set ContextIdInc */,
                0,
                0,
                1);

            // Unpack SrcB (in0/inA)
            if (unpB_partial_face) {
                TTI_UNPACR(
                    SrcB,
                    0b00010001,
                    0,
                    0,
                    0,
                    1 /*Set OvrdThreadId*/,
                    0 /*Set Dvalid*/,
                    p_unpacr::RAREFYB_DISABLE,
                    0,
                    0 /* Set ContextIdInc */,
                    0,
                    0,
                    1);
                TTI_UNPACR(
                    SrcB,
                    0b00110001,
                    0,
                    0,
                    0,
                    1 /*Set OvrdThreadId*/,
                    1 /*Set Dvalid*/,
                    p_unpacr::RAREFYB_DISABLE,
                    0,
                    0 /* Set ContextIdInc */,
                    0,
                    0,
                    1);
            } else {
                TTI_UNPACR(
                    SrcB,
                    0b00001010,
                    0,
                    0,
                    0,
                    1 /*Set OvrdThreadId*/,
                    1 /*Set Dvalid*/,
                    p_unpacr::RAREFYB_DISABLE,
                    0,
                    0 /* Set ContextIdInc */,
                    0,
                    0,
                    1);
            }

            t6_semaphore_get(semaphore::UNPACK_SYNC);
        });

    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,                                     // src B
        false,                                     // halo - just used for 4 unpacks
        lltt::replay_insn(0, replay_buf_run_len),  // runs when context is 0
        0,
        0,
        0,
        lltt::replay_insn(replay_buf_run_len, replay_buf_run_len),  // runs when context is 1
        0,
        0);

    tmp.program();
}

template <bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void _llk_unpack_AB_custom_mm_hw_configure_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t unpA_num_faces = 4,
    const std::uint32_t unpB_num_faces = 4,
    const std::uint32_t unpA_tile_size = 0,
    const std::uint32_t unpB_tile_size = 0) {
    constexpr bool is_row_pool = false;
    constexpr bool stoch_rnd_en = (stoch_rnd_mode == StochRndType::All);
    constexpr bool fpu_srnd_en = stoch_rnd_en || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr bool pack_srnd_en = stoch_rnd_en || (stoch_rnd_mode == StochRndType::Pack);

    configure_unpack_AB<is_fp32_dest_acc_en, is_row_pool, fpu_srnd_en, pack_srnd_en>(
        unpA_src_format,
        unpB_src_format,
        unpA_dst_format,
        unpB_dst_format,
        unpA_face_r_dim,
        unpB_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces,
        unpB_num_faces);

    // Configure tile size in datums
    const uint32_t unpA_x_end = unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
    const uint32_t unpB_x_end = unpB_num_faces * unpB_face_r_dim * FACE_C_DIM - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

    regfile[p_gpr_unpack::TILE_SIZE_A] = unpA_tile_size;
    regfile[p_gpr_unpack::TILE_SIZE_B] = unpB_tile_size;
    sync_regfile_write(p_gpr_unpack::TILE_SIZE_B);
}

// K-dimension optimized initialization:
// - transpose = 0 (no transpose)
// - ct_dim = 1 (column tile dimension is 1, single output tile width)
// - rt_dim = 1 (row tile dimension is 1, single output tile height)
// - unpA_partial_face = false (always use full tile unpacking for input A)
__attribute__((always_inline)) inline void _llk_unpack_AB_custom_mm_init_(
    const std::uint32_t kt_dim = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces = 4,
    const std::uint32_t unpB_num_faces = 4,
    const bool unpB_partial_face = false) {
    // also turn on within_face_16x16_transpose if it was turned off by datacopy at runtime
    // on WH, the unpacker performs both transpose of faces as well as transpose each face.
    // the former is configured in mop, the latter is configured in cfg register in hw_configure
    // in large matmul, datacopy will disable the transpose of faces, so we need it turn it back on for matmul.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const uint32_t unpA_x_end = unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);

    if (unpB_partial_face) {
        // Do face by face unpacking. Need to program correct face dim
        // to compute address of the next face
        config_unpacker_x_end<p_setadc::UNP_B>(unpB_face_r_dim);
    } else {
        // Do full tile unpacking. No need to program face dim
        // as address counter pointing to the face is not incremented
        const uint32_t unpB_x_end = unpB_num_faces * unpB_face_r_dim * FACE_C_DIM - 1;
        TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);
    }

    _llk_unpack_AB_custom_mm_mop_config_(unpB_partial_face);
}

// K-dimension optimized implementation:
// - ct_dim = 1 (column tile dimension is 1, single output tile width)
// - rt_dim = 1 (row tile dimension is 1, single output tile height)
// - MOP loops kt_dim times, unpacking both SrcA and SrcB with address increments
// - unpA_partial_face = false (always use full tile unpacking for input A)
inline void _llk_unpack_AB_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim = 1) {
    // In0/InA -> srcB (supports partial face)
    // In1/InB -> srcA

    volatile uint* cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    std::uint32_t offset_address_a = tile_size_a * tile_index_a;
    std::uint32_t offset_address_b = tile_size_b * tile_index_b;

    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t address_b = base_address_b + offset_address_b;

    // Need to reset counters and update SrcB base address for each superloop over 128 kt_dim
    // I guess its due to some counters being overflowed
    for (std::uint32_t i = 0; i < kt_dim; i += 128) {
        std::uint32_t superloop_kt_dim = kt_dim - i > 128 ? 128 : kt_dim - i;
        std::uint32_t num_loops = superloop_kt_dim / 16;
        std::uint32_t remaining_kt = superloop_kt_dim % 16;

        // Wait for all contexts to be free
        wait_for_next_context(1);
        reset_config_context();

        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
        // Configure SrcB base address, once per call as we use counters for SrcB
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_a + (i * tile_size_a);

        // Unpack 16 tiles per loop, run 16 mop iterations and risc does 16 unrolled address updates 8 cnxt0 + 8 cnxt1
        for (std::uint32_t j = 0; j < num_loops; j++) {
            TTI_MOP(0, 15, 0xAAAA);
#pragma GCC unroll 8
            for (std::uint32_t k = 0; k < 8; k++) {
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_b;
                address_b += tile_size_b;
                semaphore_post(semaphore::UNPACK_SYNC);
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_b;
                address_b += tile_size_b;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
        }

        // Do any remaining kt_dim % 16 iterations, run mop remaining_kt times and risc does similar address updates
        if (remaining_kt != 0) {
            TT_MOP(0, remaining_kt - 1, 0xAAAA);
            for (std::uint32_t j = 0; j < remaining_kt / 2; j++) {
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_b;
                address_b += tile_size_b;
                semaphore_post(semaphore::UNPACK_SYNC);
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_b;
                address_b += tile_size_b;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
            // Last address update if odd number of remaining kt_dim only hits context 0
            if ((remaining_kt % 2) != 0) {
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_b;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
        }
    }

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();
}
