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

inline void _llk_unpack_AB_custom_mm_mop_config_(const std::uint32_t ct_dim) {
    constexpr std::uint32_t replay_buf_full_len = 5;
    constexpr std::uint32_t replay_buf_reuse_len = 3;
    constexpr std::uint32_t replay_buf_ctx0_full = 0;
    constexpr std::uint32_t replay_buf_ctx1_full = replay_buf_ctx0_full + replay_buf_full_len;
    constexpr std::uint32_t replay_buf_ctx0_reuse = replay_buf_ctx1_full + replay_buf_full_len;
    constexpr std::uint32_t replay_buf_ctx1_reuse = replay_buf_ctx0_reuse + replay_buf_reuse_len;
    constexpr std::uint32_t replay_buf_prog_len = replay_buf_ctx1_reuse + replay_buf_reuse_len;

    load_replay_buf(0, replay_buf_prog_len, [] {
        // === Context 0 full ===
        // Wait for context available
        t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

        // Unpack SrcA (in1/inB)
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);

        // Unpack SrcB (in0/inA)
        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110001, 1);

        // Signal context done
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // === Context 1 full ===
        // Wait for context available
        t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

        // Unpack SrcA (in1/inB)
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);

        // Unpack SrcB (in0/inA)
        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110001, 1);

        // Signal context done
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // === Context 0 reuse ===
        // Wait for context available
        t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

        // Unpack SrcA (in1/inB)
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);

        // Signal context done
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // === Context 1 reuse ===
        // Wait for context available
        t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

        // Unpack SrcA (in1/inB)
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);

        // Signal context done
        t6_semaphore_get(semaphore::UNPACK_SYNC);
    });

    constexpr uint32_t ctx0_full = lltt::replay_insn(replay_buf_ctx0_full, replay_buf_full_len);
    constexpr uint32_t ctx1_full = lltt::replay_insn(replay_buf_ctx1_full, replay_buf_full_len);
    constexpr uint32_t ctx0_reuse = lltt::replay_insn(replay_buf_ctx0_reuse, replay_buf_reuse_len);
    constexpr uint32_t ctx1_reuse = lltt::replay_insn(replay_buf_ctx1_reuse, replay_buf_reuse_len);
    constexpr uint32_t ctx0_split1 = lltt::replay_insn(replay_buf_ctx0_full, replay_buf_full_len - 2);
    constexpr uint32_t ctx0_split2 = lltt::replay_insn(replay_buf_ctx0_full + 2, 2);

    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,                                    // Use UNPACR_B and SKIP_B instructions
        false,                                   // Dont use UNPACR_A1/2/3 instructions
        ctx0_full,                               // A0
        0,                                       // A1 (not used)
        0,                                       // A2 (not used)
        0,                                       // A3 (not used)
        ct_dim == 1 ? ctx0_split1 : ctx0_reuse,  // Skip A
        ct_dim == 1 ? ctx1_full : ctx1_reuse,    // B
        ct_dim == 1 ? ctx0_split2 : ctx1_reuse   // Skip B
    );

    tmp.program();
    TT_MOP_CFG(0);
}

template <bool transpose = false>
inline void _llk_unpack_AB_custom_mm_init_(
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM, const std::uint32_t ct_dim = 1) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    constexpr std::uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
    const std::uint32_t unpB_x_end = (TILE_NUM_FACES / 2) * unpB_face_r_dim * FACE_C_DIM - 1;
    TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

    _llk_unpack_AB_custom_mm_mop_config_(ct_dim);
}

template <bool read_transposed = false>
inline void _llk_unpack_AB_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim = 1,
    const std::uint32_t ct_dim = 1) {
    volatile uint* cfg = get_cfg_pointer();

    std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Need update SrcB base address for each superloop over 128 kt_dim
    // I guess its due to some counters being overflowed
    for (std::uint32_t i = 0; i < kt_dim; i += 128) {
        std::uint32_t superloop_kt_dim = kt_dim - i > 128 ? 128 : kt_dim - i;

        // Wait for all contexts to be free
        wait_for_next_context(1);
        reset_config_context();

        // Configure SrcB base address, once per superblock as we use counters for SrcB
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b + (i * tile_size_b);

        TT_MOP(0, (superloop_kt_dim / 2) - 1, 0);

#pragma GCC unroll 8
        for (std::uint32_t k = 0; k < (superloop_kt_dim / 2); k++) {
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            address_a += tile_size_a;
            semaphore_post(semaphore::UNPACK_SYNC);
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            address_a += tile_size_a;
            semaphore_post(semaphore::UNPACK_SYNC);
        }

        if ((superloop_kt_dim % 2) != 0) {
            TTI_MOP(0, 0, 1);
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            semaphore_post(semaphore::UNPACK_SYNC);
        }
    }

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();
}
