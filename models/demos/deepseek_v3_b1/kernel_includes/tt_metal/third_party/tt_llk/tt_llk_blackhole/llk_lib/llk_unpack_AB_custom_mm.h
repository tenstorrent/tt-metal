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
    const std::uint32_t replay_buf_prog_len = ct_dim == 1 ? 10 : 32;

    load_replay_buf(0, replay_buf_prog_len, [ct_dim] {
        // === Context 0 full ===
        // Wait for context available
        t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

        // Unpack SrcA (in1/inB)
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);

        // Unpack SrcB (in0/inA)
        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);

        // Signal context done
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        if (ct_dim == 1) {
            // === Context 1 full ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1/inB)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);

            // Unpack SrcB (in0/inA)
            TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
            TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);
        } else {
            // === Context 1 reuse ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1/inB)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            for (std::uint32_t i = 0; i < 4; i++) {
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
            }
        }
    });

    const std::uint32_t ctx0_full = lltt::replay_insn(0, 5);
    const std::uint32_t ctx1_full = lltt::replay_insn(5, 5);
    const std::uint32_t first_half = lltt::replay_insn(0, 2 + (ct_dim / 2) * 3);
    const std::uint32_t second_half = lltt::replay_insn(ct_dim == 2 ? 5 : 8, (ct_dim / 2) * 3);

    ckernel_unpack_template tmp = ckernel_unpack_template(
        ct_dim == 1,                           // Use UNPACR_B and SKIP_B instructions?
        ct_dim != 1,                           // Use UNPACR_A1/2/3 instructions?
        ct_dim == 1 ? ctx0_full : first_half,  // A0
        ct_dim == 1 ? 0 : second_half,         // A1
        ct_dim == 1 ? 0 : first_half,          // A2
        ct_dim == 1 ? 0 : second_half,         // A3
        0,                                     // Skip A
        ct_dim == 1 ? ctx1_full : 0,           // B
        0                                      // Skip B
    );

    tmp.program();
    TTI_MOP_CFG(0);
}

template <bool transpose = false>
inline void _llk_unpack_AB_custom_mm_init_(const std::uint32_t unpB_face_r_dim, const std::uint32_t ct_dim = 1) {
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
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    volatile uint* cfg = get_cfg_pointer();

    const std::uint32_t full_superloops = kt_dim / 128;
    const std::uint32_t remaining_kt = kt_dim % 128;
    const std::uint32_t superloop_increment = 128 * (tile_size_b);
    const std::uint32_t block_increment = read_transposed ? kt_dim * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? tile_size_a : ct_dim * tile_size_a;

    std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    auto kc_loop = [&address_a, ct_dim, block_increment, inner_increment, cfg](const std::uint32_t max_k) {
#pragma GCC unroll 8
        for (std::uint32_t k = 0; k < max_k; k++) {
            std::uint32_t block_start_address = address_a;
#pragma GCC unroll 8
            for (std::uint32_t ct = 0; ct < ct_dim; ct += 2) {
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = block_start_address;
                block_start_address += block_increment;
                semaphore_post(semaphore::UNPACK_SYNC);
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = block_start_address;
                block_start_address += block_increment;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
            address_a += inner_increment;
        }
    };

    auto k_loop = [&address_a, inner_increment, cfg](const std::uint32_t max_k) {
#pragma GCC unroll 8
        for (std::uint32_t k = 0; k < max_k; k += 2) {
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            address_a += inner_increment;
            semaphore_post(semaphore::UNPACK_SYNC);
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            address_a += inner_increment;
            semaphore_post(semaphore::UNPACK_SYNC);
        }
    };

    wait_for_next_context(1);
    reset_config_context();

    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
    // We can issue mop only once for up to 256 kt_dim
    TT_MOP(0, (kt_dim / 2) - 1, 0);

    // Need update SrcB base address for each superloop over 128 kt_dim
    // I guess its due to some counters being overflowed
    for (std::uint32_t i = 0; i < full_superloops; i++) {
        if (ct_dim == 1) {
            k_loop(128);
        } else {
            kc_loop(128);
        }
    }

    if (remaining_kt != 0) {
        if (ct_dim == 1) {
            k_loop(remaining_kt);
        } else {
            kc_loop(remaining_kt);
        }
    }

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();
}
