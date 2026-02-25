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
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32]
// rt_dim: 1
// ct_dim: {1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16}
// kt_dim: even number from 2 to 256 (inclusive)
// fidelity: LoFi only
// throttle: not supported

inline void _llk_unpack_AB_custom_mm_mop_config_(const std::uint32_t ct_dim) {
    const std::uint32_t replay_buf_prog_len = ct_dim % 2 == 1 ? 28 : 32;

    load_replay_buf(0, replay_buf_prog_len, [ct_dim] {
        // === Context 0 full ===
        // Wait for context available
        t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

        // Unpack SrcA (in1, full tile, uses contexts to manipulate L1 address and keeps SrcA address fixed at 0)
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);  // Also set dvalid

        // Unpack SrcB (in0, one instruction per face, uses counters to manipulate addresses for both L1 and SrcB)
        // Counters of interest are CH0 Y and Z (both with a stride of a single face in L1
        // (datum_size * 16 * face_r_dim) and CH1 Y with a stride of a face in SrcB (16 rows)
        // Each unpack instruction we move to next face in L1, but this is split among Y and Z counters
        // Because they are 8 bit counters and to cover max inner dim of 256
        // we need to increment face counter 512 times thus overflowing if we use a single counter,
        // using both counters allows us to cover exactly 512 needed increments
        // For SrcB first unpack has to land at index 0 and second one at index 16,
        // increment from first to second is straightforward, but moving back to 0 after second requires us to exploit
        // counter overflow as CH1 counters only use low 6 bits and wrap at the number of rows in a Src reg (64)
        // thus an increment of 48 rows or 3 CH1 Y increments wraps us back to 0
        // (which is conveniently the max we can increment in a single instruction)

        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);  // Also set dvalid

        // Signal context done
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        if (ct_dim % 2 == 1) {
            // === Context 1 full ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1, full tile, uses contexts to manipulate L1 address and keeps SrcA address fixed at 0)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);  // Also set dvalid

            // Unpack SrcB (in0, one instruction per face, uses counters to manipulate addresses for both L1 and SrcB)
            // Same counter shenanigans as described above
            TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
            TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);  // Also set dvalid

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);
        } else {
            // === Context 1 reuse ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1, full tile, uses contexts to manipulate L1 address and keeps SrcA address fixed at 0)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);  // Also set dvalid

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            // === Context 0 reuse ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1, full tile, uses contexts to manipulate L1 address)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);  // Also set dvalid

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            // === Context 1 reuse ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1, full tile, uses contexts to manipulate L1 address)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);  // Also set dvalid

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);
        }

        for (std::uint32_t i = 0; i < 3; i++) {
            // === Context 0 reuse ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1, full tile, uses contexts to manipulate L1 address)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);  // Also set dvalid

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            // === Context 1 reuse ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1, full tile, uses contexts to manipulate L1 address)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);  // Also set dvalid

            // Signal context done
            t6_semaphore_get(semaphore::UNPACK_SYNC);
        }
    });

    // Mop is configured to always cover two iterations of the inner dim loop, providing two benefits:
    // 1. With max mop iterations (128) we can cover up to 256 kt_dim (max supported by this API)
    // 2. No need for different templates for contexts 0 and 1 since they're always executed as a pair
    // To usefully issue up to 128 mop iterations we're limited to only using 0s in zmask
    // (not using SKIP_A/B instructions) since iterations beyond 32 always use 0s for zmask
    //
    // Replay buffer layout:
    // Odd ct_dim (1, 3, 5) uses 28 instructions:
    //   0-4: full ctx0, 5-9: full ctx1, 10-27: {reuse ctx0, reuse ctx1} x 3 pairs
    // Even ct_dim (2, 4, 6, 8, 10, 12, 14, 16) uses 32 instructions:
    //   0-4: full ctx0, 5-7: reuse ctx1, 8-31: {reuse ctx0, reuse ctx1} x 4 pairs
    //
    // Desired instruction sequences per mop iteration (covering 2 inner dim loops):
    // ct_dim == 1: full ctx0, full ctx1
    // ct_dim == 2: full ctx0, reuse ctx1, full ctx0, reuse ctx1
    // Odd ct_dim (3, 5): full ctx0, {reuse ctx1, reuse ctx0} x ((ct_dim-1)/2) times,
    //                    full ctx1, {reuse ctx0, reuse ctx1} x ((ct_dim-1)/2) times
    // Even ct_dim (4, 6, 8, 10, 12, 14, 16): full ctx0, reuse ctx1, {reuse ctx0, reuse ctx1} x ((ct_dim/2)-1) times,
    //                                        full ctx0, reuse ctx1, {reuse ctx0, reuse ctx1} x ((ct_dim/2)-1) times
    //
    // Mop template parameter assignment (breaking sequences into replay parts):
    // ct_dim == 1: A0=ctx0_full, B=ctx1_full (uses unpackB)
    // ct_dim == 2: A0=ct_dim_2,  B=ct_dim_2 (uses unpackB)
    //              ct_dim_2 captures: full ctx0, reuse ctx1
    // Odd ct_dim (3, 5): A0=ctx0_full, A1=ctx1_r_tail, A2=ctx1_full, A3=ctx0_r_tail (uses unpackHalo)
    //                    ctx1_r_tail captures: {reuse ctx1, reuse ctx0} x ((ct_dim-1)/2) times
    //                    ctx0_r_tail captures: {reuse ctx0, reuse ctx1} x ((ct_dim-1)/2) times
    // Even ct_dim: A0=first_half, A1=second_half, A2=first_half, A3=second_half (uses unpackHalo)
    //              first_half captures: full ctx0, reuse ctx1, {reuse ctx0, reuse ctx1} x ((ct_dim/2)-1) times
    //              second_half captures: {reuse ctx0, reuse ctx1} x (ct_dim/2) times

    const std::uint32_t ctx0_full = lltt::replay_insn(0, 5);
    const std::uint32_t ctx1_full = lltt::replay_insn(5, 5);
    const std::uint32_t ct_dim_2 = lltt::replay_insn(0, 8);
    const std::uint32_t ctx1_r_tail = lltt::replay_insn(13, (ct_dim - 1) * 3);
    const std::uint32_t ctx0_r_tail = lltt::replay_insn(10, (ct_dim - 1) * 3);
    const std::uint32_t first_half = lltt::replay_insn(0, 2 + (ct_dim / 2) * 3);
    const std::uint32_t second_half = lltt::replay_insn(8, (ct_dim / 2) * 3);
    const std::uint32_t even_A0 = ct_dim == 2 ? ct_dim_2 : first_half;

    ckernel_unpack_template tmp = ckernel_unpack_template(
        ct_dim <= 2,                                  // Use UNPACR_B and SKIP_B instructions?
        ct_dim > 2,                                   // Use UNPACR_A1/2/3 instructions?
        ct_dim % 2 == 1 ? ctx0_full : even_A0,        // A0
        ct_dim % 2 == 1 ? ctx1_r_tail : second_half,  // A1
        ct_dim % 2 == 1 ? ctx1_full : first_half,     // A2
        ct_dim % 2 == 1 ? ctx0_r_tail : second_half,  // A3
        0,                                            // Skip A
        ct_dim == 1 ? ctx1_full : ct_dim_2,           // B
        0                                             // Skip B
    );

    tmp.program();
    TTI_MOP_CFG(0);
}

template <bool transpose = false>
inline void _llk_unpack_AB_custom_mm_init_(const std::uint32_t unpB_face_r_dim, const std::uint32_t ct_dim = 1) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    // UnpA unpacks full tiles
    constexpr std::uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
    // UnpB unpacks [{1, 2, 4, 8}, 32] tiles which only have top two faces and only a single face per instruction
    // so a single instruction only unpacks unpB_face_r_dim rows
    const std::uint32_t unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1;
    TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

    _llk_unpack_AB_custom_mm_mop_config_(ct_dim);

    // Reset counters here since reset in the execute API is at the end
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
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

    const std::uint32_t block_increment = read_transposed ? kt_dim * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? tile_size_a : ct_dim * tile_size_a;

    std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    // Program SrcB address once, its updated using counters for up to 256 kt_dim
    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;

    // We can issue mop only once for up to 256 kt_dim
    TT_MOP(0, (kt_dim / 2) - 1, 0);

    if (ct_dim == 1) {
#pragma GCC unroll 8
        for (std::uint32_t k = 0; k < kt_dim; k += 2) {
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            address_a += inner_increment;
            semaphore_post(semaphore::UNPACK_SYNC);
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            address_a += inner_increment;
            semaphore_post(semaphore::UNPACK_SYNC);
        }
    } else if (ct_dim % 2 == 0) {
        for (std::uint32_t k = 0; k < kt_dim; k++) {
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
    } else {
        for (std::uint32_t k = 0; k < kt_dim; k += 2) {
            std::uint32_t block_start_address = address_a;
#pragma GCC unroll 8
            for (std::uint32_t ct = 0; ct + 1 < ct_dim; ct += 2) {
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = block_start_address;
                block_start_address += block_increment;
                semaphore_post(semaphore::UNPACK_SYNC);
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = block_start_address;
                block_start_address += block_increment;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = block_start_address;
            block_start_address += block_increment;
            semaphore_post(semaphore::UNPACK_SYNC);
            address_a += inner_increment;

            block_start_address = address_a;
#pragma GCC unroll 8
            for (std::uint32_t ct = 0; ct + 1 < ct_dim; ct += 2) {
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = block_start_address;
                block_start_address += block_increment;
                semaphore_post(semaphore::UNPACK_SYNC);
                wait_for_next_context(2);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = block_start_address;
                block_start_address += block_increment;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = block_start_address;
            block_start_address += block_increment;
            semaphore_post(semaphore::UNPACK_SYNC);
            address_a += inner_increment;
        }
    }

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    // Reset counters at the end
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}
