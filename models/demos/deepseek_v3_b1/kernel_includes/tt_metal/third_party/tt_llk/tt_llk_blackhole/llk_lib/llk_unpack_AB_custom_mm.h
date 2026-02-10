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
// ct_dim: {1, 2, 4, 6, 8, 10, 11, 12, 14, 16}
// kt_dim: even number from 2 to 256 (inclusive)
// fidelity: LoFi only
// throttle: not supported

inline void _llk_unpack_AB_custom_mm_mop_config_(const std::uint32_t ct_dim) {
    const std::uint32_t replay_buf_prog_len = ct_dim == 1 ? 10 : 32;

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
        // Because they are 8 bit counters and to to cover max inner dim of 256
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

        if (ct_dim == 1) {
            // === Context 1 full ===
            // Wait for context available
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);

            // Unpack SrcA (in1, full tile, uses contexts to manipulate L1 address and keeps SrcA address fixed at 0)
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);  // Also set dvalid

            // Unpack SrcB (in0, one instruction per face, uses counters to manipulate addresses for both L1 and SrcB)
            // Same counter shennanigans as descibed above
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

            for (std::uint32_t i = 0; i < 4; i++) {
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
        }
    });

    // When ct_dim == 1 we alternate between full sequences for both contexts kt_dim times in total
    // When ct_dim > 1 we start with full context 0 and then alternate between reuse contexts 1 and 0 ct_dim - 1 times
    // Since if ct_dim > 1 it must be even full seqence always lands on context 0
    // thus instruction sequence is same for each kt_dim
    // Mop is configured such that it always covers two iterations of the inner dim loop,
    // this has two benefits:
    // With the max number of mop iterations (128) we can cover up to 256 kt_dim, which is the max supported by this API
    // There is no need to have different templates for contexts 0 and 1 since they are always executed as a pair
    // In order to be able to usefully issue up to 128 mop iterations we are limited to only using 0s in zmask
    // (not using SKIP_A/B instructions) since iterations beyond 32 always use 0s for zmask
    // In ct_dim == 1 case we enable unpackB and two instructions we issue per iteration are replays for
    // full context 0 and full context 1
    // In ct_dim > 1 case we enable unpackHalo and thus have 4 instructions to execute two inner dim iterations,
    // meaning 2 instructions per inner dim
    // This is achieved with really long replay sequences that each cover half of the ct_dim iterations
    // First half is:
    // full context 0, {reuse context 1, reuse context 0} (ct_dim / 4) - 1 times, [reuse context 1] if ct_dim > 2
    // Since our replay is organized like:
    // full context 0, reuse context 1, {reuse context 0, reuse context 1} 4 times (which fills all 32 instructions)
    // We just pick first ct_dim / 2 sequences (each sequence is 3 instructions plus first full one is
    // 2 additional instructions, thus getting the equation which calculates frist half of the replay length)
    // Similarly second half is just {reuse context 0, reuse context 1} ct_dim / 4 times,
    // which is just picking ct_dim / 2 seqeunces starting from the first reuse context 0 sequence
    // (each sequence is 3 instructions so replay length calculation is simpler)
    // To be sure that everything fits, for max ct_dim of 16, second half has to issue 4 pairs of sequnces
    // which is exactly what we recorded at the end
    // First sequence already has the first pair as full context 0 and reuse context 1
    // so it needs only 3 additional pair which are covered by the 4 pairs required for the second half
    // Special case is ct_dim == 2 where first half is just full context 0
    // and second half is just reuse context 1 thus having a earlier staring index

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
    } else {
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
    }

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    // Reset counters at the end
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}
