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
// ct_dim: any integer from 1 to 16
// kt_dim: even number from 2 to 256 (inclusive)
// fidelity: LoFi only
// throttle: not supported

inline void _llk_unpack_AB_custom_mm_iter_insns(const bool post1) {
    if (post1) {
        // This nop is not actually used, it just ensures both tunings are of the same length when recording
        // When playing back the replay it is omitted
        TTI_NOP;
    }

    // Unpack SrcA (in1, full tile, uses CFGSHIFTMASK to manipulate L1 addr and keeps SrcA addr fixed at 0)
    TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);  // Also set dvalid

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
    if (!post1) {
        // Unpack second face of SrcB before CFGSHIFTMASK in post0 tuning
        TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);  // Also set dvalid
    }

    // Increment UnpA L1 address
    TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
    if (post1) {
        // Unpack second face of SrcB after CFGSHIFTMASK in post1 tuning
        TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);  // Also set dvalid;
    } else {
        // This nop is required in post0 as CFGSHIFTMASK is a 2 cycle instruction
        // And first instruction in next iteration is unpack into SrcA which uses this value
        TTI_NOP;
    }
}

inline void _llk_unpack_AB_custom_mm_mop_config_(const std::uint32_t ct_dim, const bool post1) {
    load_replay_buf(0, 32, [post1] {
        // Full unpack (both SrcA and SrcB)
        _llk_unpack_AB_custom_mm_iter_insns(post1);

        // Loop 8 times to fill up the replay buffer
        for (std::uint32_t i = 0; i < 8; i++) {
            // Reuse unpack (unpacks only SrcA, SrcB is reused across width dim)
            TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);  // Also set dvalid
            TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
        }

        // Reuse unpack (unpacks only SrcA, SrcB is reused across width dim)
        TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);  // Also set dvalid
        // This last iteration uses inner_increment instead of block_increment
        TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 1, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;
    });

    // Mop is configured to always cover two iterations of the inner (kt) dim loop, allowing us to
    // cover up to 256 kt_dim (max supported by this API) with max mop iterations (128)
    // To usefully issue up to 128 mop iterations we're limited to only using 0s in zmask
    // (not using SKIP_A/B instructions) since iterations beyond 32 always use 0s for zmask
    //
    // Replay buffer layout (always 32 instructions):
    // post0: [0-4]: full unpack, [5-28]: 8 reuse blocks (3 insns each), [29-31]: final reuse w/ inner_increment
    // post1: [0]: NOP (skipped), [1-4]: full unpack, [5-28]: 8 reuses, [29-31]: final reuse w/ inner_increment
    // Where:
    //   - full unpack = SrcA + SrcB + address increment (new k-tile)
    //   - reuse = SrcA only + address increment (same k-tile, next c-tile)
    //   - final reuse = uses inner_increment to jump to next k-row instead of block_increment
    //
    // Instruction sequence per mop iteration (covering 2 kt_dim loops):
    // Each kt_dim iteration produces ct_dim output tiles, all sharing the same SrcB tile.
    // ct_dim == 1:
    //   Sequence: [full] x 2 times
    //   (block_increment == inner_increment, so no specific full_with_inner_inc needed)
    // ct_dim == 2:
    //   Sequence: [full, reuse_with_inner_inc] x 2 times
    //   (first tile unpacks SrcB, second tile reuses SrcB and uses inner_increment)
    // ct_dim >= 3:
    //   Sequence: [full, reuse x (ct_dim-2) times, reuse_with_inner_inc] x 2 times
    //   Where middle reuses use block_increment, last reuse uses inner_increment
    //   Examples:
    //     ct_dim=3: [full, reuse, reuse_with_inner_inc] x 2 times
    //     ct_dim=4: [full, reuse x 2 times, reuse_with_inner_inc] x 2 times
    //     ct_dim=5: [full, reuse x 3 times, reuse_with_inner_inc] x 2 times
    //
    // Mop template parameter assignment (selecting replay buffer instruction ranges):
    // First, tiles per kt_dim iteration are divided between two halves:
    //   first_half_iterations = ceil(ct_dim/2)   // Rounds up for odd ct_dim
    //   second_half_iterations = floor(ct_dim/2)  // Rounds down for odd ct_dim
    //
    // Division examples:
    //   ct_dim=1: first_half=1 tile, second_half=0 tiles
    //   ct_dim=2: first_half=1 tile, second_half=1 tile
    //   ct_dim=3: first_half=2 tiles, second_half=1 tile
    //   ct_dim=4: first_half=2 tiles, second_half=2 tiles
    //   ct_dim=5: first_half=3 tiles, second_half=2 tiles
    //
    // These iteration counts are converted to replay buffer instruction ranges:
    //   first_half: starts at buffer beginning (offset by post1),
    //       captures full unpack + (first_half_iterations-1) reuses
    //               Length = (4 or 5 for full) + (first_half_iterations-1) * 3
    //   second_half: starts from buffer end, counts backwards second_half_iterations * 3 instructions
    //                Always includes the final reuse with inner_increment
    //
    // For ct_dim == 1:
    //   second_half is empty (0 instructions), so we use unpackB mode (2 template slots per mop iteration)
    //   Template: A0=first_half (just full unpack), B=first_half
    //   Sequence: A0, B → [full] x 2 times
    //
    // For ct_dim >= 2:
    //   Both halves have instructions, so we use unpackHalo mode (4 template slots per mop iteration)
    //   Template: A0=first_half, A1=second_half, A2=first_half, A3=second_half
    //   Sequence: A0, A1, A2, A3 → first_half, second_half, first_half, second_half
    //   Where A0+A1 covers first kt_dim iteration (ct_dim tiles), A2+A3 covers second kt_dim iteration

    const std::uint32_t first_half_iterations = ct_dim + 1 >> 1;  // Round up for odd ct_dim
    const std::uint32_t second_half_iterations = ct_dim >> 1;     // Round down for odd ct_dim
    // Both halves can take up to 9 iterations, meaning max of 18 tiles in width with full odd dim support
    // Although 16 is still the practical limit due to size of dst
    const std::uint32_t first_half = lltt::replay_insn(post1 ? 1 : 0, (post1 ? 1 : 2) + (first_half_iterations) * 3);
    // Second half has to count backwards to always encompass final iteration which uses inner_increment
    const std::uint32_t second_half = lltt::replay_insn(32 - second_half_iterations * 3, second_half_iterations * 3);

    ckernel_unpack_template tmp = ckernel_unpack_template(
        ct_dim == 1,  // Use UNPACR_B and SKIP_B instructions?
        ct_dim != 1,  // Use UNPACR_A1/2/3 instructions?
        first_half,   // A0
        second_half,  // A1
        first_half,   // A2
        second_half,  // A3
        0,            // Skip A
        first_half,   // B
        0             // Skip B
    );

    tmp.program();
    TTI_MOP_CFG(0);
}

template <bool transpose = false>
inline void _llk_unpack_AB_custom_mm_init_(
    const std::uint32_t unpB_face_r_dim, const std::uint32_t unpA_dst_format, const std::uint32_t ct_dim = 1) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    // UnpA unpacks full tiles
    constexpr std::uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
    // UnpB unpacks [{1, 2, 4, 8}, 32] tiles which only have top two faces and only a single face per instruction
    // so a single instruction only unpacks unpB_face_r_dim rows
    const std::uint32_t unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1;
    TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

    // This is a profiling guided heuristic that slightly tunes the instruction sequence
    // More details under tt-metal#38518
    const bool post1 = unpA_dst_format == to_underlying(DataFormat::Bfp4_b);

    _llk_unpack_AB_custom_mm_mop_config_(ct_dim, post1);

    // Reset counters here since reset in the execute API is at the end
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

inline void _llk_unpack_AB_custom_mm_run_(
    volatile uint* cfg,
    std::uint32_t address_a,
    const std::uint32_t address_b,
    const std::uint32_t block_increment,
    const std::uint32_t inner_increment,
    const std::uint32_t kt_dim) {
    // Program SrcB address once, its updated using counters for up to 256 kt_dim
    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    // Program SrcA address once, its updated using CFGSHIFTMASK
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    // Setup CFGSHIFTMASK increments
    // block_increment moves to next tile in tile_row
    cfg[SCRATCH_SEC0_val_ADDR32] = block_increment;
    // inner_increment moves to next tile_row (next inner_dim)
    cfg[SCRATCH_SEC1_val_ADDR32] = inner_increment;

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // We can issue mop only once for up to 256 kt_dim
    TT_MOP(0, (kt_dim / 2) - 1, 0);

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    // Reset counters at the end
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
    const std::uint32_t inner_increment = read_transposed ? -(((ct_dim - 1) * kt_dim) - 1) * tile_size_a : tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, block_increment, inner_increment, kt_dim);
}
