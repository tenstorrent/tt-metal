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

// SDPA_CUSTOM_MM_REUSE_DEST_SRCB
// Custom matmul that reuses SrcB from dest and only unpacks SrcA.
// Output height and width should be single tile with tile shape [1, 32].
//
// Both nt_dim and kt_dim loops are collapsed into a single MOP call using
// CFGSHIFTMASK with two scratch registers:
//   SCRATCH_SEC0 = block_increment (advance to next tile within a k-row)
//   SCRATCH_SEC1 = inner_increment (jump from end of one k-row to start of next)
//
// The replay buffer is 30 instructions, containing 9 block_increment
// reuse blocks followed by 1 inner_increment reuse block (3 insns each).
// The MOP template selects sliding windows into this buffer to cover any
// nt_dim from 1 to 16. Each MOP iteration covers 2 k-rows (via B-mode or
// halo-mode), and kt_dim/2 MOP iterations cover the full inner dimension.
//
// Constraints:
// - ct_dim = 1, rt_dim = 1 (single output tile)
// - nt_dim: 1 to 16
// - kt_dim: even number from 2 to 256 (inclusive)
// - kernel_broadcast_a = 0, kernel_broadcast_b = 0
inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_mop_config_(const std::uint32_t nt_dim) {
    // Replay buffer layout (30 instructions):
    //   [0-26]:  9 reuse blocks with block_increment (3 insns each)
    //   [27-29]: 1 final reuse block with inner_increment (3 insns)
    //
    // Each reuse block: SrcA unpack + CFGSHIFTMASK + NOP (3 insns)
    // block_increment reuse: CFGSHIFTMASK selects SCRATCH_SEC0
    // inner_increment reuse (final): CFGSHIFTMASK selects SCRATCH_SEC1
    //
    // This supports first_half up to 9 tiles and second_half up to 9 tiles = 18 max.
    // Practical limit is 16 due to dst size.

    constexpr std::uint32_t REPLAY_BUF_LEN = 30;
    load_replay_buf(0, REPLAY_BUF_LEN, [] {
        for (std::uint32_t i = 0; i < 9; i++) {
            TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);
            TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
        }
        TTI_UNPACR_COMMON(SrcA, 0b00000000, 1);
        TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 1, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;
    });

    // Mop covers two k-rows per iteration, allowing up to 256 kt_dim with 128 MOP iterations.
    // zmask is always 0 (skip path never used), which is required for iterations beyond 32.
    //
    // Each k-row processes nt_dim tiles: (nt_dim-1) tiles with block_increment, then
    // 1 tile with inner_increment. The tiles are split between two halves for the template:
    //   first_half_iterations = ceil(nt_dim/2)
    //   second_half_iterations = floor(nt_dim/2)
    //
    // first_half window: starts at buffer beginning, covers first_half_iterations * 3 insns
    // second_half window: ends at buffer end (always includes inner_increment block),
    //   covers second_half_iterations * 3 insns
    //
    // nt_dim == 1: B-mode (A0 + B), both pointing to the inner_increment block
    // nt_dim >= 2: Halo-mode (A0, A1, A2, A3), alternating first_half/second_half

    const std::uint32_t first_half_iters = (nt_dim + 1) >> 1;
    const std::uint32_t second_half_iters = nt_dim >> 1;

    const std::uint32_t first_half = lltt::replay_insn(0, first_half_iters * 3);
    const std::uint32_t second_half = lltt::replay_insn(REPLAY_BUF_LEN - second_half_iters * 3, second_half_iters * 3);

    // For nt_dim == 1: only inner_increment block is needed, use B-mode
    // block_increment == inner_increment when nt_dim == 1 so either block works,
    // but we use the inner_increment block (last 3 insns) for correctness
    const std::uint32_t single_tile = lltt::replay_insn(REPLAY_BUF_LEN - 3, 3);

    ckernel_unpack_template tmp = ckernel_unpack_template(
        nt_dim == 1,                             // B-mode when single tile per k-row
        nt_dim != 1,                             // Halo-mode when multiple tiles per k-row
        nt_dim == 1 ? single_tile : first_half,  // A0
        second_half,                             // A1
        nt_dim == 1 ? single_tile : first_half,  // A2
        second_half,                             // A3
        0,                                       // Skip A (unused, zmask always 0)
        single_tile,                             // B (used in B-mode for nt_dim==1)
        0                                        // Skip B (unused)
    );

    tmp.program();
    TTI_MOP_CFG(0);
}

__attribute__((always_inline)) inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(
    const std::uint32_t nt_dim = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces = 4) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const uint32_t unpA_x_end = unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_mop_config_(nt_dim);

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

// Both nt_dim and kt_dim loops collapsed into a single MOP call.
// CFGSHIFTMASK auto-increments the SrcA L1 address:
//   SCRATCH_SEC0 = block_increment (tile_size_a, advances within a k-row)
//   SCRATCH_SEC1 = inner_increment (jumps from end of one k-row to start of next)
// - in1_k_stride: stride between K tiles in in1 (default 1 for contiguous, use out_w for row-major layout)
inline void _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
    const std::uint32_t base_address_a,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_size_a,
    const std::uint32_t kt_dim = 1,
    const std::uint32_t nt_dim = 1,
    const std::uint32_t in1_k_stride = 1) {
    volatile uint* cfg = get_cfg_pointer();

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t block_increment = tile_size_a;
    // inner_increment: after (nt_dim-1) block_increments we're at the last tile of the k-row.
    // We need to jump to the first tile of the next k-row.
    // After processing a full k-row, address has advanced by (nt_dim-1) * tile_size_a
    // via block_increments. The inner_increment must bring us to the next k-row:
    //   (nt_dim - 1) * tile_size_a + inner_increment = in1_k_stride * tile_size_a
    //   inner_increment = (in1_k_stride - nt_dim + 1) * tile_size_a
    const std::uint32_t inner_increment = (in1_k_stride - nt_dim + 1) * tile_size_a;

    wait_for_next_context(1);
    reset_config_context();

    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    cfg[SCRATCH_SEC0_val_ADDR32] = block_increment;
    cfg[SCRATCH_SEC1_val_ADDR32] = inner_increment;

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    TT_MOP(0, (kt_dim / 2) - 1, 0);

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}
