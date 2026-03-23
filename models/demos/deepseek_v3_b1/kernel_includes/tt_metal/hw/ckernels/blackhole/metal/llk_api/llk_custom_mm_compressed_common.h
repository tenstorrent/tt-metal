// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_compressed.h"

namespace compressed {

/**
 * @brief Reconfig SrcA format for use inside custom_mm MOP loop.
 *
 * Uses direct cfg[] writes instead of cfg_reg_rmw_tensix to avoid
 * tensix instruction pipeline latency. Must be called AFTER
 * wait_for_next_context() when the unpacker is paused at semaphore.
 */
FORCE_INLINE void reconfig_custom_mm_srca(
    volatile uint* cfg, uint32_t fmt_idx, uint32_t reg0_base, uint32_t reg2_base) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
}

/** @brief Reconfig SrcA with pre-resolved DataFormat value (no lookup). */
FORCE_INLINE void reconfig_custom_mm_srca_raw(
    volatile uint* cfg, uint32_t src_format, uint32_t reg0_base, uint32_t reg2_base) {
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
}

/** @brief Reconfig SrcA input format only (REG0). REG2 stays unchanged. */
FORCE_INLINE void reconfig_custom_mm_srca_input_only(volatile uint* cfg, uint32_t src_format, uint32_t reg0_base) {
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
}

/**
 * @brief MOP config for compressed custom_mm.
 *
 * Context-switching MOP with semaphore sync. RISC-V feeds per-tile addresses
 * (and optionally format reconfigs) between semaphore sync points.
 *
 * The upstream custom_mm MOP was rewritten to use CFGSHIFTMASK (autonomous,
 * no semaphore sync), which is incompatible with the compressed RISC-V loop
 * that needs to feed per-tile addresses/formats. This function preserves the
 * pre-CFGSHIFTMASK semaphore-based MOP for the compressed path.
 *
 * @tparam use_barrier  When true, emit a CLR_SRC barrier (TTI_UNPACR_NOP)
 *   before every SrcA unpack. Required for the runtime path (IMPL 0) where
 *   format switching happens autonomously without per-tile RISC-V sync.
 *   Must be false for constexpr paths (IMPL 1/2) where RISC-V handles format
 *   reconfig at sync points — the barrier causes incorrect results at scale.
 *
 * Restrictions:
 *   use_barrier=false: ct_dim 1-16, any parity (full=5, reuse=3, buffer=28/32 insns)
 *   use_barrier=true:  ct_dim 1 or ct_dim even and ct_dim*4 <= 32 (full=6, reuse=4,
 *                      variable-length buffer). Practically ct_dim <= 8 for even,
 *                      ct_dim == 1 for odd. ct_dim 6 has a known loop-count issue
 *                      (ct_dim/4 undercounts); only ct_dim ∈ {1, 2, 4, 8} are tested.
 */
template <bool use_barrier = true, uint32_t ct_dim = 1>
inline void _llk_unpack_AB_custom_mm_compressed_mop_config_() {
    static_assert(ct_dim >= 1 && ct_dim <= 16, "ct_dim must be 1-16");
    static_assert(
        !use_barrier || ct_dim == 1 || ct_dim == 2 || ct_dim == 4 || ct_dim == 8,
        "use_barrier=true only supports ct_dim ∈ {1, 2, 4, 8}");
    if constexpr (!use_barrier) {
        // No-barrier path: identical to pre-CFGSHIFTMASK upstream MOP.
        // full=5, reuse=3, always 28 (odd ct_dim) or 32 (even) insns.
        // Supports ct_dim 1-16.
        UNPACK(({
            constexpr uint32_t full_len = 5;
            constexpr uint32_t reuse_len = 3;
            const std::uint32_t replay_buf_prog_len = ct_dim % 2 == 1 ? 28 : 32;

            load_replay_buf(0, replay_buf_prog_len, [] {
                t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);
                TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
                TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);
                t6_semaphore_get(semaphore::UNPACK_SYNC);

                if (ct_dim % 2 == 1) {
                    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                    TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);
                    TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
                    TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);
                    t6_semaphore_get(semaphore::UNPACK_SYNC);
                } else {
                    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                    TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);
                    t6_semaphore_get(semaphore::UNPACK_SYNC);

                    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                    TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);
                    t6_semaphore_get(semaphore::UNPACK_SYNC);

                    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                    TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);
                    t6_semaphore_get(semaphore::UNPACK_SYNC);
                }

                for (std::uint32_t i = 0; i < 3; i++) {
                    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                    TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);
                    t6_semaphore_get(semaphore::UNPACK_SYNC);

                    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                    TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);
                    t6_semaphore_get(semaphore::UNPACK_SYNC);
                }
            });

            const std::uint32_t ctx0_full = lltt::replay_insn(0, full_len);
            const std::uint32_t ctx1_full = lltt::replay_insn(full_len, full_len);
            const std::uint32_t ct_dim_2 = lltt::replay_insn(0, full_len + reuse_len);
            const std::uint32_t ctx1_r_tail = lltt::replay_insn(full_len + reuse_len + 1, (ct_dim - 1) * reuse_len);
            const std::uint32_t ctx0_r_tail = lltt::replay_insn(full_len + reuse_len - 2, (ct_dim - 1) * reuse_len);
            const std::uint32_t first_half = lltt::replay_insn(0, 2 + (ct_dim / 2) * reuse_len);
            const std::uint32_t second_half = lltt::replay_insn(full_len + reuse_len, (ct_dim / 2) * reuse_len);
            const std::uint32_t even_A0 = ct_dim == 2 ? ct_dim_2 : first_half;

            ckernel_unpack_template tmp = ckernel_unpack_template(
                ct_dim <= 2,
                ct_dim > 2,
                ct_dim % 2 == 1 ? ctx0_full : even_A0,
                ct_dim % 2 == 1 ? ctx1_r_tail : second_half,
                ct_dim % 2 == 1 ? ctx1_full : first_half,
                ct_dim % 2 == 1 ? ctx0_r_tail : second_half,
                0,
                ct_dim == 1 ? ctx1_full : ct_dim_2,
                0);
            tmp.program();
            TTI_MOP_CFG(0);
        }));
    } else {
        // Barrier path: CLR_SRC before every SrcA unpack.
        // full=6, reuse=4. Variable-length buffer. Supports ct_dim 1-10.
        UNPACK(({
            constexpr uint32_t full_len = 6;
            constexpr uint32_t reuse_len = 4;

            const uint32_t replay_buf_prog_len =
                ct_dim == 1 ? 2 * full_len : full_len + reuse_len + (ct_dim / 4) * 2 * reuse_len;

            load_replay_buf(0, replay_buf_prog_len, [] {
                t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
                TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);
                TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
                TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);
                t6_semaphore_get(semaphore::UNPACK_SYNC);

                if (ct_dim == 1) {
                    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                    TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
                    TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);
                    TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
                    TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);
                    t6_semaphore_get(semaphore::UNPACK_SYNC);
                } else {
                    t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                    TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
                    TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);
                    t6_semaphore_get(semaphore::UNPACK_SYNC);

                    for (uint32_t i = 0; i < ct_dim / 4; i++) {
                        t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
                        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);
                        t6_semaphore_get(semaphore::UNPACK_SYNC);

                        t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
                        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
                        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);
                        t6_semaphore_get(semaphore::UNPACK_SYNC);
                    }
                }
            });

            const uint32_t ctx0_full = lltt::replay_insn(0, full_len);
            const uint32_t ctx1_full = lltt::replay_insn(full_len, full_len);
            const uint32_t ct_dim_2 = lltt::replay_insn(0, full_len + reuse_len);
            const uint32_t first_half = lltt::replay_insn(0, full_len - reuse_len + (ct_dim / 2) * reuse_len);
            const uint32_t second_half = lltt::replay_insn(full_len + reuse_len, (ct_dim / 2) * reuse_len);
            const uint32_t even_A0 = ct_dim == 2 ? ct_dim_2 : first_half;

            ckernel_unpack_template tmp = ckernel_unpack_template(
                ct_dim <= 2,
                ct_dim > 2,
                ct_dim == 1 ? ctx0_full : even_A0,
                second_half,
                first_half,
                second_half,
                0,
                ct_dim == 1 ? ctx1_full : ct_dim_2,
                0);
            tmp.program();
            TTI_MOP_CFG(0);
        }));
    }
}

/**
 * @brief Full UNPACK init for compressed custom_mm.
 *
 * Mirrors _llk_unpack_AB_custom_mm_init_ but uses our compressed MOP config.
 */
template <bool use_barrier = true, uint32_t ct_dim = 1>
inline void _llk_unpack_AB_custom_mm_compressed_init_(const uint32_t unpB_face_r_dim) {
    UNPACK(({
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);  // no transpose

        constexpr uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
        const uint32_t unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1;
        TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
        TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

        _llk_unpack_AB_custom_mm_compressed_mop_config_<use_barrier, ct_dim>();

        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
    }));
}

/**
 * @brief API-level UNPACK init for compressed custom_mm.
 *
 * Resolves operand face_r_dim from CB, then calls the low-level init.
 * operand0 = in0 (activations, goes to SrcB), operand1 = in1 (weights, goes to SrcA).
 */
template <bool use_barrier = true, uint32_t ct_dim = 1>
inline void llk_unpack_AB_custom_mm_compressed_init(const uint32_t operand0, const uint32_t operand1) {
    UNPACK(({
        const uint32_t operandB_id = get_operand_id(operand0);
        const uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);
        _llk_unpack_AB_custom_mm_compressed_init_<use_barrier, ct_dim>(operandB_face_r_dim);
    }));
}

/**
 * @brief Full init for compressed custom_mm.
 *
 * Replaces custom_mm_block_init_short for compressed matmul. Programs a
 * semaphore-based MOP that the compressed RISC-V loop can sync with.
 *
 * @tparam use_barrier  false for constexpr paths (IMPL 1/2), true for runtime (IMPL 0).
 *   - false: supports ct_dim 1-16 (matches pre-CFGSHIFTMASK upstream MOP).
 *   - true:  supports ct_dim ∈ {1, 2, 4, 8} (barrier adds +1 insn per sequence,
 *            limiting replay buffer capacity). ct_dim 6/10/14 have a known
 *            loop-count bug; ct_dim >= 12 exceeds the 32-insn buffer limit.
 * @tparam ct_dim  Output width in tiles. Compile-time for static_assert validation.
 */
template <bool use_barrier = true, uint32_t ct_dim = 1, bool split_acc = false, bool dense_packing = false>
inline void custom_mm_compressed_block_init_short(
    const uint32_t in0_cb_id, const uint32_t in1_cb_id, const uint32_t out_cb_id) {
    llk_unpack_AB_custom_mm_compressed_init<use_barrier, ct_dim>(in0_cb_id, in1_cb_id);
    MATH((llk_math_custom_mm_init<false, split_acc, dense_packing>(in0_cb_id, in1_cb_id, ct_dim)));
    PACK((llk_pack_init<false, false>(out_cb_id)));
    if constexpr (dense_packing) {
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

}  // namespace compressed
