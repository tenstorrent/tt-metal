// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
 * @brief MOP config for compressed custom_mm (ct_dim=1 only).
 *
 * Programs the replay buffer with bfp2 transition barriers before every
 * UNPACR SrcA. This makes format switching safe without TTI instructions
 * from RISC-V during MOP execution (which would deadlock at large K).
 *
 * Replay buffer layout (12 instructions):
 *   [0-5]:  ctx0 full — sem_wait, barrier, SrcA, SrcB×2, sem_get
 *   [6-11]: ctx1 full — sem_wait, barrier, SrcA, SrcB×2, sem_get
 *
 * MOP template: A0=ctx0_full, B=ctx1_full (same structure as standard
 * custom_mm ct_dim=1, just with barrier inserted before each SrcA).
 */
inline void _llk_unpack_AB_custom_mm_compressed_mop_config_() {
    UNPACK(({
        constexpr uint32_t full_len = 6;  // sem_wait + barrier + SrcA + SrcB×2 + sem_get

        load_replay_buf(0, 2 * full_len, [] {
            // === Context 0 full ===
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1 /*Stall_Clr_Cntrl*/, 0, 0, p_unpacr_nop::CLR_SRC);
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 0, 1);
            TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
            TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            // === Context 1 full ===
            t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(semaphore::UNPACK_SYNC);
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00000000, 1, 1);
            TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
            TTI_UNPACR_COMMON(SrcB, 0b00110100, 1);
            t6_semaphore_get(semaphore::UNPACK_SYNC);
        });

        const uint32_t ctx0_full = lltt::replay_insn(0, full_len);
        const uint32_t ctx1_full = lltt::replay_insn(full_len, full_len);

        ckernel_unpack_template tmp = ckernel_unpack_template(
            true,       // Use UNPACR_B and SKIP_B
            false,      // No UNPACR_A1/2/3
            ctx0_full,  // A0
            0,          // A1
            0,          // A2
            0,          // A3
            0,          // Skip A
            ctx1_full,  // B
            0           // Skip B
        );

        tmp.program();
        TTI_MOP_CFG(0);
    }));
}

/**
 * @brief Full UNPACK init for compressed custom_mm (ct_dim=1 only).
 *
 * Mirrors _llk_unpack_AB_custom_mm_init_ but uses our compressed MOP config
 * with bfp2 barriers. Replaces the standard custom_mm UNPACK init entirely.
 */
inline void _llk_unpack_AB_custom_mm_compressed_init_(const uint32_t unpB_face_r_dim) {
    UNPACK(({
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);  // no transpose

        constexpr uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
        const uint32_t unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1;
        TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
        TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

        _llk_unpack_AB_custom_mm_compressed_mop_config_();

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
inline void llk_unpack_AB_custom_mm_compressed_init(const uint32_t operand0, const uint32_t operand1) {
    UNPACK(({
        const uint32_t operandB_id = get_operand_id(operand0);
        const uint32_t operandB_face_r_dim = get_operand_face_r_dim(operandB_id);
        _llk_unpack_AB_custom_mm_compressed_init_(operandB_face_r_dim);
    }));
}

/**
 * @brief Full init for compressed custom_mm (ct_dim=1 only).
 *
 * Replaces custom_mm_block_init_short for compressed matmul.
 * Sets up UNPACK with bfp2 barriers in MOP, plus standard MATH and PACK init.
 */
template <bool split_acc = false, bool dense_packing = false>
inline void custom_mm_compressed_block_init_short(
    const uint32_t in0_cb_id, const uint32_t in1_cb_id, const uint32_t out_cb_id) {
    llk_unpack_AB_custom_mm_compressed_init(in0_cb_id, in1_cb_id);
    MATH((llk_math_custom_mm_init<false, split_acc, dense_packing>(in0_cb_id, in1_cb_id, 1)));
    PACK((llk_pack_init<false, false>(out_cb_id)));
    if constexpr (dense_packing) {
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
}

}  // namespace compressed
