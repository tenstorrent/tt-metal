// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Private opt-in pack MOP, derived from padded_pack_probe's device-qualified
// four-tile layout. This integration changes replay placement and scheduling;
// attention correctness/performance must be qualified independently.

#if defined(TRISC_PACK) && defined(SDPA_PADDED_P16_OUTPUT) && SDPA_PADDED_P16_PACK_WIDTH == 4
static_assert(REPLAY_BUF_SIZE == 32 && 17 + 15 <= REPLAY_BUF_SIZE);
// Channel 0 strides address DST. Channel 1 strides address output L1 only.
// Standard Default init does NOT restore channel-1 strides; reset explicitly.
static inline void padded_p16_output_y_stride(uint32_t bytes) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    TT_SETDMAREG(0, LOWER_HALFWORD(bytes << PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(bytes << PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
}

// One C++ pack call, four framed BF16 tiles, one L1 address setup.
// The 15 replayed PACRs produce the first 60 rows of one tile. A separate
// final PACR closes each tile. Ydst advances 64 rows per tile, so stride=64
// bytes produces tile starts 0,4096,8192,12288. Intermediate face transitions
// reset Ysrc and increment Zsrc, exactly as the standard default pack MOP.
static inline void padded_p16_load_pack_replay() {
    load_replay_buf(17, 15, [] {
        for (uint32_t row_group = 0; row_group < 15; ++row_group) {
            if (row_group % 4 == 3) {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_2,
                    p_pacr::ADDR_CNT_CTXT_0,
                    p_pacr::P_ZERO_OUTPUT_DISABLED,
                    p_pacr::ALL_INTF_ACTIVE,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    0);
            } else {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_0,
                    p_pacr::ADDR_CNT_CTXT_0,
                    p_pacr::P_ZERO_OUTPUT_DISABLED,
                    p_pacr::ALL_INTF_ACTIVE,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    0);
            }
        }
    });
}

static inline void padded_p16_init_pack_four() {
    // Native exp executes replay 0..7; cubic refiner reloads 8..21 itself.
    // Rebuild 17..31 after SFPU completes on EVERY exp batch. Never use >=32:
    // the replay index is five bits and the per-thread replay buffer is 32 words.
    padded_p16_load_pack_replay();
    llk_pack_init<PackMode::Default>(7, 4);
    padded_p16_output_y_stride(64);
    ckernel::ckernel_template mop(
        4,
        1,
        lltt::replay_insn(17, 15),
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            p_pacr::P_ZERO_OUTPUT_DISABLED,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            1));
    // Final tile restores Ysrc/Ydst/Zsrc, as ordinary Default pack does.
    mop.set_last_outer_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1));
    mop.program();
}
#endif
