// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#ifdef TRISC_PACK
// Channel 0 strides address DST. Channel 1 strides address output L1 only.
// Standard Default init does NOT restore channel-1 strides; reset explicitly.
static inline void output_y_stride(uint32_t bytes) {
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
static inline void load_padded_replay() {
    load_replay_buf(0, 15, [] {
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

static inline void init_padded_four() {
    llk_pack_init<PackMode::Default>(8, 4);
    output_y_stride(64);
    ckernel::ckernel_template mop(
        4,
        1,
        lltt::replay_insn(0, 15),
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

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t tile_count = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);  // 0 scalar,1 compact4,2 padded4
    static_assert(batch == (DST_ACCUM_MODE ? 4 : 8));
    static_assert(mode <= 2);
    compute_kernel_hw_startup(0, 8);
    // Default scalar pack init never writes the pack thread's replay buffer.
    if constexpr (mode == 2) {
        PACK((load_padded_replay()));
    }
    for (uint32_t tile = 0; tile < tile_count; tile += batch) {
        if (tile != 0) {
            reconfig_data_format_srca(8, 0);
        }
        copy_init(0);
        MATH((ckernel::math::_configure_src_zero_flag_(false)));
        // All three CBs have BF16 format, so format-reconfig shortcuts alone
        // do not rebuild MOPs. Explicitly initialize each stage.
        pack_init(8);
        PACK((output_y_stride(0)));
        if constexpr (mode == 1) {
            PACK((llk_pack_init<PackMode::Default>(8, 4)));
        }
        if constexpr (mode == 2) {
            PACK((init_padded_four()));
        }
        cb_wait_front(0, batch);
        cb_reserve_back(8, batch);
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; ++j) {
            copy_tile(0, j, j);
        }
        tile_regs_commit();
        cb_pop_front(0, batch);
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; j += (mode == 0 ? 1 : 4)) {
            pack_tile<true>(j, 8, j);
        }
        tile_regs_release();
        cb_push_back(8, batch);

        cb_wait_front(8, batch);
        reconfig_data_format_srca(0, 8);
        copy_init(8);
        MATH((ckernel::math::_configure_src_zero_flag_(false)));
        pack_init(16);
        PACK((output_y_stride(0)));
        cb_reserve_back(16, batch);
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; ++j) {
            copy_tile(8, j, j);
        }
        tile_regs_commit();
        cb_pop_front(8, batch);
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; ++j) {
            pack_tile<true>(j, 16, j);
        }
        tile_regs_release();
        cb_push_back(16, batch);
    }
}
