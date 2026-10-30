// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef TRISC_PACK

// Shared PACK stage for the max-util and slow-cos workloads. The configured
// MOP drains all num_tiles tiles from DST to the fixed L1 output in one run.
ALWI void didt_pack_bfloat16_tiles(uint32_t num_loops, uint32_t num_tiles, uint32_t l1_output_addr) {
    constexpr bool is_fp32_dest_acc_en = false;
    _llk_pack_hw_configure_<is_fp32_dest_acc_en>(
        (uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float16_b, 128 /* tile size for float16_b >> 4 */);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    addr_mod_pack_t{
        .y_src = {.incr = 4},
    }
        .set(ADDR_MOD_0);
    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_1);
    addr_mod_pack_t{
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .z_src = {.incr = 1, .clr = 0},
    }
        .set(ADDR_MOD_2);

    const uint32_t mop_inner_loop = 4;              // face_r_dim >> 2
    const uint32_t mop_outer_loop = 4 * num_tiles;  // num_faces * num_tiles
    ckernel::ckernel_template pack_mop(
        mop_outer_loop,
        mop_inner_loop,
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            p_pacr::P_ZERO_OUTPUT_DISABLED,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            0,
            0,
            0,
            0));
    pack_mop.set_last_inner_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_2,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        0));
    pack_mop.set_last_outer_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        1));
    pack_mop.program();
    set_dst_write_addr(0);
    program_packer_destination(L1_ADDRESS(l1_output_addr));

    for (uint32_t i = 0; i < num_loops; i++) {
        _llk_packer_wait_for_math_done_();
        ckernel::ckernel_template::run();
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif  // TRISC_PACK
