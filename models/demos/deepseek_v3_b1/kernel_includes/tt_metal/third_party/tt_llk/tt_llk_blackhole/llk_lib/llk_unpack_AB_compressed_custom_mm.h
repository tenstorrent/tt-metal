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

inline void _llk_unpack_AB_compressed_custom_mm_mop_config_() {
    constexpr std::uint8_t set_dvadlid = 1;
    load_replay_buf(0, 25, [] {
        // Bfp8
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
        TTI_WRCFG(p_gpr_unpack::PERF_UNPACK_NUM_TILES_3, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32);
        TTI_NOP;
        TTI_UNPACR_COMMON(SrcA, 0b00000000, set_dvadlid);
        TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;
        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110100, set_dvadlid);
        // Bfp4
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
        TTI_WRCFG(p_gpr_unpack::PERF_UNPACK_NUM_TILES_2, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32);
        TTI_NOP;
        TTI_UNPACR_COMMON(SrcA, 0b00000000, set_dvadlid);
        TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 1, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;
        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110100, set_dvadlid);
        // Bfp2
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
        TTI_WRCFG(p_gpr_unpack::PERF_UNPACK_NUM_TILES_1, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32);
        TTI_NOP;
        TTI_UNPACR_COMMON(SrcA, 0b00000000, set_dvadlid);
        TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 2, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;
        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110100, set_dvadlid);
        TTI_UNPACR_NOP(SrcA, 0, 0, set_dvadlid, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
    });
}

template <bool transpose = false>
inline void _llk_unpack_AB_compressed_custom_mm_init_(const std::uint32_t unpB_face_r_dim) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    constexpr std::uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
    const std::uint32_t unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1;
    TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

    _llk_unpack_AB_compressed_custom_mm_mop_config_();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

constexpr std::uint32_t get_replay_insn_for_combo(const std::uint8_t combo) {
    std::uint8_t prev = combo & 0b11;
    std::uint8_t curr = (combo >> 3) & 0b11;

    bool use_b = (combo >> 2) & 0b1;
    bool need_reconfig = prev != curr;
    bool need_stall = need_reconfig && (prev == 1 || curr == 1);

    std::uint32_t start_idx = curr == 3 ? 0 : curr == 2 ? 8 : 16;
    std::uint32_t start_offset = 3;
    std::uint32_t replay_len = 3;

    if (use_b) {
        replay_len += 2;
    }
    if (need_reconfig) {
        start_offset -= 2;
        replay_len += 2;
    }
    if (need_stall) {
        start_offset -= 1;
        replay_len += 1;
    }

    if (curr == 0) {
        if (use_b) {
            return lltt::replay_insn(22, 3);
        } else {
            return lltt::replay_insn(24, 1);
        }
    }

    return lltt::replay_insn(start_idx + start_offset, replay_len);
}

template <bool clear_src = true>
inline void _llk_unpack_AB_compressed_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t base_address_meta,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    constexpr std::uint32_t FMTABLE[32] = {
        get_replay_insn_for_combo(0b000'00),  // 0b000'00 zero to zero
        get_replay_insn_for_combo(0b000'01),  // 0b000'01 bfp2 to zero
        get_replay_insn_for_combo(0b000'10),  // 0b000'10 bfp4 to zero
        get_replay_insn_for_combo(0b000'11),  // 0b000'11 bfp8 to zero

        get_replay_insn_for_combo(0b001'00),  // 0b001'00 zero to zero with b
        get_replay_insn_for_combo(0b001'01),  // 0b001'01 bfp2 to zero with b
        get_replay_insn_for_combo(0b001'10),  // 0b001'10 bfp4 to zero with b
        get_replay_insn_for_combo(0b001'11),  // 0b001'11 bfp8 to zero with b

        get_replay_insn_for_combo(0b010'00),  // 0b010'00 zero to bfp2
        get_replay_insn_for_combo(0b010'01),  // 0b010'01 bfp2 to bfp2
        get_replay_insn_for_combo(0b010'10),  // 0b010'10 bfp4 to bfp2
        get_replay_insn_for_combo(0b010'11),  // 0b010'11 bfp8 to bfp2

        get_replay_insn_for_combo(0b011'00),  // 0b011'00 zero to bfp2 with b
        get_replay_insn_for_combo(0b011'01),  // 0b011'01 bfp2 to bfp2 with b
        get_replay_insn_for_combo(0b011'10),  // 0b011'10 bfp4 to bfp2 with b
        get_replay_insn_for_combo(0b011'11),  // 0b011'11 bfp8 to bfp2 with b

        get_replay_insn_for_combo(0b100'00),  // 0b100'00 zero to bfp4
        get_replay_insn_for_combo(0b100'01),  // 0b100'01 bfp2 to bfp4
        get_replay_insn_for_combo(0b100'10),  // 0b100'10 bfp4 to bfp4
        get_replay_insn_for_combo(0b100'11),  // 0b100'11 bfp8 to bfp4

        get_replay_insn_for_combo(0b101'00),  // 0b101'00 zero to bfp4 with b
        get_replay_insn_for_combo(0b101'01),  // 0b101'01 bfp2 to bfp4 with b
        get_replay_insn_for_combo(0b101'10),  // 0b101'10 bfp4 to bfp4 with b
        get_replay_insn_for_combo(0b101'11),  // 0b101'11 bfp8 to bfp4 with b

        get_replay_insn_for_combo(0b110'00),  // 0b110'00 zero to bfp8
        get_replay_insn_for_combo(0b110'01),  // 0b110'01 bfp2 to bfp8
        get_replay_insn_for_combo(0b110'10),  // 0b110'10 bfp4 to bfp8
        get_replay_insn_for_combo(0b110'11),  // 0b110'11 bfp8 to bfp8

        get_replay_insn_for_combo(0b111'00),  // 0b111'00 zero to bfp8 with b
        get_replay_insn_for_combo(0b111'01),  // 0b111'01 bfp2 to bfp8 with b
        get_replay_insn_for_combo(0b111'10),  // 0b111'10 bfp4 to bfp8 with b
        get_replay_insn_for_combo(0b111'11),  // 0b111'11 bfp8 to bfp8 with b
    };

    volatile uint* cfg = get_cfg_pointer();

    const std::uint32_t address_a = base_address_a;
    const std::uint32_t address_b = base_address_b;
    const std::uint32_t full_iters = (kt_dim * ct_dim) / 10;
    const std::uint32_t rem_iters = (kt_dim * ct_dim) % 10;
    const std::uint32_t* meta_ptr = reinterpret_cast<const std::uint32_t*>(base_address_meta);

    wait_for_next_context(1);
    reset_config_context();

    if constexpr (clear_src) {
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    }

    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    cfg[SCRATCH_SEC0_val_ADDR32] = 68;
    cfg[SCRATCH_SEC1_val_ADDR32] = 36;
    cfg[SCRATCH_SEC2_val_ADDR32] = 20;
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_3] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp8_b);
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_2] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp4_b);
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_1] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp2_b);

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    for (std::uint32_t i = 0; i < full_iters; ++i) {
        std::uint32_t meta = meta_ptr[i];

        std::uint32_t idx0 = (meta >> 0) & 0b11111;
        std::uint32_t idx1 = (meta >> 3) & 0b11111;
        std::uint32_t idx2 = (meta >> 6) & 0b11111;
        std::uint32_t idx3 = (meta >> 9) & 0b11111;
        std::uint32_t idx4 = (meta >> 12) & 0b11111;
        std::uint32_t idx5 = (meta >> 15) & 0b11111;
        std::uint32_t idx6 = (meta >> 18) & 0b11111;
        std::uint32_t idx7 = (meta >> 21) & 0b11111;
        std::uint32_t idx8 = (meta >> 24) & 0b11111;
        std::uint32_t idx9 = (meta >> 27) & 0b11111;

        std::uint32_t data0 = FMTABLE[idx0];
        std::uint32_t data1 = FMTABLE[idx1];
        std::uint32_t data2 = FMTABLE[idx2];
        std::uint32_t data3 = FMTABLE[idx3];
        std::uint32_t data4 = FMTABLE[idx4];
        std::uint32_t data5 = FMTABLE[idx5];
        std::uint32_t data6 = FMTABLE[idx6];
        std::uint32_t data7 = FMTABLE[idx7];
        std::uint32_t data8 = FMTABLE[idx8];
        std::uint32_t data9 = FMTABLE[idx9];

        ckernel::instrn_buffer[0] = data0;
        ckernel::instrn_buffer[0] = data1;
        ckernel::instrn_buffer[0] = data2;
        ckernel::instrn_buffer[0] = data3;
        ckernel::instrn_buffer[0] = data4;
        ckernel::instrn_buffer[0] = data5;
        ckernel::instrn_buffer[0] = data6;
        ckernel::instrn_buffer[0] = data7;
        ckernel::instrn_buffer[0] = data8;
        ckernel::instrn_buffer[0] = data9;
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t i = 0; i < rem_iters; ++i) {
        std::uint32_t idx0 = (meta >> 0) & 0b11111;
        std::uint32_t data0 = FMTABLE[idx0];
        ckernel::instrn_buffer[0] = data0;
        meta >>= 3;
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    wait_for_next_context(1);
    reset_config_context();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}
