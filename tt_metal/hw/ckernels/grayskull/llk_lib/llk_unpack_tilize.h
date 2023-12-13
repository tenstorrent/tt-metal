// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

inline void _llk_unpack_tilize_mop_config_() {
#if SKIP_UNP == 1
    static constexpr uint unpack_srca = TT_OP_NOP;
#else
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
    ckernel_unpack_template tmp = ckernel_unpack_template::lA(unpack_srca);
    tmp.program(instrn_buffer);
}

inline void _llk_unpack_tilize_hw_configure_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format) {
    configure_unpack_AB(
        unpack_src_format,
        unpack_src_format,
        unpack_dst_format,
        unpack_dst_format
    );
}

inline void _llk_unpack_tilize_init_(const std::uint32_t unpack_src_format=0, const std::uint32_t unpack_dst_format=0, const std::uint32_t ct_dim=0) {
    const std::uint32_t block_c_dim = ct_dim * TILE_C_DIM;

    // Save state of unpacker config for quick restore
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_0, THCON_SEC0_REG2_Out_data_format_ADDR32); // Save unpack config[0]
    TTI_RDCFG(p_gpr_unpack::SR_UNPACK_TILIZER_STATE_1, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32); // Save tile x dim per context

    // Override default settings
    unpack_config_u config = {0};
    config.f.out_data_format = unpack_dst_format;
    config.f.throttle_mode = 2;
    config.f.tileize_mode = 1;
    config.f.shift_amount = (SCALE_DATUM_SIZE(unpack_src_format, block_c_dim)) >> 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0]
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 << 16)

    _llk_unpack_tilize_mop_config_();
}

inline void _llk_unpack_tilize_(const std::uint32_t base_address, const std::uint32_t tile_index, const std::uint32_t unpack_src_format, const std::uint32_t block_ct_dim) {

    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE(unpack_src_format, tile_index) << 1;
                                                   // Each iteration unpacks 2 16x16 faces (1st 0,1 2nd 2,3)
                                                   // Offset address is in 16B words
                                                   // Datum count = tile_index*16 (/16 to get word count)

    std::uint32_t bot_face_offset_address =
        SCALE_DATUM_SIZE(unpack_src_format, block_ct_dim*TILE_C_DIM);  //*16 rows / 16 to get 16B word aligned address

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    for (std::uint32_t n = 0; n < 2; n++) {
        std::uint32_t address = base_address + top_face_offset_address + ((n == 1) ? bot_face_offset_address : 0);

        // Clear z/w start counters
        TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

        // Wait for free context
        wait_for_next_context(2);

        // Trisc::SEMPOST for context acquire
        semaphore_post(semaphore::UNPACK_SYNC);

        // Get tile address
        if (0 == unp_cfg_context) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        }

        // Run MOP
        mop_run(0, 2);

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}
