// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_defs.h"

#include "ckernel.h"
#include "ckernel_template.h"
#include "llk_pack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::packer;

inline void _llk_pack_untilize_configure_addrmod_() {

    addr_mod_pack_t{
        .y_src = {.incr = 15}, // 4-bit value so max is 15. incadcxy will increment it by 1
    }
    .set(ADDR_MOD_0);

    addr_mod_pack_t{
        .y_src = { .incr = 0, .clr = 0, .cr = 1  },
    }.set(ADDR_MOD_1);

    addr_mod_pack_t{
        .y_src = { .incr = 0, .clr = 1, .cr = 0  },
    }.set(ADDR_MOD_2);

}

template <std::uint32_t block_ct_dim>
inline void _llk_pack_untilize_mop_config_() {
    const uint PACKCNT = 4;
    constexpr uint MEGAROW = 1;
    constexpr uint ZERO_OUTPUT_FLAG = p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr uint MOP_INNER_LOOP = 1;

    constexpr uint MOP_OUTER_LOOP = block_ct_dim;

    // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
    ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
    tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
    tmp.set_end_ops(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0),
                    TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 0, 1)); // z cnt points to the next tile
    tmp.program(instrn_buffer);
}

template <std::uint32_t block_ct_dim>
inline void _llk_pack_untilize_init_() {

    _llk_pack_untilize_configure_addrmod_();

    _llk_pack_untilize_mop_config_<block_ct_dim>();
}

template <std::uint32_t block_ct_dim>
inline void _llk_pack_untilize_(const std::uint32_t address, const std::uint32_t pack_dst_format) {

    program_packer_untilized_destination<block_ct_dim>(address, pack_dst_format);

    for (std::uint32_t row=0; row<TILE_R_DIM/4; row++) {
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Z, 0); // Clear tile counter
        ckernel::ckernel_template::run(instrn_buffer);
        TTI_ADDRCRXY(p_setadc::PAC, 0, 0, 1, 0, 0b0010); // Read new row in the tile
    }

    TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 1); // close block
}
