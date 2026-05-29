// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "llk_assert.h"
#include "llk_unpack_A.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// Low-level unpack for Top32 row-major tiles.

template <bool unpack_to_dest = false>
inline void _llk_unpack_A_top32_rm_init_(
    const std::uint32_t within_face_16x16_transpose,
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format) {
    if constexpr (unpack_to_dest) {
        if (is_32bit_input(unpack_src_format, unpack_dst_format)) {
            _llk_unpack_dbg_feature_disable_();
        }
    }

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose);

    TTI_SETADCXX(p_setadc::UNP_A, 16 - 1, 0x0);

    // set y stride in src to 16 data elements and dst to 16x16 data elements
    const DataFormat dst_format = static_cast<DataFormat>(unpack_dst_format & 0x3);
    const std::uint32_t unpA_x_stride = dst_format == DataFormat::Float32   ? 4
                                        : dst_format == DataFormat::Float16 ? 2
                                                                            : 1;
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(16 | (16 << 16));
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(16 * 16 * unpA_x_stride);
}

template <bool unpack_to_dest = false>
inline void _llk_unpack_A_top32_rm_(
    const std::uint32_t num_faces,
    const std::uint32_t address,
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format) {
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");

    TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1010);
    TTI_SETADCZW(0b001, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Set upk0/1 L1 read addr
    const std::uint32_t upk0_reg =
        (unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
    cfg[upk0_reg] = address;

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    if constexpr (unpack_to_dest) {
        if (is_32bit_input(unpack_src_format, unpack_dst_format)) {
            set_dst_write_addr(unp_cfg_context, unpack_dst_format);
            wait_for_dest_available();
        }
    }

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run unpack sequence
    if constexpr (unpack_to_dest) {
        for (std::uint32_t face_index = 0; face_index < num_faces; face_index++) {
            TTI_UNPACR(
                SrcA,
                0b01'00'01'00,
                0,
                0,
                0,
                1 /*Set OvrdThreadId*/,
                0 /*Set Dvalid*/,
                p_unpacr::RAREFYB_DISABLE,
                0,
                0,
                0,
                0,
                1);
        }
    } else {
        // clear A to -infinity
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
        // clear B to zero
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC);

        for (std::uint32_t face_index = 0; face_index < num_faces - 1; face_index++) {
            TTI_UNPACR(
                SrcA,
                0b01'00'01'00,
                0,
                0,
                0,
                1 /*Set OvrdThreadId*/,
                0 /*Set Dvalid*/,
                p_unpacr::RAREFYB_DISABLE,
                0,
                0,
                0,
                0,
                0);
        }

        TTI_UNPACR(
            SrcA,
            0b01'00'01'00,
            0,
            0,
            0,
            1 /*Set OvrdThreadId*/,
            1 /*Set Dvalid*/,
            p_unpacr::RAREFYB_DISABLE,
            0,
            0,
            0,
            0,
            1);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    if (unpack_to_dest) {
        if (is_32bit_input(unpack_src_format, unpack_dst_format)) {
            unpack_to_dest_tile_done(unp_cfg_context);
        }
    }

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
