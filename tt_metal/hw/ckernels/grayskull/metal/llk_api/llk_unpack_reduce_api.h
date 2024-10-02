// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_reduce.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
* LLK UNPACK REDUCE
*************************************************************************/

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en = false /*not used*/, StochRndType stoch_rnd_mode = StochRndType::None/*not used*/>
inline void llk_unpack_reduce_hw_configure(
    const llk_unpack_reduce_params_t *unpack_reduce_params, const float const_mult) {

    const std::uint32_t unpA_operand_id = get_operand_id(unpack_reduce_params->unpA_operand);

    _llk_unpack_reduce_hw_configure_<type, dim>(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id]
    );
}

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en=false /*not used*/, StochRndType stoch_rnd_mode = StochRndType::None /*not used*/>
inline void llk_unpack_reduce_hw_configure_disaggregated(const std::uint32_t unpA_operand, const float mult) {
    const llk_unpack_reduce_params_t unpack_reduce_params = {.unpA_operand = unpA_operand};
    llk_unpack_reduce_hw_configure<type, dim, is_fp32_dest_acc_en, stoch_rnd_mode>(&unpack_reduce_params, mult);
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_mop_config() {
    _llk_unpack_reduce_mop_config_<type, dim>();
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_init(const std::uint32_t within_face_16x16_transpose=0) {

    WAYPOINT("UPRW");
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    constexpr std::uint32_t unpA_operand_id = 0;

    // Set first 32 bites of tile descriptor, only need data format change
    unpack_tile_descriptor_u tile_descriptor = {0};

    tile_descriptor.f.in_data_format  = (uint) DataFormat::Float32;
    tile_descriptor.f.uncompressed = 1; // Input tile is uncompressed
    tile_descriptor.f.x_dim        = 256;

    unpack_config_u config = {0};

    config.f.out_data_format = (((uint)unpack_dst_format[unpA_operand_id]>>2)&0x1) ? (uint) DataFormat::Float16_b : (uint) DataFormat::Float16;
    config.f.throttle_mode = 2;

    wait_for_idle();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);

    uint32_t alu_config_data = gl_alu_format_spec_reg;

    gl_alu_format_spec_reg = cfg_rmw_mmio_rd_tensix_wr(ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, ALU_FORMAT_SPEC_REG1_SrcB_SHAMT, ALU_FORMAT_SPEC_REG1_SrcB_MASK,
                                                        config.f.out_data_format,
                                                        alu_config_data);

    cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32] = tile_descriptor.val[0];
    cfg[THCON_SEC1_REG2_Out_data_format_ADDR32] = config.val[0];

    _llk_unpack_reduce_init_<type, dim>(within_face_16x16_transpose);
    WAYPOINT("UPRD");
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce(const std::uint32_t operand, const std::uint32_t tile_index) {

    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;
    std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX(unpack_src_format[operand_id], tile_index);
    std::uint32_t address = base_address + offset_address;

    WAYPOINT("UPRW");
    _llk_unpack_reduce_<type, dim>(
        address
    );
    WAYPOINT("UPRD");
}
