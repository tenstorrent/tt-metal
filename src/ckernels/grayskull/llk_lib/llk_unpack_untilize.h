#pragma once

#include "llk_io_unpack.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

inline void llk_unpack_untilize_mop_config() {
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b01000001, 0, 0, 0, 0, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    static constexpr uint unpack_addcr = TT_OP_ADDRCRZW(0b001, 0, 0, 0, 0, 0b0001);
    static constexpr uint unpack_addr_offset =
        TT_OP_ADDDMAREG(0, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_SIZE);
    static constexpr uint unpack_wr_addr_offset = TT_OP_REG2FLOP(
        1, 0, 0, 0, THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TILE_OFFSET);
    // static constexpr uint unpack_inc_w_cnt = TT_OP_INCADCZW(0b001, 0, 0, 1, 0);
    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,  // src B
        true,  // halo - just used for 4 unpacks
        unpack_srca,
        unpack_srca,
        unpack_addr_offset,
        unpack_wr_addr_offset,
        0,
        unpack_addcr,
        TT_OP_NOP);

    tmp.program(instrn_buffer);
}

inline void llk_unpack_untilize_hw_configure(const llk_unpack_untilize_params_t *unpack_untilize_params) {
    configure_unpack_AB(
        get_operand_id(unpack_untilize_params->unpA_operand), get_operand_id(unpack_untilize_params->unpA_operand), 1);
}

inline void llk_unpack_untilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_untilize_params_t unpack_untilize_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_untilize_hw_configure(&unpack_untilize_params);
}

inline void llk_unpack_untilize_init(const uint32_t operand) {
    llk_unpack_untilize_mop_config();

    wait_for_idle();

    // // Override default settings
    std::uint32_t input = get_operand_id(operand);
    // Get pointer to registers for current state ID
    volatile uint *cfg = get_cfg_pointer();

    constexpr uint32_t srca_face_height = 1;
    constexpr uint32_t srcb_face_height = 16;

    uint unpA_ch1_x_stride = (uint) (unpack_dst_format[operand]&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format[operand]&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
    uint unpB_ch1_x_stride = (uint) (unpack_dst_format[operand]&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format[1]&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
    uint unpA_ch1_y_stride = 16*srca_face_height*unpA_ch1_x_stride;
    uint unpB_ch1_y_stride = 16*srcb_face_height*unpB_ch1_x_stride;

    cfg[UNP0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32] = (unpA_ch1_y_stride << UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT) |
                                                    (            0 << UNP0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT);  // X and Y stride for dest address (ch1)
    //cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =  // Z and W stride for dest address (ch1)
    cfg[UNP1_ADDR_CTRL_XY_REG_1_Xstride_ADDR32] = (unpB_ch1_y_stride << UNP1_ADDR_CTRL_XY_REG_0_Ystride_SHAMT) |
                                                    (            0 << UNP1_ADDR_CTRL_XY_REG_0_Xstride_SHAMT);  // X and Y stride for dest address (ch1)


    uint Tile_x_dim = FACE_HEIGHT;
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = Tile_x_dim | (Tile_x_dim << 16);
    unpack_tile_descriptor_u tile_descriptor;
    tile_descriptor.val[0] = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 0];
    tile_descriptor.val[1] = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1];
    tile_descriptor.f.x_dim = 16;
    tile_descriptor.f.y_dim = FACE_HEIGHT;
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 0] = tile_descriptor.val[0];
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1] = tile_descriptor.val[1];

    cfg[UNP0_ADDR_BASE_REG_0_Base_ADDR32] =
        ((((int)unpack_dst_format[input] & 0x3) == 1) ? 0x80 : 0x40)
        << UNP0_ADDR_BASE_REG_1_Base_SHAMT;  // base address skips halo rows in srcA (ch1)

    regfile[p_gpr_unpack::TILE_SIZE] = GET_L1_TILE_SIZE((uint)unpack_src_format[input]);
    regfile[p_gpr_unpack::TILE_OFFSET] = 0;
    TTI_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::TILE_OFFSET));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
}

inline void llk_unpack_untilize_uninit(uint32_t operand) {
    wait_for_idle();

    configure_unpack_AB(operand, operand, 16, 16, false, true);
}

template <bool first_pass = true>
inline void llk_unpack_untilize_(std::uint32_t operand, std::uint32_t block_c_tiles) {
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = cb_read_interface[input].fifo_rd_ptr - 1; // Remove header size added by descriptor
    std::uint32_t rem_blocks_in_row = block_c_tiles;

    // Program srcA and srcB base addresses
    volatile uint *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b0010);  // Clear l1 addr y cnt
    if constexpr (first_pass) {
        // Select bootom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 0);
    } else {
        // Select bootom faces in the 2nd pass
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Z, 2);
    }

    // Wait for free context
    wait_for_next_context(1);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Get tile address
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = base_address;


    std::uint32_t face_2xr_cnt = 0;
    for (std::uint32_t r = 0; r < FACE_HEIGHT; r++) {
        rem_blocks_in_row = block_c_tiles;  // reset remaining blocks in row

        do {
            if ((face_2xr_cnt + rem_blocks_in_row) >= (FACE_HEIGHT / 2)) {
                // Run MOP
                TT_MOP(0, 8 - face_2xr_cnt - 1, 0);                                              // Run the MOP
                TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);  // set data valid
                TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000);  // Clear srcA addr y cnt
                rem_blocks_in_row -= (8 - face_2xr_cnt);
                face_2xr_cnt = 0;
            } else {
                TT_MOP(0, rem_blocks_in_row - 1, 0);  // Run the MOP
                face_2xr_cnt += rem_blocks_in_row;
                rem_blocks_in_row = 0;
            }
        } while (rem_blocks_in_row > 0);

        TTI_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::TILE_OFFSET));  // Clear offset pointer
        TTI_REG2FLOP(
            1,
            0,
            0,
            0,
            THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32,
            p_gpr::ZERO);                 // Clear offset register
        TTI_INCADCXY(0b001, 0, 0, 1, 0);  // inc l1 addr y cnt
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Need to update the read pointer here
}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    llk_unpack_untilize_<true>(operand, block_c_tiles);
    llk_unpack_untilize_<false>(operand, block_c_tiles);
}
