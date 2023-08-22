
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
    constexpr uint replay_buf_len = 5;
    TTI_REPLAY(0, replay_buf_len, 0, 1);
    TTI_UNPACR(SrcA, 0b01000001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, 0b01000001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_ADDDMAREG(0, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_OFFSET, p_gpr_unpack::TILE_SIZE);
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TILE_OFFSET);
    TTI_ADDRCRZW(0b001, 0, 0, 0, 0, 0b0001);
    ckernel_unpack_template tmp = ckernel_unpack_template(
        false,  // src B
        false,  // halo - just used for 4 unpacks
        TT_OP_REPLAY(0, replay_buf_len, 0, 0),
        0,
        0,
        0,
        0,
        0,
        0);

    tmp.program(instrn_buffer);
}

inline void llk_unpack_untilize_hw_configure(const llk_unpack_untilize_params_t *unpack_untilize_params) {
    configure_unpack_AB(
        get_operand_id(unpack_untilize_params->unpA_operand), get_operand_id(unpack_untilize_params->unpA_operand), 1);
    // Override default settings
    std::uint32_t input = get_operand_id(unpack_untilize_params->unpA_operand);

    uint unpA_ch1_x_stride = (uint) (unpack_dst_format[unpack_untilize_params->unpA_operand]&0x3) == (uint) DataFormat::Float32 ? 4 : (uint) (unpack_dst_format[unpack_untilize_params->unpA_operand]&0x3) == (uint) DataFormat::Float16 ? 2 : 1;
    uint unpA_ch1_y_stride = 16*unpA_ch1_x_stride;
    // Get pointer to registers for current state ID
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(unpA_ch1_y_stride);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xFFFF>(FACE_WIDTH | (FACE_WIDTH << 16));
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 16, 0xFFFF00>(FACE_WIDTH);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32+1, 0, 0xFFFF>(FACE_HEIGHT);
    regfile[p_gpr_unpack::TILE_SIZE] = cb_interface[input].fifo_page_size;
    regfile[p_gpr_unpack::TILE_OFFSET] = 0;
    sync_regfile_write(p_gpr_unpack::TILE_OFFSET);
    TTI_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::TILE_OFFSET));
    TTI_SETDMAREG(0, 0, 0, HI_16(p_gpr_unpack::TILE_OFFSET));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG7_Offset_address_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);

}

inline void llk_unpack_untilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_untilize_params_t unpack_untilize_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_untilize_hw_configure(&unpack_untilize_params);
}

inline void llk_unpack_untilize_init() { llk_unpack_untilize_mop_config(); }

template <bool first_pass = true>
inline void llk_unpack_untilize_(std::uint32_t operand, std::uint32_t block_tile_cols) {
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[input].fifo_rd_ptr - 1;
    std::uint32_t rem_blocks_in_row = block_tile_cols;

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

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
        rem_blocks_in_row = block_tile_cols;  // reset remaining blocks in row

        do {
            if ((face_2xr_cnt + rem_blocks_in_row) >= (FACE_HEIGHT / 2)) {
                // Run MOP
                TT_MOP(0, 8 - face_2xr_cnt - 1, 0);                                              // Run the MOP
#if SKIP_UNP == 1
                TTI_NOP;
#else
                TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);  // set data valid

                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
                TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
#endif
                TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000);  // Clear srcA addr y cnt
                rem_blocks_in_row -= (8 - face_2xr_cnt);
                face_2xr_cnt = 0;
            } else {
                TT_MOP(0, rem_blocks_in_row - 1, 0);  // Run the MOP
                face_2xr_cnt += rem_blocks_in_row;
                rem_blocks_in_row = 0;
                // if (face_2xr_cnt==FACE_HEIGHT/2) {
                //   TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); //set data valid
                //   TTI_SETADCXY(0b001, 0, 0, 0, 0, 0b1000); // Clear srcA addr y cnt
                //   face_2xr_cnt = 0;
                //}
            }
        } while (rem_blocks_in_row > 0);

        TTI_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::TILE_OFFSET));  // Clear offset pointer
        TTI_SETDMAREG(0, 0, 0, HI_16(p_gpr_unpack::TILE_OFFSET));  // Clear offset pointer
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

}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    llk_unpack_untilize_<true>(operand, block_c_tiles);
    llk_unpack_untilize_<false>(operand, block_c_tiles);
}

inline void llk_unpack_untilize_uninit(uint32_t operand) {
    wait_for_idle();

    configure_unpack_AB(operand, operand, 16, 16, false, true);
}
