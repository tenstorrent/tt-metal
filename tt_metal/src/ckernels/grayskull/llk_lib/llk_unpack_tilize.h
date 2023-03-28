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

inline void llk_unpack_tilize_mop_config() {
#if SKIP_UNP0 == 1
    static constexpr uint unpack_srca = TT_OP_NOP;
#else
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
    ckernel_unpack_template tmp = ckernel_unpack_template::lA(unpack_srca);
    tmp.program(instrn_buffer);
}

inline void llk_unpack_tilize_hw_configure(const llk_unpack_tilize_params_t *unpack_tilize_params) {
    configure_unpack_AB(
        get_operand_id(unpack_tilize_params->unpA_operand), get_operand_id(unpack_tilize_params->unpA_operand));
}

inline void llk_unpack_tilize_hw_configure_disaggregated(
    const std::uint32_t unpA_operand) {
    const llk_unpack_tilize_params_t unpack_tilize_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_tilize_hw_configure(&unpack_tilize_params);
}

inline void llk_unpack_tilize_uninit() {
    // Undo tilize if added
    wait_for_idle();
    volatile uint *cfg = get_cfg_pointer();
    unpack_config_u config;

    config.val[0] = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 0];
    config.f.tileize_mode = 0;
    config.f.shift_amount = 0;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = 256 | (256 << 16);
}

inline void llk_unpack_tilize_init(const uint32_t unpA_operand, const uint32_t unpA_block_c_tiles) {
    llk_unpack_tilize_mop_config();
    // Only doing this to bypass need to pass in args to this func. Too difficult to
    // do in HLKC right now, and we're going to make clean APIs without need of
    // HLKC anyways, so good enough for time being.

    wait_for_idle();

    // Override default settings
    std::uint32_t input = get_operand_id(unpA_operand);
    unpack_config_u config;
    volatile uint *cfg = get_cfg_pointer();
    config.val[0] = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 0];
    config.f.tileize_mode = 1;
    config.f.shift_amount =
        (SCALE_DATUM_SIZE((uint)unpack_src_format[input], unpA_block_c_tiles << 5)) >> 4;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = 16 | (16 << 16);


}

inline void llk_unpack_tilize_block(std::uint32_t operand, std::uint32_t block_c_tiles) {

    std::uint32_t input = get_operand_id(operand);
    for (std::uint32_t tile_index = 0; tile_index < block_c_tiles; tile_index++) {
        std::uint32_t base_address = cb_read_interface[input].fifo_rd_ptr - 1; // Remove header size added by descriptor
        std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE((uint)unpack_src_format[input], tile_index)
                                                << 1;  // Each iteration unpacks 2 16x16 faces (1st 0,1 2nd 2,3)
                                                    // Offset address is in 16B words
                                                    // Datum count = tile_index*16 (/16 to get word count)

        std::uint32_t bot_face_offset_address =
            SCALE_DATUM_SIZE((uint)unpack_src_format[input], block_c_tiles << 5);  //*16 rows / 16 to get 16B word aligned address

        // Program srcA and srcB base addresses
        volatile uint *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

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

            // Stall unpacker until pending CFG writes from Trisc have completed
            TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

            // Run MOP
            mop_run(0, 2);

            // T6::SEMGET for context release
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            // Switch unpacker config context
            switch_config_context(unp_cfg_context);
        }
    }

}

// inline void llk_unpack_tilize(std::uint32_t operand, int tile_index, std::uint32_t block_c_tiles) {
//      // Since unpack is tightly coupled with math, and math datacopy works on the tile level,
//     // we need a mechanism to only run the unpacking of a block once
//     bool nop = tile_index != 0;
//     if (nop) {
//         return;
//     }
//     llk_unpack_tilize_(operand, block_c_tiles);
// }
