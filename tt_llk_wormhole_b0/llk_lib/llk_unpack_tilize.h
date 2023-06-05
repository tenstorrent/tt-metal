
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
    constexpr uint replay_buf_len = 3;

    #if SKIP_UNP0 == 1
        TTI_REPLAY(0, 1, 0, 1);
        TTI_NOP;
    #else
        TTI_REPLAY(0, replay_buf_len, 0, 1);
        
        TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
        TTI_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    #endif    

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

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure(const llk_unpack_tilize_params_t *unpack_tilize_params) {

    constexpr bool is_row_pool = false;
    constexpr uint32_t srca_height = 16;
    constexpr uint32_t srcb_height = 16;
    constexpr bool transpose_xy_srca = false;
    const uint32_t srca_operand = get_operand_id(unpack_tilize_params->unpA_operand);
    configure_unpack_AB(srca_operand, srca_operand, srca_height, srcb_height, is_row_pool, transpose_xy_srca, is_fp32_dest_acc_en);

    // Override default settings
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();
    unpack_config_u config = {0};

    config.f.out_data_format = (uint)unpack_dst_format[srca_operand];
    config.f.throttle_mode = 2;
    config.f.tileize_mode = 1;
    config.f.shift_amount = (SCALE_DATUM_SIZE((uint)unpack_src_format[srca_operand], unpack_tilize_params->unpA_block_c_dim)) >> 4;

    wait_for_idle();

    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = 16 | (16 << 16);
}


template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpA_block_ct_dim) {
    const llk_unpack_tilize_params_t unpack_tilize_params = {
        .unpA_operand = unpA_operand,
        .unpA_block_c_dim = unpA_block_ct_dim*TILE_WIDTH,
    };
    llk_unpack_tilize_hw_configure<is_fp32_dest_acc_en>(&unpack_tilize_params);
}

inline void llk_unpack_tilize_init() { llk_unpack_tilize_mop_config(); }

inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t block_ct_dim) {
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = operands[input].f.fifo_rd_ptr - 1;  // Remove header size added by descriptor
    std::uint32_t top_face_offset_address = SCALE_DATUM_SIZE((uint)unpack_src_format[input], tile_index) << 1;  
                                                    // Each iteration unpacks 2 16x16 faces (1st 0,1 2nd 2,3)
                                                    // Offset address is in 16B words
                                                    // Datum count = tile_index*16 (/16 to get word count)

    std::uint32_t bot_face_offset_address =
        SCALE_DATUM_SIZE((uint)unpack_src_format[input], block_ct_dim*TILE_WIDTH);  //*16 rows / 16 to get 16B word aligned address

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
