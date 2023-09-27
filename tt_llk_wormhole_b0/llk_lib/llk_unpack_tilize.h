
#include "llk_io_unpack.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

inline void llk_unpack_tilize_mop_config(const std::uint32_t operand_id) {
    #if SKIP_UNP == 1
        static constexpr uint unpack_srca = TT_OP_NOP;
        static constexpr uint unpack_srcb_zerosrc = TT_OP_NOP;
        static constexpr uint unpack_srcb_set_dvalid = TT_OP_NOP;
    #else
        static constexpr uint unpack_srca = TT_OP_UNPACR(SrcA, 0b1 /*Z inc*/, 0, 0, 0, 1 /* Set OvrdThreadId*/, 1 /*Set Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        static constexpr uint unpack_srcb_zerosrc    = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_ZEROSRC);
        static constexpr uint unpack_srcb_set_dvalid = TT_OP_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID); //WA for tenstorrent/budabackend#1230
    #endif    

    const uint32_t outerloop = get_narrow_tile(operand_id) ? 1 : 2;
    constexpr uint32_t innerloop = 1;   
    ckernel_template tmp(outerloop, innerloop, unpack_srcb_zerosrc, unpack_srcb_set_dvalid);
    tmp.set_start_op(unpack_srca);
    tmp.program(instrn_buffer);
}

template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure(const llk_unpack_tilize_params_t *unpack_tilize_params) {

    constexpr bool is_row_pool = false;
    constexpr bool transpose_xy_srca = false;
    constexpr bool srnd_fpu_en = false;

    const uint32_t unpA_operand_id = get_operand_id(unpack_tilize_params->unpA_operand);
    const uint32_t unpA_num_faces = get_num_faces(unpA_operand_id);
    const uint32_t unpA_face_r_dim = get_face_r_dim(unpA_operand_id);

    configure_unpack_AB(unpA_operand_id, unpA_operand_id, unpA_face_r_dim, unpA_face_r_dim, is_row_pool, transpose_xy_srca, is_fp32_dest_acc_en, srnd_fpu_en, unpA_num_faces, unpA_num_faces);

    const std::uint32_t unpA_block_c_dim = unpack_tilize_params->unpA_block_ct_dim * (get_narrow_tile(unpA_operand_id) ? FACE_C_DIM : TILE_C_DIM);

    // Override default settings
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();
    unpack_config_u config = {0};

    config.f.out_data_format = (uint)unpack_dst_format[unpA_operand_id];
    config.f.throttle_mode = 2;
    config.f.tileize_mode = 1;
    config.f.shift_amount = (SCALE_DATUM_SIZE((uint)unpack_src_format[unpA_operand_id], unpA_block_c_dim)) >> 4;

    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[0]), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(config.val[0]), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG2_Out_data_format_ADDR32+0-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0); // Load unpack config[0] 
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16); //GPR preloaded with  16 | (16 << 16)
}


template <bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpA_block_ct_dim) {
    TT_LLK_DUMP("llk_unpack_tilize_hw_configure_disaggregated<{}>({}, {})", is_fp32_dest_acc_en, unpA_operand, unpA_block_ct_dim);
    const llk_unpack_tilize_params_t unpack_tilize_params = {
        .unpA_operand = unpA_operand,
        .unpA_block_ct_dim = unpA_block_ct_dim
    };
    llk_unpack_tilize_hw_configure<is_fp32_dest_acc_en>(&unpack_tilize_params);
}

inline void llk_unpack_tilize_init(const std::uint32_t operand = 0) {
    TT_LLK_DUMP("llk_unpack_tilize_init()");
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);

    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_face_r_dim(operand_id);

    constexpr std::uint32_t UNP_SEL = p_setadc::UNP_A;
    TT_SETADCXX(UNP_SEL, face_r_dim*FACE_C_DIM-1, 0x0);
    TTI_REG2FLOP(1,0,0,0,THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::FACE_DIM_1x16);
    llk_unpack_tilize_mop_config(operand_id);
}

inline void llk_unpack_tilize(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t block_ct_dim) {
    TT_LLK_DUMP("llk_unpack_tilize({}, {}, {})", operand, tile_index, block_ct_dim);
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = operands[operand_id].f.fifo_rd_ptr - 1;  // Remove header size added by descriptor
    const std::uint32_t face_r_dim = get_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_num_faces(operand_id);
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID


    std::uint32_t top_face_offset_address = ((SCALE_DATUM_SIZE((uint)unpack_src_format[operand_id], tile_index) * face_r_dim) >> 4) << (get_narrow_tile(operand_id) ? 0 : 1);  
                                                    // Each iteration unpacks 2 face_r_dimx16 faces (1st 0,1 2nd 2,3 unless tile is <=16x32)
                                                    // For narrow tile we unpack 1 face in each iteration
                                                    // Offset address is in 16B words
                                                    // Datum count = tile_index*face_r_dim (/16 to get word count)

    const std::uint32_t block_c_dim_16B = block_ct_dim * (get_narrow_tile(operand_id) ? FACE_C_DIM/16 : TILE_C_DIM/16);
    std::uint32_t bot_face_offset_address =
        SCALE_DATUM_SIZE((uint)unpack_src_format[operand_id], face_r_dim*block_c_dim_16B);  //*N rows / 16 to get 16B word aligned address

    // Program srcA and srcB base addresses
    std::uint32_t num_loops = get_narrow_tile(operand_id) ? 2 : num_faces/2;

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
        ckernel::ckernel_template::run(instrn_buffer);

        // T6::SEMGET for context release
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Switch unpacker config context
        switch_config_context(unp_cfg_context);
    }

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}
