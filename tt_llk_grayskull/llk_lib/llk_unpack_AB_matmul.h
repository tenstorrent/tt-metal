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

inline void _llk_unpack_AB_matmul_mop_config_(const bool transpose) {
    /*
    static constexpr uint unpack_srcb_top  = TT_OP_UNPACR(SrcB, 0b01000001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0,
    0, 0, 0, 1); static constexpr uint unpack_srcb_bot =  TT_OP_UNPACR(SrcB, 0b01000001, 0, 0, 0, 1, 1,
    p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); static constexpr uint unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1,
    1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); ckernel_unpack_template tmp  = ckernel_unpack_template(false, // src B
                                                            true, // halo - just used for 4 unpacks
                                                            unpack_srcb_top,
                                                            unpack_srcb_bot,
                                                            unpack_srca,
                                                            unpack_srca,
                                                            0, 0, 0);
    */
    // UNPACK SRCB Z 0,2,1,3
    static constexpr uint unpack_src_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 1, 0b0001);
    static constexpr uint unpack_src_set_z_transpose = TT_OP_SETADCZW(0b011, 0, 0, 0, 1, 0b0001);
#if SKIP_UNP == 1
    static constexpr uint unpack_srca0 = TT_OP_NOP;
    static constexpr uint unpack_srca1 = TT_OP_NOP;
    static constexpr uint unpack_srca0_transpose = TT_OP_NOP;
    static constexpr uint unpack_srca1_transpose = TT_OP_NOP;

    static constexpr uint unpack_srcb_top = TT_OP_NOP;
    static constexpr uint unpack_srcb_bot = TT_OP_NOP;
#else
    static constexpr uint unpack_srca0 = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srca1 = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    static constexpr uint unpack_srca0_transpose = TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srca1_transpose = TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    static constexpr uint unpack_srcb_top =
        TT_OP_UNPACR(SrcB, 0b010010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb_bot =
        TT_OP_UNPACR(SrcB, 0b010010, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,  // src B
        true,  // halo - just used for 4 unpacks
        unpack_srcb_top,
        unpack_srcb_bot,
        transpose ? unpack_srca0_transpose : unpack_srca0,
        transpose ? unpack_srca1_transpose : unpack_srca1,
        0,
        transpose ? unpack_src_set_z_transpose : unpack_src_set_z,
        0);

    tmp.program(instrn_buffer);
}

inline void _llk_unpack_AB_matmul_hw_configure_(const std::uint32_t unpA_src_format, const std::uint32_t unpB_src_format, const std::uint32_t unpA_dst_format, const std::uint32_t unpB_dst_format) {
    configure_unpack_AB(
        unpA_src_format,
        unpB_src_format,
        unpA_dst_format,
        unpB_dst_format,
        16, 16
    );
}

inline void _llk_unpack_AB_matmul_init_(
    const std::uint32_t transpose=0, const std::uint32_t ct_dim=0 /* unused */, const std::uint32_t rt_dim=0 /* unused */, const std::uint32_t kt_dim=0 /* unused */) {
    _llk_unpack_AB_matmul_mop_config_(transpose>0);
}

inline void _llk_unpack_AB_matmul_(
    const std::uint32_t inputA, const std::uint32_t inputB, const std::uint32_t base_address_a, const std::uint32_t base_address_b,
    const std::uint32_t unpA_src_format, const std::uint32_t unpB_src_format, const std::uint32_t tile_index_a, const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim=1, const std::uint32_t rt_dim=1, const std::uint32_t kt_dim=1) {

    // Todo: do something with tile dim flags
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    for (std::uint32_t rt=0; rt<rt_dim; rt++) {
        std::uint32_t offset_address_a = MUL_TILE_SIZE_AND_INDEX(unpA_src_format, (tile_index_a + rt*kt_dim));
        std::uint32_t address_a = base_address_a + offset_address_a;

        for (std::uint32_t ct=0; ct<ct_dim; ct++) {

            std::uint32_t offset_address_b = MUL_TILE_SIZE_AND_INDEX(unpB_src_format, (tile_index_b+ct));
            std::uint32_t address_b = base_address_b + offset_address_b;

            // Clear z/w start counters
            TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

            // Wait for free context
            wait_for_next_context(2);

            // Program srcA and srcB base addresses
            if (0 == unp_cfg_context) {
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_b;
                cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_a;
            } else {
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_b;
                cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_a;
            }

            semaphore_post(semaphore::UNPACK_SYNC);  // Trisc::SEMPOST for context acquire

            // Stall unpacker until pending CFG writes from Trisc have completed
            // TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

            // Run MOP
            mop_run(0, 2);

            // T6::SEMGET for context release
            t6_semaphore_get(semaphore::UNPACK_SYNC);

            // Switch unpacker config context
            switch_config_context(unp_cfg_context);

            #ifdef PERF_DUMP
                first_unpack_recorded = true;
            #endif
        }    
    }    
}