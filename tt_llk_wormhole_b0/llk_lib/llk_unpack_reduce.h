
#include "llk_io_unpack.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_mop_config() {
#if SKIP_UNP == 1
    static constexpr uint unpack_srca = TT_OP_NOP;
#else
    static constexpr uint unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
    static constexpr uint unpack_zerosrca = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_ZEROSRC);
#if SKIP_UNP == 1
    static constexpr uint unpack_srcb = TT_OP_NOP;
#else
    static constexpr uint unpack_srcb =
        TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,  // src B
        true,  // halo - just used for 4 unpacks
        unpack_zerosrca,
        unpack_srca,
        TT_OP_NOP,
        TT_OP_NOP,
        0,
        unpack_srcb,
        0);
    tmp.program(instrn_buffer);
}

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en = false, StochRndMode stoch_rnd_mode = StochRndMode::None>
inline void llk_unpack_reduce_hw_configure(
    const llk_unpack_reduce_params_t *unpack_reduce_params, const float const_mult) {

    constexpr uint32_t srca_height = 16;
    constexpr uint32_t srcb_height = 16;
    constexpr bool is_row_pool = true;
    constexpr bool transpose_xy_per_face = (ReduceDim::REDUCE_ROW == dim);

    configure_unpack_AB<is_row_pool, is_fp32_dest_acc_en, stoch_rnd_mode>(
        get_operand_id(unpack_reduce_params->unpA_operand),
        get_operand_id(unpack_reduce_params->unpA_operand),
        srca_height,
        srcb_height,
        transpose_xy_per_face);

    if constexpr (type != PoolType::MAX) {
        union {
            float f;
            uint32_t u;
        } f2u = {.f = const_mult};

        for (uint i = 0; i < 16; i++) l1_buffer[i] = f2u.u;  // Load const into L1 buffer
    }    
}

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en=false, StochRndMode stoch_rnd_mode = StochRndMode::None>
inline void llk_unpack_reduce_hw_configure_disaggregated(const std::uint32_t unpA_operand, const float mult) {
    TT_LLK_DUMP("llk_unpack_reduce_hw_configure_disaggregated<{}, {}, {}, {}>({}, {})", type, dim, is_fp32_dest_acc_en, (uint8_t)stoch_rnd_mode, unpA_operand, mult);
    const llk_unpack_reduce_params_t unpack_reduce_params = {.unpA_operand = unpA_operand};
    llk_unpack_reduce_hw_configure<type, dim, is_fp32_dest_acc_en, stoch_rnd_mode>(&unpack_reduce_params, mult);
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_init(const std::uint32_t within_face_16x16_transpose=0) {
    TT_LLK_DUMP("llk_unpack_reduce_init<{}, {}>({})", type, dim, within_face_16x16_transpose);
    llk_unpack_reduce_mop_config<type, dim>();
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    const uint unpack_src_df  = (uint) DataFormat::Float32;

    const uint unpack_dst_df = ((uint)unpack_dst_format[0] == (uint) DataFormat::Int8) ? (uint) DataFormat::Float16 : // Int8 is treated as fp16_a 
                               ((((uint)unpack_dst_format[0]>>2)&0x1) ? (uint) DataFormat::Float16_b : (uint) DataFormat::Float16);

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(unpack_dst_df);

    // REDUCE_ROW requires transpose itself; additionaly, within_face_16x16_transpose flag could require transpose;
    // if we have the flag set with REDUCE_ROW, we don't need to do anything
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(ReduceDim::REDUCE_ROW == dim ? !within_face_16x16_transpose : within_face_16x16_transpose);

    TTI_SETADCXX(0b11, FACE_WIDTH*FACE_HEIGHT-1, 0x0);

    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0xf>(unpack_src_df);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpack_dst_df);

    TTI_WRCFG(p_gpr_unpack::L1_BUFFER_ADDR, p_cfg::WRCFG_32b, THCON_SEC1_REG3_Base_address_ADDR32);
    TTI_WRCFG(p_gpr_unpack::L1_BUFFER_ADDR, p_cfg::WRCFG_32b, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
    TTI_NOP; TTI_NOP;
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce(const std::uint32_t operand, const std::uint32_t tile_index) {
    TT_LLK_DUMP("llk_unpack_reduce<{}, {}>({}, {})", type, dim, operand, tile_index);
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = operands[input].f.fifo_rd_ptr;
    std::uint32_t offset_address = operands[input].f.tile_size_words * tile_index;
    std::uint32_t address = base_address + offset_address;

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Load only 16 datums into srcB
    TTI_SETADCXX(p_setadc::UNP1, DATUMS_PER_ROW-1, 0x0);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Get tile address
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    }

    // Run MOP
    mop_run(0, 4);

    // Restore face height
    TTI_SETADCXX(p_setadc::UNP1, FACE_HEIGHT*16-1, 0x0);  

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);

#ifdef PERF_DUMP
    first_unpack_recorded = true;
#endif
}
