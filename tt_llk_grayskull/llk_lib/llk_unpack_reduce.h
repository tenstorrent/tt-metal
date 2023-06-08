
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
    static constexpr uint unpack_zerosrca = TT_OP_UNPACR_NOP(SrcA, p_unpacr::UNP_ZEROSRC);
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

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_hw_configure(
    const llk_unpack_reduce_params_t *unpack_reduce_params, const float const_mult) {
    configure_unpack_AB(
        get_operand_id(unpack_reduce_params->unpA_operand),
        get_operand_id(unpack_reduce_params->unpA_operand),
        16,16,true);

    if constexpr (type != PoolType::MAX) {
        union {
            float f;
            uint32_t u;
        } f2u = {.f = const_mult};

        for (uint i = 0; i < 16; i++) l1_buffer[i] = f2u.u;  // Load const into L1 buffer
    }    
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_hw_configure_disaggregated(const std::uint32_t unpA_operand, const float mult) {
    const llk_unpack_reduce_params_t unpack_reduce_params = {.unpA_operand = unpA_operand};
    llk_unpack_reduce_hw_configure<type, dim>(&unpack_reduce_params, mult);
}

template <PoolType type, ReduceDim dim>
// within_face_16x16_transpose is used on WH but not used for GS, this transpose is done in math on GS
inline void llk_unpack_reduce_init(const std::uint32_t within_face_16x16_transpose=0) {
    llk_unpack_reduce_mop_config<type, dim>();
    
    volatile uint tt_reg_ptr *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Set first 32 bites of tile descriptor, only need data format change
    unpack_tile_descriptor_u tile_descriptor = {0};

    tile_descriptor.f.in_data_format  = (uint) DataFormat::Float32;
    tile_descriptor.f.uncompressed = 1; // Input tile is uncompressed
    tile_descriptor.f.x_dim        = 256; 

    unpack_config_u config = {0};

    config.f.out_data_format = (((uint)unpack_dst_format[0]>>2)&0x1) ? (uint) DataFormat::Float16_b : (uint) DataFormat::Float16;
    config.f.throttle_mode = 2;

    wait_for_idle();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);

    uint32_t alu_config_data = gl_alu_format_spec_reg;

    gl_alu_format_spec_reg = cfg_rmw_mmio_rd_tensix_wr(ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, ALU_FORMAT_SPEC_REG1_SrcB_SHAMT, ALU_FORMAT_SPEC_REG1_SrcB_MASK, 
                                                        config.f.out_data_format, 
                                                        alu_config_data);

    cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32] = tile_descriptor.val[0];
    cfg[THCON_SEC1_REG2_Out_data_format_ADDR32] = config.val[0];

    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = (((uint)l1_buffer) >> 4) - 1;        // Set l1 buffer address
    cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = (((uint)l1_buffer) >> 4) - 1;  // Set l1 buffer address
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = operands[input].f.fifo_rd_ptr;
    std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[input], tile_index);
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
