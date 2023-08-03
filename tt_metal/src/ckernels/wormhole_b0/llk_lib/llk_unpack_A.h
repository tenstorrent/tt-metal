#pragma once
#include "llk_io_unpack.h"
#include "llk_param_structs.h"
#include "llk_unpack_common.h"

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool acc_to_dest = false>
inline void llk_unpack_A_mop_config() {

    static_assert((!(acc_to_dest && (BType != BroadcastType::NONE))) && "accumulate into dest with broadcast is not supported!");
    static_assert((!(acc_to_dest && transpose_xy)) && "accumulate into dest with transpose xy is not supported!");
    static_assert((!(transpose_xy && (BType != BroadcastType::NONE))) && "transpose xy with broadcast is not supported!");

    if constexpr (acc_to_dest) {
#if SKIP_UNP0 == 1
        static constexpr uint unpack_srca = TT_OP_NOP;
#else
        static constexpr uint unpack_srca =
            TT_OP_UNPACR_NOP(p_unpacr_nop::UNP0, p_unpacr_nop::UNP_ZEROSRC|p_unpacr_nop::UNP_SET_DVALID);
#endif
#if SKIP_UNP1 == 1
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
        ckernel_unpack_template tmp = ckernel_unpack_template(
            true,   // src B
            false,  // halo - just used for 4 unpacks
            unpack_srca,
            0,
            0,
            0,
            0,
            unpack_srcb,
            0);
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::COL) {
#if SKIP_UNP0 == 1
        static constexpr uint unpack_srca = TT_OP_NOP;
#else
        static constexpr uint unpack_srca =
            TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
#if SKIP_UNP1 == 1
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
        static constexpr uint unpack_srcb_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 2, 0b0001);
        ckernel_unpack_template tmp = ckernel_unpack_template(
            false,  // src B
            true,   // halo - just used for 4 unpacks
            unpack_srcb,
            unpack_srcb_set_z,
            unpack_srcb,
            unpack_srca,
            0,
            0,
            0);
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::ROW) {
#if SKIP_UNP1 == 1
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
        static constexpr uint unpack_srcb_clear_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 0, 0b0001);
        ckernel_unpack_template tmp = ckernel_unpack_template(
            false,  // src B
            true,   // halo - just used for 4 unpacks
            unpack_srcb,
            unpack_srcb,
            unpack_srcb_clear_z,
            TT_OP_NOP,
            0,
            0,
            0);
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::SCALAR) {
#if SKIP_UNP1 == 1
        static constexpr uint unpack_srcb = TT_OP_NOP;
#else
        static constexpr uint unpack_srcb =
            TT_OP_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
        ckernel_unpack_template tmp = ckernel_unpack_template::lB(unpack_srcb, TT_OP_NOP);
        tmp.program(instrn_buffer);
    } else {
        if constexpr (transpose_xy) {
             static constexpr uint unpack_srca_set_z = TT_OP_SETADCZW(0b001, 0, 0, 0, 1, 0b0001);
#if SKIP_UNP0 == 1
            static constexpr uint unpack_srca = TT_OP_NOP;
#else
            static constexpr uint unpack_srca =
                TT_OP_UNPACR(SrcA, 0b01000010, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
#if SKIP_UNP1 == 1
            static constexpr uint unpack_srcb = TT_OP_NOP;
#else
            static constexpr uint unpack_srcb = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP1, p_unpacr_nop::UNP_ZEROSRC|p_unpacr_nop::UNP_SET_DVALID);
#endif
            ckernel_unpack_template tmp = ckernel_unpack_template(
                true,  // src B
                true,   // halo - just used for 4 unpacks
                unpack_srca,
                unpack_srcb,
                unpack_srca,
                unpack_srcb,
                0,
                unpack_srca_set_z,
                0);
            tmp.program(instrn_buffer);
        } else {
#if SKIP_UNP0 == 1
            static constexpr uint unpack_srca = TT_OP_NOP;
#else
            static constexpr uint unpack_srca =
                TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
#if SKIP_UNP1 == 1
            static constexpr uint unpack_srcb = TT_OP_NOP;
#else
            static constexpr uint unpack_srcb = TT_OP_UNPACR_NOP(p_unpacr_nop::UNP1, p_unpacr_nop::UNP_ZEROSRC|p_unpacr_nop::UNP_SET_DVALID);
#endif

            ckernel_unpack_template tmp = ckernel_unpack_template(
                true,   // src B
                false,  // halo - just used for 4 unpacks
                unpack_srca,
                0,
                0,
                0,
                0,
                unpack_srcb,
                0);
            tmp.program(instrn_buffer);
        }
    }
}

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool transpose_xy_srca = false, bool acc_to_dest = false, bool is_fp32_dest_acc_en = false>
inline void llk_unpack_A_hw_configure(const llk_unpack_A_params_t *unpack_A_params) {

    static_assert((!(acc_to_dest && (BType != BroadcastType::NONE))) && "accumulate into dest with broadcast is not supported!");
    static_assert((!(acc_to_dest && transpose_xy)) && "accumulate into dest with transpose xy is not supported!");
    static_assert((!(transpose_xy && (BType != BroadcastType::NONE))) && "transpose xy with broadcast is not supported!");
    static_assert((!(transpose_xy_srca && (BType != BroadcastType::NONE))) && "transpose_xy_srca with broadcast is not supported!");

    constexpr bool is_row_pool = false;

    if constexpr ((BType == BroadcastType::SCALAR) || (BType == BroadcastType::ROW)) {
        constexpr uint32_t srca_height = 1;
        constexpr uint32_t srcb_height = 1;
        configure_unpack_AB(get_operand_id(unpack_A_params->unpA_operand), get_operand_id(unpack_A_params->unpA_operand),
                            srca_height, srcb_height, is_row_pool, transpose_xy_srca, is_fp32_dest_acc_en);
    } else if constexpr ((BType == BroadcastType::COL) || acc_to_dest) {
        // broadcast_type = p_movb2d::MOV_8_ROW_BRCST_D0_BRCST;
        // MOVB2D with column broadcast doesn't work due to the bug in FPU tile
        // which masks dest write enable signals when instrn_mode[1:0] == 2'b01
        // ELTWADD with zeros will be used as a workaround
        // TTI_ZEROSRC(0, 1, 1, 1); // Zero out SrcA
        // TTI_SETDVALID(p_setrwc::SET_A);
        constexpr uint32_t srca_height = 0;
        constexpr uint32_t srcb_height = 16;
        configure_unpack_AB(get_operand_id(unpack_A_params->unpA_operand), get_operand_id(unpack_A_params->unpA_operand),
                            srca_height, srcb_height, is_row_pool, transpose_xy_srca, is_fp32_dest_acc_en);
    } else {
        constexpr uint32_t srca_height = 16;
        constexpr uint32_t srcb_height = 16;
        configure_unpack_AB(get_operand_id(unpack_A_params->unpA_operand), get_operand_id(unpack_A_params->unpA_operand),
                            srca_height, srcb_height, is_row_pool, transpose_xy_srca, is_fp32_dest_acc_en);
    }
}

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool transpose_xy_srca = false, bool acc_to_dest = false, bool is_fp32_dest_acc_en = false>
inline void llk_unpack_A_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_A_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_A_hw_configure<BType, transpose_xy, transpose_xy_srca, acc_to_dest, is_fp32_dest_acc_en>(&unpack_A_params);
}

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool acc_to_dest = false>
inline void llk_unpack_A_init(const std::uint32_t within_face_16x16_transpose=0) {
    llk_unpack_A_mop_config<BType, transpose_xy, acc_to_dest>();
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW, p_gpr_unpack::TMP0, p_gpr_unpack::TMP1>(within_face_16x16_transpose);
}

template <BroadcastType BType = BroadcastType::NONE, bool transpose_xy = false, bool acc_to_dest = false>
inline void llk_unpack_A(std::uint32_t operand, std::uint32_t tile_index) {
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[input].fifo_rd_ptr;
    std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[input], tile_index);
    // note: unpacker is programmed to automatically skip the tile header (+1)
    // since there is no tile header, we need to -1 the address (in terms of 16B words), to offet unpacker's automatic +1
    std::uint32_t address = base_address + offset_address - 1;

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Get tile address
    if (0 == unp_cfg_context) {
        if constexpr ((BType == BroadcastType::NONE) && (!acc_to_dest)) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
        } else {
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address;
        }
    } else {
        if constexpr ((BType == BroadcastType::NONE) && (!acc_to_dest)) {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        } else {
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address;
        }
    }

    // Stall unpacker until pending CFG writes from Trisc have completed
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

#ifdef PERF_DUMP
    if (record_perf_events && !first_unpack_recorded) {
        uint32_t event_id_first_unpack = perf::get_event_id(
            0, 0, perf::EventType::UNPACK_FIRST_INSTRUCTION, current_outer_loop_iter);
        record_timestamp_64b(event_id_first_unpack);
        first_unpack_recorded = true;
    }
#endif

    // Run MOP
     if constexpr ((BType == BroadcastType::ROW) || (transpose_xy)) {
        mop_run(0, 2);
    } else if constexpr ((BType == BroadcastType::SCALAR) || (BType == BroadcastType::COL)) {
        mop_run(0, 1);
    } else {
        mop_run(0, 4);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
