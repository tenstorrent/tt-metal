// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_mop_config() {
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

    if constexpr (BType == BroadcastType::COL) {
        static constexpr uint unpack_srcb_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 2, 0b0001);
        ckernel_unpack_template tmp = ckernel_unpack_template(
            false,  // src B
            true,   // halo - just used for 4 unpacks
            unpack_srcb,
            unpack_srca,
            unpack_srca,
            unpack_srcb_set_z,
            0,
            0,
            0);
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::ROW) {
        static constexpr uint unpack_srcb_clear_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 0, 0b0001);
        ckernel_unpack_template tmp = ckernel_unpack_template(
            true,  // src B
            true,  // halo - just used for 4 unpacks
            unpack_srcb,
            unpack_srca,
            unpack_srcb,
            unpack_srca,
            0,
            unpack_srcb_clear_z,
            0);
        tmp.program(instrn_buffer);
    } else if constexpr (BType == BroadcastType::SCALAR) {
        ckernel_unpack_template tmp = ckernel_unpack_template(
            true,  // src B
            true,  // halo - just used for 4 unpacks
            unpack_srca,
            unpack_srca,
            unpack_srca,
            unpack_srca,
            0,
            unpack_srcb,
            0);
        tmp.program(instrn_buffer);
    } else {
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

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_hw_configure(const llk_unpack_AB_params_t *unpack_AB_params) {
    configure_unpack_AB(get_operand_id(unpack_AB_params->unpA_operand), get_operand_id(unpack_AB_params->unpB_operand));
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {

    const llk_unpack_AB_params_t unpack_AB_params = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};
    llk_unpack_AB_hw_configure<BType>(&unpack_AB_params);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init() {
    llk_unpack_AB_mop_config<BType>();
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    std::uint32_t operandA, std::uint32_t operandB, std::uint32_t tile_index_a, std::uint32_t tile_index_b) {

    std::uint32_t inputA = get_operand_id(operandA);
    std::uint32_t inputB = get_operand_id(operandB);

    std::uint32_t base_address_a = cb_interface[inputA].fifo_rd_ptr;
    std::uint32_t offset_address_a = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputA], tile_index_a);
    //DPRINT << "tia=" << tile_index_a << ENDL();
    //DPRINT << "oaa=" << offset_address_a << ENDL();
    std::uint32_t base_address_b = cb_interface[inputB].fifo_rd_ptr;
    std::uint32_t offset_address_b = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputB], tile_index_b);

    // note: unpacker is programmed to automatically skip the tile header (+1)
    // since there is no tile header, we need to -1 the address (in terms of 16B words), to offet unpacker's automatic +1
    std::uint32_t address_a = base_address_a + offset_address_a - 1;
    std::uint32_t address_b = base_address_b + offset_address_b - 1;

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    // Program srcA and srcB base addresses
    volatile uint *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Get tile address
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }

    // Trisc::SEMPOST for context acquire
    semaphore_post(semaphore::UNPACK_SYNC);

    // Stall unpacker until pending CFG writes from Trisc have completed
    // TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

#ifdef PERF_DUMP
    if (record_perf_events && !first_unpack_recorded) {
        uint32_t event_id_first_unpack = perf::get_event_id(
            0, 0, perf::EventType::UNPACK_FIRST_INSTRUCTION, current_outer_loop_iter);
        record_timestamp_64b(event_id_first_unpack);
        first_unpack_recorded = true;
    }
#endif

    // Run MOP
    if constexpr ((BType == BroadcastType::ROW) || (BType == BroadcastType::COL)) {
        mop_run(0, 2);
    } else if constexpr (BType == BroadcastType::SCALAR) {
        mop_run(0, 1);
    } else {
        mop_run(0, 4);
    }

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
