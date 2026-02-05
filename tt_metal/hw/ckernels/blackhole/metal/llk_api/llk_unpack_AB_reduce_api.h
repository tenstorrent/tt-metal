// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB REDUCE
 *************************************************************************/

template <PoolType pool_type, ReduceDim reduce_dim>
inline void llk_unpack_AB_reduce_mop_config(const std::uint32_t operand_id = 0) {
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    _llk_unpack_AB_reduce_mop_config_<pool_type, reduce_dim>(face_r_dim, num_faces);
}

template <PoolType pool_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
inline void llk_unpack_AB_reduce_init(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);  // face_r_d for srcA
    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);

    if constexpr (enforce_fp32_accumulation) {
        // Set necessary config regs for MOVB2D hi16/lo16 to work
        _llk_unpack_dbg_feature_disable_();
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    }
    _llk_unpack_AB_reduce_init_<pool_type, reduce_dim>(face_r_dim, num_faces);
}

template <PoolType pool_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
inline void llk_unpack_AB_reduce(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_a = get_local_cb_interface(operandA_id).fifo_page_size * tile_index_a;
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_b = get_local_cb_interface(operandB_id).fifo_page_size * tile_index_b;
    std::uint32_t address_b = base_address_b + offset_address_b;

    WAYPOINT("UABW");
    _llk_unpack_AB_reduce_<pool_type, reduce_dim>(address_a, address_b);
    WAYPOINT("UABD");
}
