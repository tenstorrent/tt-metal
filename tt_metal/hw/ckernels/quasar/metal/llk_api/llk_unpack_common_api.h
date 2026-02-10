// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "internal/circular_buffer_interface.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "api/debug/waypoint.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_unpack_common.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

/**
 * @brief Programs l1 info & source register format for both UNP_A and UNP_B
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 */
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
    const std::uint32_t unpA_operand_id = get_operand_id(unpA_operand);
    const std::uint32_t unpB_operand_id = get_operand_id(unpB_operand);

    // Program buffer descriptors for all 32 circular buffers
    for (std::uint32_t i = 0; i < NUM_CIRCULAR_BUFFERS; ++i) {
        const DataFormat l1_data_format = static_cast<DataFormat>(unpack_src_format[i]);

        if (l1_data_format == DataFormat::Invalid) {
            continue;
        }

        buffer_descriptor_u bd_val = {0};
        bd_val.f.l1_addr_16B = get_local_cb_interface(i).fifo_limit - get_local_cb_interface(i).fifo_size;
        bd_val.f.format = static_cast<std::uint8_t>(l1_data_format);
        bd_val.f.x_dim = unpack_tile_face_r_dim[i];
        bd_val.f.y_dim = ckernel::trisc::FACE_C_DIM;
        bd_val.f.z_dim = unpack_tile_num_faces[i];

        ckernel::trisc::_configure_buf_desc_table_(i, bd_val);
    }

    tdma_descriptor_t td_val_A, td_val_B;
    td_val_A.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format[unpA_operand_id]);
    td_val_B.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format[unpB_operand_id]);

    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
}

/**
 * @brief Programs l1 info & source register format for UNP_A
 *
 * @param operandA: The input operand circular buffer
 */
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand) {
    llk_unpack_hw_configure(unpA_operand, unpA_operand);
}
