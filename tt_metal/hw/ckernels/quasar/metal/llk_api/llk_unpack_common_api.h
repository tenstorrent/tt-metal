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
#include "llk_param_structs.h"
#include "llk_unpack_common.h"

/*************************************************************************
 * LLK UNPACK COMMON
 *************************************************************************/

// inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
//     const uint32_t unpA_operand_id = get_operand_id(unpA_operand);
//     const uint32_t unpB_operand_id = get_operand_id(unpB_operand);
//     std::uint32_t base_addr_A =
//         get_local_cb_interface(unpA_operand_id).fifo_limit - get_local_cb_interface(unpA_operand_id).fifo_size;
//     std::uint32_t base_addr_B =
//         get_local_cb_interface(unpB_operand_id).fifo_limit - get_local_cb_interface(unpB_operand_id).fifo_size;

//     buffer_descriptor_u bd_val_A = {0};
//     bd_val_A.f.l1_addr_16B = base_addr_A;
//     bd_val_A.f.format = static_cast<uint8_t>(unpack_src_format[unpA_operand_id]);
//     bd_val_A.f.x_dim = get_operand_face_r_dim(unpA_operand_id);
//     bd_val_A.f.y_dim = 16;  // face_c_dim
//     bd_val_A.f.z_dim = get_operand_num_faces(unpA_operand_id);

//     buffer_descriptor_u bd_val_B = {0};
//     bd_val_B.f.l1_addr_16B = base_addr_B;
//     bd_val_B.f.format = static_cast<uint8_t>(unpack_src_format[unpB_operand_id]);
//     bd_val_B.f.x_dim = get_operand_face_r_dim(unpB_operand_id);
//     bd_val_B.f.y_dim = 16;  // face_c_dim
//     bd_val_B.f.z_dim = get_operand_num_faces(unpB_operand_id);

//     tdma_descriptor_t td_val_A, td_val_B;

//     td_val_A.buf_desc = bd_val_A;
//     td_val_A.buf_desc_id = get_operand_id(unpA_operand);
//     td_val_A.reg_data_format = static_cast<uint8_t>(unpack_dst_format[unpA_operand_id]);

//     td_val_B.buf_desc = bd_val_B;
//     td_val_B.buf_desc_id = get_operand_id(unpB_operand);
//     td_val_B.reg_data_format = static_cast<uint8_t>(unpack_dst_format[unpB_operand_id]);

//     _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);
// }

// inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand) {
//     llk_unpack_hw_configure(unpA_operand, unpA_operand);
// }
