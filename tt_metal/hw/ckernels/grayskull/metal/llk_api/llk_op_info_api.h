// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

extern uint32_t op_info_offset;

inline void llk_get_next_op_info(tt::op_info_t& op_info_struct) {

    uint32_t* op_info_ptr = reinterpret_cast<uint32_t*>(OP_INFO_BASE_ADDR + op_info_offset);
    static constexpr uint32_t op_info_num_items = 7;

    volatile tt_l1_ptr uint32_t* op_info_struct_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&op_info_struct);
    for (uint32_t i = 0; i < op_info_num_items; i++) {
        op_info_struct_ptr[i] = op_info_ptr[i];
    }
    op_info_offset += 28;

    if (op_info_offset == OP_INFO_SIZE) {
        op_info_offset = 0; // In case we go out of bounds
    }
}
