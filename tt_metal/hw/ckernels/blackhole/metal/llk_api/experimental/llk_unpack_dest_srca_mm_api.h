// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_unpack_dest_srca_mm.h"
#include "llk_unpack_common_api.h"

// Unpack side of DEST_SRCA_MM (experimental/llk_unpack_dest_srca_mm.h): SrcA handed to the math thread, in0 to SrcB.

inline void llk_unpack_dest_srca_mm_init(const std::uint32_t in0_operand) {
    _llk_unpack_dest_srca_mm_init_(get_operand_num_faces(get_operand_id(in0_operand)));
}

inline void llk_unpack_dest_srca_mm(const std::uint32_t in0_operand, const std::uint32_t in0_tile_index) {
    const std::uint32_t operand_id = get_operand_id(in0_operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    _llk_unpack_dest_srca_mm_(base_address + get_local_cb_interface(operand_id).fifo_page_size * in0_tile_index);
}
