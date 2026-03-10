
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_unpack_A_custom.h"
#include "llk_unpack_common_api.h"

inline void llk_unpack_A_custom(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;

    WAYPOINT("UPAW");
    _llk_unpack_A_custom_(address);
    WAYPOINT("UPAD");
}
