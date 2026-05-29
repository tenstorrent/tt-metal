
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_unpack_A_custom.h"
#include "llk_unpack_cb_tile_access.h"
#include "llk_unpack_common_api.h"

inline void llk_unpack_A_custom(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t address = llk_unpack_tile_address(operand_id, tile_index);

    LLK_ASSERT_BLOCK(validate_unpack_tile_access(operand_id, tile_index, 1));

    WAYPOINT("UPAW");
    _llk_unpack_A_custom_(address);
    WAYPOINT("UPAD");
}
