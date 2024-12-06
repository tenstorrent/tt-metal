// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal::hw_abstract {

/**
 * @brief Uses the hardware abstraction layer to inform client of architecture specific L1 Size
 *
 * @return Size in bytes of the L1 SRAM buffer associated with the currently present architecture.
 */
uint32_t get_l1_size();

}  // namespace tt::tt_metal::hw_abstract
