// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

int find_max_block_size(uint32_t val, uint32_t max_block_size = 8);

}  // namespace tt::tt_metal
