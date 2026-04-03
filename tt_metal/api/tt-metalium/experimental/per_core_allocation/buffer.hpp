// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

// Buffer free functions — friended by Buffer to access private per-core state.

bool is_per_core_allocation(const Buffer& buffer);
DeviceAddr get_per_core_address(const Buffer& buffer, CoreCoord core);
const std::unordered_map<CoreCoord, DeviceAddr>& get_per_core_addresses(const Buffer& buffer);
void copy_per_core_addresses(Buffer& dst, const Buffer& src);

// BufferShardingArgs free functions.

BufferShardingArgs& set_per_core_allocation(BufferShardingArgs& args, bool enable);
bool is_per_core_allocation(const BufferShardingArgs& args);

}  // namespace tt::tt_metal::experimental::per_core_allocation
