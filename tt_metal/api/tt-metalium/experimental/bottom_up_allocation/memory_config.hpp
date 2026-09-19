// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>

namespace tt::tt_metal::experimental::bottom_up_allocation {

// Select low-to-high address placement for buffers created from this config.
// The default tensor allocation direction remains unchanged when disabled.
bool is_bottom_up_allocation(const MemoryConfig& config);
void set_bottom_up_allocation(MemoryConfig& config, bool enable);

}  // namespace tt::tt_metal::experimental::bottom_up_allocation
