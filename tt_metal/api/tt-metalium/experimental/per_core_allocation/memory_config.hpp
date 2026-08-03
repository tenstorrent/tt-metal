// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

// MemoryConfig free functions — access experimental per-core allocation state on MemoryConfigImpl.

bool is_per_core_allocation(const MemoryConfig& config);
void set_per_core_allocation(MemoryConfig& config, bool enable);

}  // namespace tt::tt_metal::experimental::per_core_allocation
