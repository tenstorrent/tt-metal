// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>

namespace tt::tt_metal::experimental::range_lockstep_allocation {

// Experimental and subject to change: this header carries no API-stability guarantee.
//
// MemoryConfig free functions — access range lockstep state on MemoryConfigImpl.
//
// See experimental/range_lockstep_allocation/buffer.hpp for what the mode means. Setting it here
// is how a tensor carries the request down to buffer allocation.

bool is_range_lockstep_allocation(const MemoryConfig& config);
void set_range_lockstep_allocation(MemoryConfig& config, bool enable);

}  // namespace tt::tt_metal::experimental::range_lockstep_allocation
