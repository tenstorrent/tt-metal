// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {
class IDevice;

namespace detail {
// Check the complete padded range before creating a view or initializing an addressed semaphore.
void validate_sram_address(IDevice& device, uint64_t address, uint64_t bytes);
}  // namespace detail
}  // namespace tt::tt_metal
