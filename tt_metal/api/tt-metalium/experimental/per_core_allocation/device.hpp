// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <tt_stl/span.hpp>
#include <tt-metalium/experimental/per_core_allocation/allocator_mode.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental::per_core_allocation {

// Initialize a device with a specific allocator mode.
// This is the experimental counterpart of IDevice::initialize() that accepts AllocatorMode.
bool initialize_device(
    IDevice& device,
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_size,
    ttsl::Span<const std::uint32_t> l1_bank_remap = {},
    bool minimal = false,
    AllocatorMode allocator_mode = AllocatorMode::HYBRID);

}  // namespace tt::tt_metal::experimental::per_core_allocation
