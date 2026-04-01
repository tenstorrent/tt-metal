// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/device.hpp>
#include "device/device_impl.hpp"

namespace tt::tt_metal::experimental::per_core_allocation {

bool initialize_device(
    IDevice& device,
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_size,
    ttsl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal,
    AllocatorMode allocator_mode) {
    auto& dev = dynamic_cast<class Device&>(device);
    return dev.initialize(
        num_hw_cqs, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap, minimal, allocator_mode);
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
