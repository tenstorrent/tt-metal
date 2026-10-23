// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal::experimental::fixed_address_allocation {

// Experimental and subject to change: this API carries no stability guarantee.
// Allocate and own a nonempty lockstep L1 or L1_SMALL range at exactly this address.
// The allocator checks alignment and availability, including other allocator reservations.
// Unlike MeshBuffer::create(..., address), this reserves storage and releases it on deallocation.
// Per-core allocation and Light Metal host capture are not supported.
std::shared_ptr<distributed::MeshBuffer> allocate_at_address(
    const distributed::MeshBufferConfig& mesh_buffer_config,
    const distributed::DeviceLocalBufferConfig& device_local_config,
    distributed::MeshDevice* mesh_device,
    DeviceAddr address);

}  // namespace tt::tt_metal::experimental::fixed_address_allocation
