// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

namespace tt::tt_metal {

class AllocatorImpl;
struct AllocatorConfig;

namespace experimental {

// Factory: creates a MockAllocator wrapped as unique_ptr<AllocatorImpl>.
// Internal — used by Device::initialize_allocator() and the MeshDevice mesh-allocator path
// when the device's MetalContext targets Mock.
std::unique_ptr<AllocatorImpl> make_mock_allocator(const AllocatorConfig& config);

}  // namespace experimental
}  // namespace tt::tt_metal
