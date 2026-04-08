// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file mock_allocator.hpp
 * @brief Experimental allocator state management for MockDevice
 *
 * When a device is in mock mode, its allocator is a MockAllocator (derived from
 * L1BankingAllocator) that supports checkpoint/restore of allocation state.
 * This enables op-by-op memory planning workflows.
 *
 * The MockAllocator is created automatically during mock device initialization.
 * Use get_mock_allocator() to access it, then call extract/override state.
 */

#include <memory>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/allocator_state.hpp>

namespace tt::tt_metal {

struct AllocatorConfig;

namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

class MockAllocator;

// Factory: creates a MockAllocator wrapped as unique_ptr<AllocatorImpl>.
// Used internally by Device::initialize_allocator() for mock devices.
std::unique_ptr<AllocatorImpl> make_mock_allocator(const AllocatorConfig& config);

// Returns MockAllocator* if device is in mock mode, nullptr otherwise.
MockAllocator* get_mock_allocator(distributed::MeshDevice* device);

// Extract allocator state snapshot from a mock device.
// TT_FATAL if device is not in mock mode.
AllocatorState extract_mock_allocator_state(distributed::MeshDevice* device);

// Restore allocator to a previous state snapshot.
// TT_FATAL if device is not in mock mode.
void override_mock_allocator_state(distributed::MeshDevice* device, const AllocatorState& state);

}  // namespace tt::tt_metal::experimental
