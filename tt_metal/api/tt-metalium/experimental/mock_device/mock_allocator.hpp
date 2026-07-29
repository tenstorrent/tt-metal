// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * When a device is in mock mode, its allocator is a MockAllocator (derived from
 * L1BankingAllocator) that supports checkpoint/restore of allocation state.
 *
 * The MockAllocator is created automatically during mock device initialization.
 * Use extract_mock_allocator_state() and override_mock_allocator_state() to
 * snapshot and restore allocation state across queries. get_mock_allocator()
 * returns a non-null handle when the device is in mock mode and is intended
 * as an opaque "is this a mock device" probe — MockAllocator itself is only
 * forward-declared here.
 *
 * MockAllocatorState is an opaque value type. The underlying allocator state
 * representation lives in tt_metal/impl/ and is not part of the public API.
 */

#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

class MockAllocator;
class MockAllocatorState;

// Identifies a single allocation within a MockAllocatorState.
struct AllocationRecord {
    BufferType buffer_type = BufferType::L1;
    DeviceAddr address = 0;
    DeviceAddr size_per_bank = 0;
};

// Returns MockAllocator* if device is in mock mode, nullptr otherwise.
MockAllocator* get_mock_allocator(distributed::MeshDevice& device);

// Opaque, caller-owned snapshot of a MockAllocator's allocation state.
// Copy/move enabled so callers can checkpoint and restore.
class MockAllocatorState {
public:
    MockAllocatorState();
    ~MockAllocatorState();
    MockAllocatorState(const MockAllocatorState& other);
    MockAllocatorState(MockAllocatorState&& other) noexcept;
    MockAllocatorState& operator=(const MockAllocatorState& other);
    MockAllocatorState& operator=(MockAllocatorState&& other) noexcept;

    DeviceAddr total_allocated_size() const;
    DeviceAddr total_allocated_size(BufferType buffer_type) const;
    bool is_empty(BufferType buffer_type) const;
    std::optional<DeviceAddr> lowest_occupied(BufferType buffer_type) const;

    // Return a state with this state's allocator (bank) config but its allocations replaced by `live`.
    MockAllocatorState with_allocations(const std::vector<AllocationRecord>& live) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend MockAllocatorState extract_mock_allocator_state(distributed::MeshDevice& device);
    friend void override_mock_allocator_state(distributed::MeshDevice& device, const MockAllocatorState& state);
};

// Extract allocator state snapshot from a mock device.
MockAllocatorState extract_mock_allocator_state(distributed::MeshDevice& device);

// Restore allocator to a previous state snapshot.
void override_mock_allocator_state(distributed::MeshDevice& device, const MockAllocatorState& state);

}  // namespace tt::tt_metal::experimental
