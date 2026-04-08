// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/mock_allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "impl/allocator/l1_banking_allocator.hpp"

#include <unordered_set>

namespace tt::tt_metal::experimental {

// AllocatorImpl has no virtual methods, so dynamic_cast is not available.
// Track MockAllocator instances via a static registry to enable safe downcasting.
static std::unordered_set<void*> s_mock_allocator_registry;

class MockAllocator : public L1BankingAllocator {
public:
    using L1BankingAllocator::L1BankingAllocator;
};

std::unique_ptr<AllocatorImpl> make_mock_allocator(const AllocatorConfig& config) {
    auto alloc = std::make_unique<MockAllocator>(config);
    s_mock_allocator_registry.insert(alloc.get());
    return alloc;
}

MockAllocator* get_mock_allocator(distributed::MeshDevice* device) {
    auto* impl = device->allocator_impl().get();
    if (s_mock_allocator_registry.count(impl)) {
        return static_cast<MockAllocator*>(impl);
    }
    return nullptr;
}

AllocatorState extract_mock_allocator_state(distributed::MeshDevice* device) {
    auto* mock = get_mock_allocator(device);
    TT_FATAL(mock != nullptr, "extract_mock_allocator_state requires a mock device");
    return mock->extract_state();
}

void override_mock_allocator_state(distributed::MeshDevice* device, const AllocatorState& state) {
    auto* mock = get_mock_allocator(device);
    TT_FATAL(mock != nullptr, "override_mock_allocator_state requires a mock device");
    mock->override_state(state);
}

}  // namespace tt::tt_metal::experimental
