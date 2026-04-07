// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/mock_allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "impl/allocator/l1_banking_allocator.hpp"

namespace tt::tt_metal::experimental {

class MockAllocator : public L1BankingAllocator {
public:
    using L1BankingAllocator::L1BankingAllocator;
};

MockAllocator* get_mock_allocator(distributed::MeshDevice* device) {
    auto* impl = device->allocator_impl().get();
    return dynamic_cast<MockAllocator*>(impl);
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
