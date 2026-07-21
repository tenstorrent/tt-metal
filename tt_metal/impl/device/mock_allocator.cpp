// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>

#include <tt-metalium/experimental/mock_device/mock_allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "impl/allocator/allocator_state.hpp"
#include "impl/allocator/l1_banking_allocator.hpp"
#include "impl/device/mock_allocator.hpp"

namespace tt::tt_metal::experimental {

class MockAllocator : public L1BankingAllocator {
public:
    using L1BankingAllocator::L1BankingAllocator;
};

std::unique_ptr<AllocatorImpl> make_mock_allocator(const AllocatorConfig& config) {
    return std::make_unique<MockAllocator>(config);
}

MockAllocator* get_mock_allocator(distributed::MeshDevice& device) {
    return dynamic_cast<MockAllocator*>(device.allocator_impl().get());
}

struct MockAllocatorState::Impl {
    AllocatorState state;
};

MockAllocatorState::MockAllocatorState() : impl_(std::make_unique<Impl>()) {}
MockAllocatorState::~MockAllocatorState() = default;

MockAllocatorState::MockAllocatorState(const MockAllocatorState& other) : impl_(std::make_unique<Impl>(*other.impl_)) {}

MockAllocatorState::MockAllocatorState(MockAllocatorState&& other) noexcept = default;

MockAllocatorState& MockAllocatorState::operator=(const MockAllocatorState& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

MockAllocatorState& MockAllocatorState::operator=(MockAllocatorState&& other) noexcept = default;

DeviceAddr MockAllocatorState::total_allocated_size() const { return impl_->state.total_allocated_size(); }

DeviceAddr MockAllocatorState::total_allocated_size(BufferType buffer_type) const {
    DeviceAddr total = 0;
    for (const auto& [start, end] : impl_->state.get_allocated_regions(buffer_type)) {
        total += (end - start);
    }
    return total;
}

bool MockAllocatorState::is_empty(BufferType buffer_type) const {
    return impl_->state.get_allocated_regions(buffer_type).empty();
}

std::optional<DeviceAddr> MockAllocatorState::lowest_occupied(BufferType buffer_type) const {
    const auto& regions = impl_->state.get_allocated_regions(buffer_type);
    if (regions.empty()) {
        return std::nullopt;
    }
    return regions.front().first;
}

MockAllocatorState extract_mock_allocator_state(distributed::MeshDevice& device) {
    MockAllocator* allocator = get_mock_allocator(device);
    TT_FATAL(allocator != nullptr, "extract_mock_allocator_state requires a mock device");
    MockAllocatorState wrapper;
    wrapper.impl_->state = allocator->extract_state();
    return wrapper;
}

void override_mock_allocator_state(distributed::MeshDevice& device, const MockAllocatorState& state) {
    MockAllocator* allocator = get_mock_allocator(device);
    TT_FATAL(allocator != nullptr, "override_mock_allocator_state requires a mock device");
    allocator->override_state(state.impl_->state);
}

MockAllocatorState MockAllocatorState::with_allocations(const std::vector<AllocationRecord>& live) const {
    // Replace regions with the live records' intervals.
    auto states = impl_->state.get_states_per_buffer_type();  // copy: config kept, regions reset below
    for (auto& [buffer_type, bts] : states) {
        bts.allocated_regions.clear();
        bts.region_source_allocator_ids.clear();
    }

    for (const auto& record : live) {
        auto it = states.find(record.buffer_type);
        TT_FATAL(
            it != states.end(),
            "with_allocations: state has no bank config for buffer type {}; the template must come from a device "
            "extract (which always carries DRAM/L1/L1_SMALL/TRACE)",
            enchantum::to_string(record.buffer_type));
        it->second.allocated_regions.emplace_back(record.address, record.address + record.size_per_bank);
    }

    for (auto& [buffer_type, bts] : states) {
        bts.normalize();  // canonical form: sort + coalesce, matching a real extract_state
    }

    MockAllocatorState result;
    result.impl_->state = AllocatorState(std::move(states), /*all_allocated_buffers=*/{});
    return result;
}

}  // namespace tt::tt_metal::experimental
