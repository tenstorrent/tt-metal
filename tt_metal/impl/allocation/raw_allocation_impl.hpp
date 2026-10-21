// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <utility>
#include <variant>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/allocation/raw_allocation_properties.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/buffer.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include <tt_stl/assert.hpp>
#include "buffer.hpp"

namespace tt::tt_metal::experimental {

// Represent lockstep allocation across all banks of a common type,
// with a common starting address and uniform extent.
struct LockstepAllocation {
    DeviceAddr addr;
    LockstepAllocationProperties config;

    auto size() const { return config.per_bank_size; }
};

// // Represent per-node allocation on a subset of available banks,
// // with potentially different starting addresses and a common extent.
// struct PerNodeAllocation {
//     PerNodeAllocation(std::unordered_map<CoreCoord, DeviceAddr> addrs, uint32_t size) :
//         addr_per_node(std::move(addrs)), size(size) {}
//
//     std::unordered_map<CoreCoord, DeviceAddr> addr_per_node;
//     uint32_t size;
// };

// Represent range lockstep allocation on a subset of available banks,
// with a common starting address and uniform extent.
struct RangeLockstepAllocation {
    DeviceAddr addr;
    RangeLockstepAllocationProperties config;

    auto size() const { return config.per_bank_size; }

    auto banks_allocated() const { return config.banks_to_be_allocated_on; }
};

struct RawAllocationImpl {
    using AllocationLayout = std::variant<LockstepAllocation, /* PerNodeAllocation, */ RangeLockstepAllocation>;

    RawAllocationImpl(AllocationLayout detail, distributed::MeshBuffer backing) :
        layout(std::move(detail)), backing(std::move(backing)) {}

    // General move-only RAII resource type:
    RawAllocationImpl() = delete;
    RawAllocationImpl(const RawAllocationImpl&) = delete;
    RawAllocationImpl& operator=(const RawAllocationImpl&) = delete;
    RawAllocationImpl(RawAllocationImpl&&) noexcept = default;
    RawAllocationImpl& operator=(RawAllocationImpl&&) noexcept = default;
    // Release allocation
    ~RawAllocationImpl() = default;

    static RawAllocationImpl allocate_l1(distributed::MeshDevice& device, LockstepAllocationProperties properties) {
        TT_FATAL(properties.per_bank_size > 0, "allocate_l1 requires a non-zero per-bank size");

        auto num_banks = device.allocator()->get_num_banks(BufferType::L1);
        auto total_size = properties.per_bank_size * num_banks;

        auto buffer_ptr = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = total_size},
            distributed::DeviceLocalBufferConfig{
                // It is unsure if this could blow-up in our face when we attempt to allocate a huge page size.
                // We should be fine....? as long as we don't transfer this buffer between host and device.
                //
                // Alignment is handled by the MeshBuffer infrastructure.
                .page_size = properties.per_bank_size,
                .buffer_type = BufferType::L1},
            &device);

        auto addr = buffer_ptr->address();
        return RawAllocationImpl(LockstepAllocation(addr, properties), std::move(*buffer_ptr));
    }

    static RawAllocationImpl allocate_l1(
        distributed::MeshDevice& device, RangeLockstepAllocationProperties properties) {
        TT_FATAL(properties.per_bank_size > 0, "allocate_l1 requires a non-zero per-bank size");

        auto num_banks = properties.banks_to_be_allocated_on.num_cores();
        TT_FATAL(num_banks > 0, "allocate_l1 requires a non-empty bank set");
        const DeviceAddr total_size = DeviceAddr{properties.per_bank_size} * num_banks;

        // One page on each bank
        BufferShardingArgs shardingArgs{BufferDistributionSpec{
            Shape{num_banks, 1},
            Shape{1, 1},
            properties.banks_to_be_allocated_on,
            ShardOrientation::ROW_MAJOR,
        }};
        range_lockstep_allocation::set_range_lockstep_allocation(shardingArgs, true);

        auto buffer_ptr = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = total_size},
            distributed::DeviceLocalBufferConfig{// Same comment as above
                                                 .page_size = properties.per_bank_size,
                                                 .buffer_type = BufferType::L1,
                                                 .sharding_args = std::move(shardingArgs)},
            &device);

        auto addr = buffer_ptr->address();
        return RawAllocationImpl(RangeLockstepAllocation(addr, std::move(properties)), std::move(*buffer_ptr));
    }

    // Describes the allocation layout (kind and their respective attributes)
    AllocationLayout layout;
    // Token to reserve the memory for us,
    // using MeshBuffer in the intrim,
    // in the future MeshBuffer should use RawAllocation instead of the other way around.
    //
    // Invariant:
    // - Always allocated
    // - Always owned
    distributed::MeshBuffer backing;
};

}  // namespace tt::tt_metal::experimental
