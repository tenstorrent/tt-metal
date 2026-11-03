// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/allocator/allocation_request.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::allocator {

namespace {

std::vector<NodeCoord> shard_grid_nodes(const ShardSpecBuffer& shard_spec) {
    const bool row_major = shard_spec.tensor_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    return corerange_to_cores(shard_spec.tensor_shard_spec.grid, std::nullopt, row_major);
}

}  // namespace

BankRequest compute_bank_request(
    BufferType buffer_type,
    DeviceAddr size,
    DeviceAddr page_size,
    uint32_t num_banks,
    uint32_t alignment_bytes,
    DeviceAddr interleaved_address_limit,
    const CoreRangeSet& compute_grid,
    std::optional<uint32_t> num_shards) {
    if (num_shards.has_value()) {
        const auto num_compute_banks = compute_grid.num_cores();
        TT_FATAL(
            num_shards.value() <= num_compute_banks,
            "Expected number of shards {} to be less than or equal to total number of L1 banks {} in compute cores",
            num_shards.value(),
            num_compute_banks);
        num_banks = num_shards.value();
    }
    const DeviceAddr size_per_bank =
        tt::tt_metal::detail::calculate_bank_size_spread(size, page_size, num_banks, alignment_bytes);
    DeviceAddr address_limit = 0;
    if (!num_shards.has_value() && buffer_type == BufferType::L1) {
        address_limit = interleaved_address_limit;
        TT_FATAL(address_limit > 0, "Address limit {} needs to be larger than zero.", address_limit);
    }
    return BankRequest{
        .num_banks = num_banks,
        .size_per_bank = size_per_bank,
        .address_limit = address_limit,
    };
}

std::vector<NodeCoord> per_node_allocation_nodes(const std::optional<ShardSpecBuffer>& shard_spec) {
    TT_FATAL(shard_spec.has_value(), "per_core_allocation requires a shard_spec with core grid");
    std::vector<NodeCoord> nodes = shard_grid_nodes(*shard_spec);
    TT_FATAL(!nodes.empty(), "per_core_allocation: shard grid resolved to zero cores");
    return nodes;
}

std::vector<NodeCoord> range_lockstep_allocation_nodes(
    const std::optional<BufferDistributionSpec>& distribution_spec, const std::optional<ShardSpecBuffer>& shard_spec) {
    std::vector<NodeCoord> nodes;
    // A tensor carries both specs, so the order matters: the distribution spec wins, as it
    // does in Buffer::num_cores(). Its cores_with_data() is the set that actually gets an
    // allocation, which the shard grid overstates when the data does not fill it.
    if (distribution_spec.has_value()) {
        nodes = distribution_spec->cores_with_data();
    } else if (shard_spec.has_value()) {
        nodes = shard_grid_nodes(*shard_spec);
    }
    // Scanning nothing is not a safe fallback -- it would place the buffer without avoiding
    // anything. set_range_lockstep_allocation() requires one of the two specs and Buffer
    // carries both through, so this cannot fire, but that invariant now lives in another
    // file and this is where it is relied on.
    TT_FATAL(
        !nodes.empty(),
        "range lockstep resolved to zero cores to scan; the buffer must carry a shard spec or a "
        "distribution spec naming the cores it occupies");
    return nodes;
}

}  // namespace tt::tt_metal::allocator
