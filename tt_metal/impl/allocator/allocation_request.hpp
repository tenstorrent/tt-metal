// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/vocabulary/node_coord.hpp>

namespace tt::tt_metal::allocator {

/// How a request of `size` bytes spreads over the banks of one buffer type.
struct BankRequest {
    /// Banks the buffer spreads over: the shard count when sharded, otherwise every bank.
    uint32_t num_banks = 0;
    DeviceAddr size_per_bank = 0;
    /// Lowest address the request may take; 0 when unlimited. Only interleaved L1 buffers are limited.
    DeviceAddr address_limit = 0;
};

/// Sizes a request of `size` bytes in pages of `page_size` over `num_banks` banks, or over
/// `num_shards` banks when sharded.
BankRequest compute_bank_request(
    BufferType buffer_type,
    DeviceAddr size,
    DeviceAddr page_size,
    uint32_t num_banks,
    uint32_t alignment_bytes,
    DeviceAddr interleaved_address_limit,
    const CoreRangeSet& compute_grid,
    std::optional<uint32_t> num_shards);

/// The nodes of a per-node buffer with `shard_spec`, each of which gets an independent address:
/// every node of its shard grid, in shard orientation order. Throws when there is no shard spec or
/// the grid is empty.
std::vector<NodeCoord> per_node_allocation_nodes(const std::optional<ShardSpecBuffer>& shard_spec);

/// The nodes a range lockstep buffer occupies, whose address only has to be free on them: the
/// nodes with data of `distribution_spec`, otherwise the grid of `shard_spec`. See
/// experimental/range_lockstep_allocation/buffer.hpp. Throws when neither names any nodes.
std::vector<NodeCoord> range_lockstep_allocation_nodes(
    const std::optional<BufferDistributionSpec>& distribution_spec, const std::optional<ShardSpecBuffer>& shard_spec);

}  // namespace tt::tt_metal::allocator
