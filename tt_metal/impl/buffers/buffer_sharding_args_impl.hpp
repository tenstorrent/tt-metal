// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>

#include <optional>

namespace tt::tt_metal {

class BufferShardingArgsImpl {
public:
    BufferShardingArgsImpl() = default;
    BufferShardingArgsImpl(
        std::optional<BufferDistributionSpec> buffer_distribution_spec,
        std::optional<ShardSpecBuffer> shard_spec,
        TensorMemoryLayout buffer_layout,
        bool per_core_allocation = false) :
        buffer_distribution_spec_(std::move(buffer_distribution_spec)),
        shard_spec_(std::move(shard_spec)),
        buffer_layout_(buffer_layout),
        per_core_allocation_(per_core_allocation) {}

    std::optional<BufferDistributionSpec> buffer_distribution_spec_;
    std::optional<ShardSpecBuffer> shard_spec_;
    TensorMemoryLayout buffer_layout_ = TensorMemoryLayout::INTERLEAVED;
    bool per_core_allocation_ = false;
};

}  // namespace tt::tt_metal
