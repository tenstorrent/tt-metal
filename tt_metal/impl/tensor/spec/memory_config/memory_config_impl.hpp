// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {

class MemoryConfigImpl {
public:
    MemoryConfigImpl() = default;

    MemoryConfigImpl(
        TensorMemoryLayout memory_layout,
        BufferType buffer_type,
        std::optional<ShardSpec> shard_spec,
        std::optional<NdShardSpec> nd_shard_spec,
        bool created_with_nd_shard_spec,
        bool per_core_allocation = false) :
        memory_layout_(memory_layout),
        buffer_type_(buffer_type),
        shard_spec_(std::move(shard_spec)),
        nd_shard_spec_(std::move(nd_shard_spec)),
        created_with_nd_shard_spec_(created_with_nd_shard_spec),
        per_core_allocation_(per_core_allocation) {}

    MemoryConfigImpl(const MemoryConfigImpl&) = default;
    MemoryConfigImpl(MemoryConfigImpl&&) noexcept = default;
    MemoryConfigImpl& operator=(const MemoryConfigImpl&) = default;
    MemoryConfigImpl& operator=(MemoryConfigImpl&&) noexcept = default;
    ~MemoryConfigImpl() = default;

    TensorMemoryLayout memory_layout_ = TensorMemoryLayout::INTERLEAVED;
    BufferType buffer_type_ = BufferType::DRAM;
    std::optional<ShardSpec> shard_spec_ = std::nullopt;
    std::optional<NdShardSpec> nd_shard_spec_ = std::nullopt;
    bool created_with_nd_shard_spec_ = false;
    // Experimental: access only via experimental::per_core_allocation free functions.
    bool per_core_allocation_ = false;
};

}  // namespace tt::tt_metal
