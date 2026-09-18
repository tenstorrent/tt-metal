// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::experimental {
class PinnedMemory;
class ShardDataTransferHelper;
}  // namespace tt::tt_metal::experimental

namespace tt::tt_metal::distributed {

// Specifies host data to be written to or read from a MeshBuffer shard.
class ShardDataTransfer {
private:
    MeshCoordinate shard_coord_;
    // TODO: consider making host data const when it's only read from.
    void* host_data_ = nullptr;
    std::optional<BufferRegion> region_;
    std::shared_ptr<experimental::PinnedMemory> pinned_memory_ = nullptr;
    friend class experimental::ShardDataTransferHelper;

public:
    explicit ShardDataTransfer(const MeshCoordinate& shard_coord) : shard_coord_(shard_coord) {}

    MeshCoordinate shard_coord() const { return shard_coord_; }
    void* host_data() const { return host_data_; }
    std::optional<BufferRegion> region() const { return region_; }

    ShardDataTransfer& shard_coord(const MeshCoordinate& shard_coord) {
        shard_coord_ = shard_coord;
        return *this;
    }
    ShardDataTransfer& host_data(void* host_data) {
        host_data_ = host_data;
        return *this;
    }
    ShardDataTransfer& region(std::optional<BufferRegion> region) {
        region_ = region;
        return *this;
    }
};

}  // namespace tt::tt_metal::distributed
