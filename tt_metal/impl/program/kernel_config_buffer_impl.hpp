// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/experimental/kernel_config_buffer.hpp>
#include <tt-metalium/vocabulary/hal_types.hpp>
#include <tt-metalium/vocabulary/mesh_coord.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

namespace tt::tt_metal {

class Buffer;

namespace experimental {

// A kernel config buffer is allocated per device and per core: each device runs its own program's kernel groups, and
// each core gets a shard sized for its kernel group's binaries on that device, at its own address. On one device, the
// cores whose shards are the same size share one per-core-allocated Buffer.
class KernelConfigBufferImpl {
public:
    // One per-core-allocated Buffer: a `size`-byte shard on each of `cores` of one device.
    struct Shards {
        CoreRangeSet cores;
        uint32_t size = 0;
        std::shared_ptr<Buffer> buffer;
    };

    // `shards` on disjoint cores of each device of `mesh_device`. Records the MeshDevice's ID and its local devices'
    // IDs, for addressing by MeshCoordinate, but keeps no reference to it.
    KernelConfigBufferImpl(std::vector<Shards> shards, const distributed::MeshDevice& mesh_device);

    // The largest shard on any device.
    uint32_t max_size() const { return max_size_; }
    // ID of the MeshDevice it is allocated on.
    int mesh_device_id() const { return mesh_device_id_; }

    // The shard on `core` of a device. Throws if there is none.
    uint32_t size(ChipId device_id, const CoreCoord& core) const;
    DeviceAddr address(ChipId device_id, const CoreCoord& core) const;
    uint32_t size(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;
    DeviceAddr address(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;

    // The shards on `core` of `device_id`, or null.
    const Shards* find(ChipId device_id, const CoreCoord& core) const;

private:
    ChipId device_id(const distributed::MeshCoordinate& device_coord) const;
    const Shards& at(ChipId device_id, const CoreCoord& core) const;

    std::vector<Shards> shards_;
    // Device, then core, to its index in shards_.
    std::unordered_map<ChipId, std::unordered_map<CoreCoord, size_t>> shard_map_;
    int mesh_device_id_ = 0;
    // Each local device's MeshCoordinate to its ID.
    std::unordered_map<distributed::MeshCoordinate, ChipId> device_ids_;
    uint32_t max_size_ = 0;
};

}  // namespace experimental
}  // namespace tt::tt_metal
