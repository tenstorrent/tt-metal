// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/kernel_config_buffer.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

namespace tt::tt_metal {

class Buffer;
class Program;

namespace experimental {

// A kernel config buffer is allocated per device and per core: each device runs its own program's kernel groups, and
// each core gets a shard sized for its kernel group's block on that device, at its own address. On one device, the
// cores whose blocks are the same size share one per-core-allocated Buffer.
class KernelConfigBufferImpl {
public:
    // One per-core-allocated Buffer: a `size`-byte shard on each of `cores` of one device.
    struct Shards {
        CoreRangeSet cores;
        uint32_t size = 0;
        std::shared_ptr<Buffer> buffer;
    };

    // `shards` on disjoint cores of each device of `mesh_device`. `lockstep_address`: on each device, every shard is at
    // one address and one size, checked when allocated.
    KernelConfigBufferImpl(
        std::vector<Shards> shards, bool lockstep_address, const distributed::MeshDevice* mesh_device = nullptr);

    // Tensix cores with a shard, on any device.
    const CoreRangeSet& cores() const { return cores_; }
    // The largest shard on any device.
    uint32_t size() const { return size_; }
    const distributed::MeshDevice* mesh_device() const { return mesh_device_; }

    // The shard on `core` of a device. Throws if there is none.
    uint32_t size(ChipId device_id, const CoreCoord& core) const;
    DeviceAddr address(ChipId device_id, const CoreCoord& core) const;
    uint32_t size(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;
    DeviceAddr address(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;

    // The shards on `core` of `device_id`, or null.
    const Shards* find(ChipId device_id, const CoreCoord& core) const;
    // Whether the buffer has shards on `device_id`, a device local to this host.
    bool has_shard_on(ChipId device_id) const { return shards_on_device_.contains(device_id); }
    bool is_allocated() const;
    void deallocate();
    // Whether, on each device, every shard is at one address. Only such a buffer runs a program with Tensix program
    // semaphores: a kernel signals a peer's semaphore at its own semaphore's address.
    bool lockstep_address() const { return lockstep_address_; }

private:
    ChipId device_id(const distributed::MeshCoordinate& device_coord) const;
    const Shards& at(ChipId device_id, const CoreCoord& core) const;

    std::vector<Shards> shards_;
    // Device, then core, to its index in shards_.
    std::unordered_map<ChipId, std::unordered_map<CoreCoord, size_t>> shards_on_device_;
    std::unordered_map<distributed::MeshCoordinate, ChipId> device_ids_;
    const distributed::MeshDevice* mesh_device_ = nullptr;
    CoreRangeSet cores_;
    uint32_t size_ = 0;
    bool lockstep_address_ = false;
};

// Checks that `program`, laid out, can run from `buffer` on `device_id`: each kernel group's block fits in the shard on
// each of its cores there, and the program has no Tensix program semaphores unless the buffer has a lockstep address.
void ValidateKernelConfigBuffer(Program& program, const KernelConfigBufferImpl& buffer, ChipId device_id);

}  // namespace experimental
}  // namespace tt::tt_metal
