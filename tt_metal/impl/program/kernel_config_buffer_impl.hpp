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
class IDevice;
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

    // `shards` on disjoint cores of each device. `mesh_device` is the MeshDevice the devices belong to, for addressing
    // by MeshCoordinate; null for a buffer launched per device, outside a MeshWorkload. `lockstep_address`: on each
    // device, every shard is at one address and one size, checked when allocated.
    KernelConfigBufferImpl(
        std::vector<Shards> shards, bool lockstep_address, const distributed::MeshDevice* mesh_device = nullptr);

    // The largest shard on any device.
    uint32_t max_size() const { return max_size_; }
    const distributed::MeshDevice* mesh_device() const { return mesh_device_; }

    // The shard on `core` of a device. Throws if there is none.
    uint32_t size(ChipId device_id, const CoreCoord& core) const;
    DeviceAddr address(ChipId device_id, const CoreCoord& core) const;
    uint32_t size(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;
    DeviceAddr address(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const;

    // The shards on `core` of `device_id`, or null.
    const Shards* find(ChipId device_id, const CoreCoord& core) const;
    // Whether the buffer has shards on `device_id`, a device local to this host.
    bool has_shard_on(ChipId device_id) const { return shard_map_.contains(device_id); }
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
    std::unordered_map<ChipId, std::unordered_map<CoreCoord, size_t>> shard_map_;
    const distributed::MeshDevice* mesh_device_ = nullptr;
    uint32_t max_size_ = 0;
    bool lockstep_address_ = false;
};

// CreateKernelConfigBuffer for programs launched per device rather than as a MeshWorkload, such as the fast-dispatch
// command queue programs, which may run before a MeshDevice exists: one program per device, already laid out. No
// slow-dispatch check, since fast dispatch may be active.
std::shared_ptr<KernelConfigBufferImpl> CreateKernelConfigBuffer(
    const std::vector<std::pair<IDevice*, Program*>>& programs, bool lockstep_address);

}  // namespace experimental
}  // namespace tt::tt_metal
